"""Concurrency control for AI backend requests.

Three independent mechanisms:

AISemaphore
    Caps the number of simultaneous AI backend calls across all endpoints.
    Excess requests are rejected immediately (non-blocking) with a 429.

BuildTracker
    Two guards keyed on build ID:

    * in-flight set  — a build that is currently being processed cannot be
      started again.  The second request gets 409 "build_already_processing".
      The slot is released (via a context manager) when the work finishes,
      regardless of success or failure.

    * completed set  — once a build has received a successful AI analysis the
      BFF records its ID permanently (for the lifetime of the process).  Any
      future request for the same build ID is rejected with 409
      "build_already_analyzed".  This prevents replay / curl simulation abuse
      even after the original request has finished.
"""

import asyncio
import logging

logger = logging.getLogger("az_ai_bff.concurrency")


# ---------------------------------------------------------------------------
# AISemaphore
# ---------------------------------------------------------------------------

class _Slot:
    """Returned by ``AISemaphore.acquire()``.

    Use as an async context manager — the semaphore is released on exit only
    when the slot was actually granted (``ok is True``).
    """

    __slots__ = ("ok", "_sem")

    def __init__(self, ok: bool, sem: asyncio.Semaphore | None) -> None:
        self.ok = ok
        self._sem = sem

    async def __aenter__(self) -> "_Slot":
        return self

    async def __aexit__(self, *_) -> None:
        if self.ok and self._sem is not None:
            self._sem.release()


class AISemaphore:
    """Concurrency cap for AI backend calls.

    Parameters
    ----------
    max_concurrent:
        Maximum number of simultaneous AI backend requests.
        ``0`` means unlimited — no semaphore is created.
    """

    def __init__(self, max_concurrent: int) -> None:
        self._max = max_concurrent
        self._sem: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        )

    @property
    def max_concurrent(self) -> int:
        return self._max

    async def acquire(self) -> _Slot:
        """Try to claim a slot without waiting.

        Returns immediately.  Check ``.ok`` on the returned slot before
        proceeding; use it as an ``async with`` context manager to guarantee
        the slot is released.
        """
        if self._sem is None:
            return _Slot(ok=True, sem=None)

        # _value is the number of free slots remaining.
        if self._sem._value > 0:
            await self._sem.acquire()
            logger.debug(
                "ai_semaphore_acquired",
                extra={"remaining": self._sem._value, "max": self._max},
            )
            return _Slot(ok=True, sem=self._sem)

        logger.warning(
            "ai_semaphore_full",
            extra={"max_concurrent": self._max},
        )
        return _Slot(ok=False, sem=None)


# ---------------------------------------------------------------------------
# BuildTracker
# ---------------------------------------------------------------------------

class _BuildSlot:
    """Returned by ``BuildTracker.try_start()``.

    ``ok``      — False means the build is already in-flight or already done.
    ``reason``  — error code string when ``ok`` is False.

    Use as a context manager around the entire analyze pipeline.  On exit it
    removes the build from the in-flight set.  Call ``mark_done()`` when the
    AI analysis completes successfully to move it to the completed set.
    """

    __slots__ = ("ok", "reason", "_tracker", "_build_id", "_done")

    def __init__(
        self,
        ok: bool,
        reason: str,
        tracker: "BuildTracker",
        build_id: str,
    ) -> None:
        self.ok = ok
        self.reason = reason
        self._tracker = tracker
        self._build_id = build_id
        self._done = False

    def mark_done(self) -> None:
        """Call after a successful AI analysis to record the build as completed."""
        if self.ok:
            self._tracker._mark_completed(self._build_id)
            self._done = True

    def __enter__(self) -> "_BuildSlot":
        return self

    def __exit__(self, *_) -> None:
        if self.ok:
            self._tracker._release(self._build_id)


class BuildTracker:
    """Tracks build IDs that are in-flight or have already been analyzed.

    Thread-safety note: asyncio runs on a single thread; plain set operations
    are atomic enough here.  No lock is needed.
    """

    def __init__(self) -> None:
        self._in_flight: set[str] = set()
        self._completed: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def try_start(self, build_id: str) -> _BuildSlot:
        """Attempt to register *build_id* as in-flight.

        Returns a :class:`_BuildSlot` whose ``.ok`` is True when the
        registration succeeded.  ``ok=False`` means either the build is
        currently being processed or it has already been analyzed.
        """
        if build_id in self._completed:
            logger.warning(
                "build_already_analyzed",
                extra={"build_id": build_id},
            )
            return _BuildSlot(ok=False, reason="build_already_analyzed", tracker=self, build_id=build_id)

        if build_id in self._in_flight:
            logger.warning(
                "build_already_processing",
                extra={"build_id": build_id},
            )
            return _BuildSlot(ok=False, reason="build_already_processing", tracker=self, build_id=build_id)

        self._in_flight.add(build_id)
        logger.debug("build_started", extra={"build_id": build_id})
        return _BuildSlot(ok=True, reason="", tracker=self, build_id=build_id)

    # ------------------------------------------------------------------
    # Internal helpers called by _BuildSlot
    # ------------------------------------------------------------------

    def _release(self, build_id: str) -> None:
        self._in_flight.discard(build_id)

    def _mark_completed(self, build_id: str) -> None:
        self._in_flight.discard(build_id)
        self._completed.add(build_id)
        logger.info("build_analysis_completed", extra={"build_id": build_id})

    # ------------------------------------------------------------------
    # Inspection (tests / health endpoint)
    # ------------------------------------------------------------------

    @property
    def in_flight(self) -> frozenset[str]:
        return frozenset(self._in_flight)

    @property
    def completed(self) -> frozenset[str]:
        return frozenset(self._completed)
