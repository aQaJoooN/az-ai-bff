"""Concurrency control for AI backend requests.

Two independent mechanisms are applied in order inside the analyze route:

AISemaphore
    Caps the number of simultaneous AI backend calls across all endpoints.
    Excess requests are rejected immediately (non-blocking) with 429 so there
    is never a hidden queue of slow builds backing up.

PipelineInFlightTracker
    In-memory guard keyed on the pipeline *definition* ID.  While any run of a
    given pipeline is being analyzed, a second request for the same pipeline is
    rejected with 409 ``pipeline_already_processing``.  The lock is always
    released on exit — errors allow retries.

    Keying on definition ID (not run ID) prevents two parallel runs of the same
    pipeline from racing to write the ``ai-analyzed`` tag or flooding the AI.

The completed-run idempotency check lives in ``AzureDevOpsLogFetcher``
(``analyze.py``), which reads and writes build tags directly on Azure DevOps so
the state is durable across app restarts and shared across all BFF instances.
"""

import asyncio
import logging

logger = logging.getLogger("az_ai_bff.concurrency")


# ---------------------------------------------------------------------------
# AISemaphore
# ---------------------------------------------------------------------------

class _Slot:
    """Context manager returned by ``AISemaphore.acquire()``.

    Releases the semaphore on exit only when the slot was actually granted
    (``ok is True``).  A rejected slot's ``__aexit__`` is a no-op.
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
    """Non-blocking concurrency cap for AI backend calls.

    Parameters
    ----------
    max_concurrent:
        Maximum simultaneous AI requests.  ``0`` means unlimited.
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

        Returns immediately.  Check ``.ok``; use the slot as an ``async with``
        context manager to guarantee release.
        """
        if self._sem is None:
            return _Slot(ok=True, sem=None)

        # _value is the internal free-slot counter of asyncio.Semaphore.
        # We read it to avoid waiting — if zero, reject the request outright.
        if self._sem._value > 0:
            await self._sem.acquire()
            logger.debug(
                "ai_semaphore_acquired",
                extra={"remaining": self._sem._value, "max": self._max},
            )
            return _Slot(ok=True, sem=self._sem)

        logger.warning("ai_semaphore_full", extra={"max_concurrent": self._max})
        return _Slot(ok=False, sem=None)


# ---------------------------------------------------------------------------
# PipelineInFlightTracker
# ---------------------------------------------------------------------------

class _PipelineSlot:
    """Context manager returned by ``PipelineInFlightTracker.try_start()``.

    ``ok``          — False if this pipeline is already in-flight.
    ``reason``      — error code string when ``ok`` is False.
    ``pipeline_id`` — the definition ID that was registered.

    Use as a synchronous ``with`` context manager.  The lock is released on
    exit regardless of whether the analysis succeeded or failed.
    """

    __slots__ = ("ok", "reason", "pipeline_id", "_tracker")

    def __init__(
        self,
        ok: bool,
        reason: str,
        pipeline_id: str,
        tracker: "PipelineInFlightTracker",
    ) -> None:
        self.ok = ok
        self.reason = reason
        self.pipeline_id = pipeline_id
        self._tracker = tracker

    def __enter__(self) -> "_PipelineSlot":
        return self

    def __exit__(self, *_) -> None:
        if self.ok:
            self._tracker._release(self.pipeline_id)


class PipelineInFlightTracker:
    """Prevents two runs of the same pipeline definition from being analyzed
    simultaneously.

    State is in-memory only — a restart clears all locks.  This is correct
    because an in-flight analysis cannot survive a restart anyway.
    """

    def __init__(self) -> None:
        self._in_flight: set[str] = set()

    def try_start(self, pipeline_id: str) -> _PipelineSlot:
        """Attempt to register *pipeline_id* as in-flight.

        Returns a slot with ``ok=True`` on success, or ``ok=False`` if the
        pipeline is already being processed.
        """
        if pipeline_id in self._in_flight:
            logger.warning("pipeline_already_processing", extra={"pipeline_id": pipeline_id})
            return _PipelineSlot(
                ok=False,
                reason="pipeline_already_processing",
                pipeline_id=pipeline_id,
                tracker=self,
            )

        self._in_flight.add(pipeline_id)
        logger.debug("pipeline_started", extra={"pipeline_id": pipeline_id})
        return _PipelineSlot(ok=True, reason="", pipeline_id=pipeline_id, tracker=self)

    def _release(self, pipeline_id: str) -> None:
        self._in_flight.discard(pipeline_id)

    @property
    def in_flight(self) -> frozenset[str]:
        """Read-only snapshot of currently in-flight pipeline IDs."""
        return frozenset(self._in_flight)
