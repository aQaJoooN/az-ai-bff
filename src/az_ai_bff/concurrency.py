"""Concurrency control for AI backend requests.

Three independent mechanisms, applied in order inside the analyze route:

AISemaphore
    Caps the number of simultaneous AI backend calls across all endpoints.
    Excess requests are rejected immediately (non-blocking) with 429.

PipelineInFlightTracker
    In-memory guard keyed on the pipeline *definition* ID (not the run/build
    ID).  While any run of a given pipeline is being analyzed, a second
    request for any run of the same pipeline is rejected with 409
    ``pipeline_already_processing``.  The lock is released whether the
    analysis succeeds or fails, so errors always allow retries.

    Rationale: a pipeline definition can produce many runs.  Allowing two
    different runs to be analyzed concurrently would create a race on writing
    the ``ai-analyzed`` tag and would flood the AI backend.

The completed-run idempotency check is NOT stored here — it lives in
``AzureDevOpsLogFetcher`` which reads and writes build tags directly on the
Azure DevOps server, making it durable across app restarts and visible to
every client.
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
# PipelineInFlightTracker
# ---------------------------------------------------------------------------

class _PipelineSlot:
    """Returned by ``PipelineInFlightTracker.try_start()``.

    ``ok``            — False if a run of this pipeline is already in-flight.
    ``reason``        — error code when ``ok`` is False.
    ``pipeline_id``   — the definition ID that was registered.

    Use as a synchronous context manager.  The lock is released on exit
    regardless of whether the analysis succeeded.
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
    at the same time.

    Keyed on the pipeline *definition* ID, not the build/run ID.  This is
    intentional: even if two different runs of the same pipeline are
    requested concurrently, only one is allowed through at a time.

    State is in-memory only — a restart clears all locks.  This is correct
    because an in-flight analysis cannot survive a restart anyway.
    """

    def __init__(self) -> None:
        self._in_flight: set[str] = set()

    def try_start(self, pipeline_id: str) -> _PipelineSlot:
        """Attempt to register *pipeline_id* as in-flight.

        Returns a :class:`_PipelineSlot` whose ``.ok`` is True when the
        registration succeeded.
        """
        if pipeline_id in self._in_flight:
            logger.warning(
                "pipeline_already_processing",
                extra={"pipeline_id": pipeline_id},
            )
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
        return frozenset(self._in_flight)
