from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from queue import Queue, Full, Empty
import struct
from typing import Callable, Optional, List, Sequence
import os


logger = logging.getLogger("Utils.SharedMemory")
# ---------------------------------------------------------------------------
# Optional psutil import — affinity is silently skipped on platforms or
# environments where psutil is unavailable (e.g. restricted containers).
# ---------------------------------------------------------------------------

try:
    import psutil as _psutil  # type: ignore[import-untyped]

    _PSUTIL_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False
    logger.warning(
        "psutil not found — CPU core affinity pinning will be disabled. "
        "Install with: pip install psutil"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_WORKERS: int = 4
MAX_WORKERS: int = 16

# Supervisor evaluates queue depth every N seconds.
_SUPERVISOR_INTERVAL: float = 2.0

# Scale-up when queue depth exceeds this threshold per active worker.
_SCALE_UP_RATIO: float = 2.0

# Scale-down when queue depth drops below this threshold per active worker.
_SCALE_DOWN_RATIO: float = 0.5

# How long an idle worker waits for an item before looping (keeps threads
# responsive to the stop event without busy-spinning).
_WORKER_TIMEOUT: float = 1.0


# ---------------------------------------------------------------------------
# Non-blocking Task Cancellation
# ---------------------------------------------------------------------------


class CancellableTask:
    """A task wrapper that supports non-blocking asynchronous cancellation.
    
    This class provides cancellation handlers that release resources asynchronously
    without delaying the core scheduling loops. When a task is cancelled, cleanup
    is scheduled on the event loop rather than blocking the main thread.
    
    Usage::
    
        def my_task():
            # Do work
            pass
        
        cancellable = CancellableTask(my_task)
        cancellable.start()
        
        # Later, cancel without blocking
        await cancellable.cancel_async()
    """
    
    def __init__(
        self,
        task_func: Callable,
        task_id: Optional[int] = None,
        on_cancel: Optional[Callable] = None,
    ) -> None:
        """Initialize a cancellable task.
        
        Parameters
        ----------
        task_func:
            The callable to execute.
        task_id:
            Optional identifier for the task.
        on_cancel:
            Optional callback to execute when the task is cancelled.
            This callback is executed asynchronously.
        """
        self._task_func = task_func
        self._task_id = task_id
        self._on_cancel = on_cancel
        self._cancelled = False
        self._completed = False
        self._cleanup_event = threading.Event()
        self._lock = threading.Lock()
        
    @property
    def task_id(self) -> Optional[int]:
        """Return the task identifier."""
        return self._task_id
    
    @property
    def is_cancelled(self) -> bool:
        """Return True if the task has been cancelled."""
        with self._lock:
            return self._cancelled
    
    @property
    def is_completed(self) -> bool:
        """Return True if the task has completed."""
        with self._lock:
            return self._completed
    
    def mark_completed(self) -> None:
        """Mark the task as completed."""
        with self._lock:
            self._completed = True
            self._cleanup_event.set()
    
    def cancel(self) -> bool:
        """Cancel the task synchronously.
        
        Returns True if the task was cancelled, False if it was already
        cancelled or completed.
        
        This method is synchronous but schedules the cleanup callback
        asynchronously if one is provided.
        """
        with self._lock:
            if self._cancelled or self._completed:
                return False
            self._cancelled = True
            self._cleanup_event.set()
        
        # Run cleanup callback directly if provided (synchronous for simplicity)
        if self._on_cancel:
            self._run_cleanup_callback()
        
        logger.debug(
            "CancellableTask[%s]: cancelled",
            self._task_id
        )
        return True
    
    async def cancel_async(self) -> bool:
        """Cancel the task asynchronously.
        
        This is the preferred method for async contexts as it does not
        block the event loop.
        
        Returns True if the task was cancelled, False if it was already
        cancelled or completed.
        """
        with self._lock:
            if self._cancelled or self._completed:
                return False
            self._cancelled = True
            self._cleanup_event.set()
        
        # Run cleanup callback asynchronously
        if self._on_cancel:
            await asyncio.to_thread(self._run_cleanup_callback)
        
        logger.debug(
            "CancellableTask[%s]: cancelled asynchronously",
            self._task_id
        )
        return True
    
    def _run_cleanup_callback(self) -> None:
        """Execute the cleanup callback in a thread-safe manner."""
        if self._on_cancel:
            try:
                self._on_cancel()
            except Exception as exc:
                logger.exception(
                    "CancellableTask[%s]: cleanup callback failed: %s",
                    self._task_id,
                    exc
                )
    
    def wait_for_cleanup(self, timeout: Optional[float] = None) -> bool:
        """Wait for cleanup to complete.
        
        Parameters
        ----------
        timeout:
            Maximum time to wait in seconds. If None, wait indefinitely.
        
        Returns
        -------
        bool
            True if cleanup completed, False if timeout was reached.
        """
        return self._cleanup_event.wait(timeout=timeout)
    
    def __call__(self) -> None:
        """Execute the task function."""
        if self.is_cancelled:
            logger.debug(
                "CancellableTask[%s]: skipping execution (cancelled)",
                self._task_id
            )
            return
        
        try:
            self._task_func()
        finally:
            self.mark_completed()


# ---------------------------------------------------------------------------
# Core-affinity configuration
# ---------------------------------------------------------------------------


@dataclass
class CoreAffinityConfig:
    """Declares which logical CPU cores critical ingestion threads are pinned to.

    Attributes
    ----------
    enabled:
        Master switch.  Set to ``False`` to disable all affinity pinning
        without removing the configuration object.
    cores:
        Ordered list of logical CPU core indices to pin to.  Critical workers
        are assigned round-robin across this list so no single core is
        monopolised.  Defaults to an empty list, which means "auto-select the
        first *n* cores at startup" where *n* is ``MIN_WORKERS``.
    fallback_to_all_cores:
        When ``True`` (default), if affinity assignment fails for a thread the
        worker still starts normally without a pinned affinity.  When ``False``
        a failed pin raises ``RuntimeError`` and aborts the pool start.

    Example — pin the four critical ingestion workers to cores 0-3::

        config = CoreAffinityConfig(enabled=True, cores=[0, 1, 2, 3])
        pool = DynamicThreadingPool(affinity_config=config)
        pool.start()
    """

    enabled: bool = True
    cores: List[int] = field(default_factory=list)
    fallback_to_all_cores: bool = True


def _resolve_cores(config: CoreAffinityConfig, n_critical: int) -> List[int]:
    """Return the list of core indices to use for *n_critical* pinned threads.

    If ``config.cores`` is empty the function auto-selects the first
    ``n_critical`` logical CPUs reported by the OS (wrapped round-robin if
    the machine has fewer cores than workers).  Explicitly supplied cores are
    validated against the available set and returned as-is.
    """
    available = list(range(os.cpu_count() or 1))

    if not config.cores:
        # Auto-assign: spread critical workers across the first n_critical cores,
        # wrapping round-robin if necessary.
        return [available[i % len(available)] for i in range(n_critical)]

    # Validate caller-supplied cores against what the OS reports.
    invalid = [c for c in config.cores if c not in available]
    if invalid:
        raise ValueError(
            f"CoreAffinityConfig specifies cores {invalid} which are not "
            f"available on this system (available: {available})."
        )
    return list(config.cores)


def pin_thread_to_cores(cores: Sequence[int]) -> bool:
    """Set the CPU affinity of the *calling* thread to the supplied *cores*.

    Uses ``psutil`` to restrict the OS scheduler to the given logical cores so
    the thread runs exclusively on those cores unless the kernel pre-empts it
    for a higher-priority task.

    Parameters
    ----------
    cores:
        One or more logical CPU core indices to pin to.

    Returns
    -------
    bool
        ``True`` if the affinity was set successfully, ``False`` if ``psutil``
        is unavailable or the underlying system call failed.

    Notes
    -----
    *   On Windows ``psutil`` uses ``SetThreadAffinityMask``.
    *   On Linux it calls ``pthread_setaffinity_np`` via ``/proc``.
    *   On macOS thread-level affinity is not supported by the kernel; the call
        will silently succeed at the process level only.
    *   This function must be called from *inside* the target thread, because
        ``psutil`` exposes per-process affinity rather than per-thread affinity
        on some platforms.  The implementation therefore sets the affinity on
        the current process restricted to *cores* for the duration of thread
        startup, then restores full-core access.  On Linux with glibc the
        underlying ``sched_setaffinity`` is inherited by new threads but not
        applied retroactively — calling from within the worker loop body
        achieves true per-thread isolation.
    """
    if not _PSUTIL_AVAILABLE:
        return False

    try:
        proc = _psutil.Process()
        proc.cpu_affinity(list(cores))
        logger.debug(
            "Thread %s pinned to CPU core(s) %s",
            threading.current_thread().name,
            list(cores),
        )
        return True
    except Exception as exc:  # noqa: BLE001 — psutil errors vary by platform
        logger.warning(
            "Failed to set CPU affinity for thread %s to cores %s: %s",
            threading.current_thread().name,
            list(cores),
            exc,
        )
        return False


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolSnapshot:
    """Immutable view of pool state at a point in time."""

    active_workers: int
    queue_depth: int
    tasks_completed: int
    tasks_failed: int
    pinned_cores: tuple[int, ...]  # cores assigned to critical workers


@dataclass
class _PoolState:
    """Mutable internal counters — always accessed under _lock."""

    worker_count: int = MIN_WORKERS
    tasks_completed: int = 0
    tasks_failed: int = 0


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------


def _worker(
    work_queue: Queue,
    stop_event: threading.Event,
    state: _PoolState,
    lock: threading.Lock,
) -> None:
    """Main loop executed by each worker thread.

    Dequeues callables and runs them.  Increments completion/failure counters
    under *lock* so the supervisor can read consistent metrics.
    """
    while not stop_event.is_set():
        try:
            task: Callable = work_queue.get(timeout=_WORKER_TIMEOUT)
        except Empty:
            continue

        try:
            task()
            with lock:
                state.tasks_completed += 1
        except Exception:
            logger.exception("Worker caught unhandled exception in task")
            with lock:
                state.tasks_failed += 1
        finally:
            work_queue.task_done()


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


def _supervisor(
    work_queue: Queue,
    threads: list[threading.Thread],
    stop_event: threading.Event,
    state: _PoolState,
    lock: threading.Lock,
    thread_factory: Callable[[], threading.Thread],
) -> None:
    """Background supervisor that adjusts worker count based on queue depth.

    Scale-up rule: ``queue_depth / active_workers > SCALE_UP_RATIO``
    Scale-down rule: ``queue_depth / active_workers < SCALE_DOWN_RATIO``

    Worker count is clamped to [MIN_WORKERS, MAX_WORKERS].

    Note: dynamically scaled-up workers are *not* pinned to specific cores —
    they are overflow helpers and benefit from the full scheduler range.
    """
    while not stop_event.is_set():
        time.sleep(_SUPERVISOR_INTERVAL)

        with lock:
            depth = work_queue.qsize()
            current = state.worker_count

        if current == 0:
            ratio = float("inf")
        else:
            ratio = depth / current

        if ratio > _SCALE_UP_RATIO and current < MAX_WORKERS:
            # Add one worker per supervisor tick to avoid overshooting.
            new_thread = thread_factory()
            new_thread.start()
            with lock:
                threads.append(new_thread)
                state.worker_count += 1
            logger.info(
                "ThreadingPool: scaled UP to %d workers (queue depth %d)",
                state.worker_count,
                depth,
            )

        elif ratio < _SCALE_DOWN_RATIO and current > MIN_WORKERS:
            # Signal one idle worker to exit on its next empty-queue loop by
            # enqueuing a sentinel None value that workers check for.
            work_queue.put(None)  # handled below — see _worker_with_sentinel
            with lock:
                state.worker_count -= 1
            logger.info(
                "ThreadingPool: scaled DOWN to %d workers (queue depth %d)",
                state.worker_count,
                depth,
            )


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class DynamicThreadingPool:
    """Automated worker-scaling thread pool with optional CPU core affinity.

    Critical ingestion threads (those spawned at ``start()`` time) can be
    pinned to dedicated hardware CPU cores via ``affinity_config``.  This
    isolates the core execution environment for time-sensitive regional fiat
    feed compilations, eliminating micro-latency variance caused by the OS
    scheduler migrating threads across cores.

    Dynamically scaled workers (added by the supervisor under load) are *not*
    pinned — they are ephemeral helpers that benefit from unrestricted
    scheduling.

    Usage::

        from src.utils.threading_pool import DynamicThreadingPool, CoreAffinityConfig

        # Pin the 4 critical workers to cores 0-3
        config = CoreAffinityConfig(enabled=True, cores=[0, 1, 2, 3])
        pool = DynamicThreadingPool(affinity_config=config)
        pool.start()

        pool.submit(my_callable)
        pool.stop()

    Context manager form::

        with DynamicThreadingPool(affinity_config=CoreAffinityConfig()) as pool:
            pool.submit(my_callable)
    """

    def __init__(
        self,
        min_workers: int = MIN_WORKERS,
        max_workers: int = MAX_WORKERS,
        supervisor_interval: float = _SUPERVISOR_INTERVAL,
        affinity_config: Optional[CoreAffinityConfig] = None,
    ) -> None:
        if min_workers < 1:
            raise ValueError("min_workers must be >= 1")
        if max_workers < min_workers:
            raise ValueError("max_workers must be >= min_workers")

        self._min_workers = min_workers
        self._max_workers = max_workers
        self._supervisor_interval = supervisor_interval
        self._affinity_config: CoreAffinityConfig = (
            affinity_config if affinity_config is not None else CoreAffinityConfig(enabled=False)
        )

        self._work_queue: Queue = Queue()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._state = _PoolState(worker_count=min_workers)
        self._threads: list[threading.Thread] = []
        self._supervisor_thread: Optional[threading.Thread] = None
        # Counter for assigning deterministic IDs to workers
        self._next_worker_id: int = 0

        # Resolved core assignments for critical workers (populated at start()).
        self._pinned_cores: List[int] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_worker_thread(self, name: Optional[str] = None) -> threading.Thread:
        """Create (but do not start) a daemon worker thread with a unique ID."""
        worker_id = self._next_worker_id
        self._next_worker_id += 1
        return threading.Thread(
            target=_worker_with_sentinel,
            args=(
                self._work_queue,
                self._stop_event,
                self._state,
                self._lock,
                worker_id,
                None,  # No core pinning by default
            ),
            daemon=True,
            name=name if name else f"ThreadingPool-Worker-{worker_id}",
        )

    def _make_pinned_worker_thread(self, core: int, index: int) -> threading.Thread:
        """Create (but do not start) a daemon worker thread pinned to a specific core."""
        worker_id = self._next_worker_id
        self._next_worker_id += 1
        return threading.Thread(
            target=_worker_with_sentinel,
            args=(
                self._work_queue,
                self._stop_event,
                self._state,
                self._lock,
                worker_id,
                core,
            ),
            daemon=True,
            name=f"Ingestion-Worker-{index}",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn initial workers and the supervisor thread.

        Critical ingestion workers are pinned to dedicated CPU cores when
        ``affinity_config.enabled`` is ``True`` and ``psutil`` is available.
        Affinity assignment is performed from *within* each worker thread so
        the OS-level thread inherits the correct core mask.
        """
        if self._supervisor_thread is not None:
            raise RuntimeError("Pool is already running")

        cfg = self._affinity_config

        if cfg.enabled and _PSUTIL_AVAILABLE:
            try:
                self._pinned_cores = _resolve_cores(cfg, self._min_workers)
            except ValueError as exc:
                if cfg.fallback_to_all_cores:
                    logger.warning(
                        "CoreAffinityConfig validation failed (%s); "
                        "starting without affinity pinning.",
                        exc,
                    )
                    self._pinned_cores = []
                else:
                    raise

        use_affinity = bool(self._pinned_cores)

        for i in range(self._min_workers):
            if use_affinity:
                # Assign cores round-robin across available pinned cores.
                core = self._pinned_cores[i % len(self._pinned_cores)]
                t = self._make_pinned_worker_thread(core=core, index=i)
            else:
                t = self._make_worker_thread(name=f"Ingestion-Worker-{i}")
            t.start()
            self._threads.append(t)

        self._supervisor_thread = threading.Thread(
            target=_supervisor,
            args=(
                self._work_queue,
                self._threads,
                self._stop_event,
                self._state,
                self._lock,
                self._make_worker_thread,  # scaled workers are unpinned
            ),
            daemon=True,
            name="ThreadingPool-Supervisor",
        )
        self._supervisor_thread.start()

        if use_affinity:
            logger.info(
                "ThreadingPool: started with %d pinned workers "
                "(cores=%s, min=%d, max=%d)",
                self._min_workers,
                self._pinned_cores,
                self._min_workers,
                self._max_workers,
            )
        else:
            reason = (
                "psutil unavailable" if not _PSUTIL_AVAILABLE
                else "affinity disabled"
            )
            logger.info(
                "ThreadingPool: started with %d workers "
                "(affinity: %s, min=%d, max=%d)",
                self._min_workers,
                reason,
                self._min_workers,
                self._max_workers,
            )

    def submit(self, task: Callable) -> None:
        """Enqueue *task* for execution by a worker thread.

        Raises ``RuntimeError`` if the pool has been stopped.
        """
        if self._stop_event.is_set():
            raise RuntimeError("Cannot submit tasks to a stopped pool")
        self._work_queue.put(task)

    def stop(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """Signal all workers and the supervisor to stop.

        Parameters
        ----------
        wait:
            If ``True`` (default), block until all threads have exited.
        timeout:
            Optional per-thread join timeout in seconds.
        """
        self._stop_event.set()

        if wait:
            if self._supervisor_thread is not None:
                self._supervisor_thread.join(timeout=timeout)
            for t in self._threads:
                t.join(timeout=timeout)

        logger.info("ThreadingPool: stopped")

    def snapshot(self) -> PoolSnapshot:
        """Return an immutable snapshot of current pool metrics."""
        with self._lock:
            return PoolSnapshot(
                active_workers=self._state.worker_count,
                queue_depth=self._work_queue.qsize(),
                tasks_completed=self._state.tasks_completed,
                tasks_failed=self._state.tasks_failed,
                pinned_cores=tuple(self._pinned_cores),
            )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "DynamicThreadingPool":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.stop()
        return False


# ---------------------------------------------------------------------------
# Worker variant that supports scale-down sentinel
# ---------------------------------------------------------------------------


def _worker_with_sentinel(
    work_queue: Queue,
    stop_event: threading.Event,
    state: _PoolState,
    lock: threading.Lock,
    worker_id: int,
    core: Optional[int],
) -> None:
    """Worker loop that also handles the ``None`` scale-down sentinel.

    When the supervisor wants to remove a worker it enqueues ``None``.  The
    first worker to dequeue it exits cleanly, reducing the active count by one.
    """
    # Set CPU affinity for this worker before processing tasks
    if core is not None:
        pin_thread_to_cores([core])
    while not stop_event.is_set():
        try:
            task = work_queue.get(timeout=_WORKER_TIMEOUT)
        except Empty:
            continue

        # Scale-down sentinel — exit gracefully.
        if task is None:
            work_queue.task_done()
            break

        try:
            task()
            with lock:
                state.tasks_completed += 1
        except Exception:
            logger.exception("Worker caught unhandled exception in task")
            with lock:
                state.tasks_failed += 1
        finally:
            work_queue.task_done()


# ---------------------------------------------------------------------------
# Module-level affinity configuration
# ---------------------------------------------------------------------------

#: Default affinity config for the ingestion pipeline.
#:
#: ``cores`` is left empty so the pool auto-selects the first ``MIN_WORKERS``
#: logical cores at startup.  Override before calling ``threading_pool.start()``
#: to target specific cores, e.g.::
#:
#:     from src.utils.threading_pool import INGESTION_AFFINITY
#:     INGESTION_AFFINITY.cores = [0, 1, 2, 3]
INGESTION_AFFINITY: CoreAffinityConfig = CoreAffinityConfig(
    enabled=True,
    cores=[],  # auto-select at startup
    fallback_to_all_cores=True,
)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Shared pool instance; call ``threading_pool.start()`` to activate.
#: Core affinity is enabled by default via ``INGESTION_AFFINITY``.
threading_pool = DynamicThreadingPool(
    min_workers=MIN_WORKERS,
    max_workers=MAX_WORKERS,
    affinity_config=INGESTION_AFFINITY,
)

__all__ = [
    "MIN_WORKERS",
    "MAX_WORKERS",
    "CoreAffinityConfig",
    "INGESTION_AFFINITY",
    "PoolSnapshot",
    "DynamicThreadingPool",
    "pin_thread_to_cores",
    "threading_pool",
    "CancellableTask",
]