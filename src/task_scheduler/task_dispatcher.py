"""Task dispatch, concurrency control, retry mechanism, status reporting."""
import asyncio
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any

from common.config_loader import get_config
from common.enums.status_enums import TaskType
from common.exception.exceptions import TaskException
from common.util.logger import bind_task_id, get_logger
from common.util.utils import now_iso

logger = get_logger()


@dataclass
class TaskContext:
    """Task execution context — created from Kafka message."""

    task_id: str
    task_type: str  # FILE_PROCESS, CHUNK_PROCESS, EMBED_PROCESS, DOCUMENT_DELETE
    document_id: int
    kb_id: int
    data: dict = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)
    retry_count: int = 0
    _kafka_meta: dict | None = None  # {topic, partition, offset} for offset commit

    def set_kafka_meta(self, topic: str, partition: int, offset: int):
        self._kafka_meta = {"topic": topic, "partition": partition, "offset": offset}

    def get_kafka_meta(self) -> dict | None:
        return self._kafka_meta


class TaskDispatcher:
    """Manages async task execution with concurrency limits, retry, and status callbacks."""

    def __init__(self):
        cfg = get_config()["task"]
        self._max_concurrent = cfg["max_concurrent"]
        self._retry_max = cfg["retry_max"]
        self._retry_base_delay = cfg["retry_base_delay"]
        self._task_timeout = cfg["task_timeout"]
        self._executor = ThreadPoolExecutor(max_workers=self._max_concurrent)
        self._futures: dict[str, Future] = {}
        # Buffer queue for pending tasks when all workers are busy
        self._pending: deque = deque()
        self._pending_event = threading.Event()
        self._lock = threading.Lock()
        # Drainer thread pulls from pending queue into executor
        self._drainer_running = True
        self._drainer_thread = threading.Thread(target=self._drain_queue, daemon=True, name="task-drainer")
        self._drainer_thread.start()
        # Callbacks set by communication layer
        self._on_complete: callable | None = None
        self._on_failed: callable | None = None

    def register_callbacks(self, on_complete: callable, on_failed: callable):
        """Register callbacks: on_complete(context, result), on_failed(context, error)."""
        self._on_complete = on_complete
        self._on_failed = on_failed

    def _drain_queue(self):
        """Background thread that drains the pending queue into the executor."""
        while self._drainer_running:
            try:
                with self._lock:
                    if len(self._futures) < self._max_concurrent and self._pending:
                        context, processor = self._pending.popleft()
                    else:
                        context = None
                if context is not None:
                    self._do_submit(context, processor)
                else:
                    self._pending_event.wait(timeout=1.0)
                    self._pending_event.clear()
            except Exception:
                logger.error(f"Drainer error: {traceback.format_exc()}")

    def _on_task_done(self, context: TaskContext):
        """Called after a task completes or fails — triggers drain to submit next pending."""
        self._pending_event.set()

    def submit(self, context: TaskContext, processor: callable) -> Future:
        """Submit a task. If at capacity, enqueue in pending buffer."""
        with self._lock:
            if len(self._futures) < self._max_concurrent:
                return self._do_submit(context, processor)
            self._pending.append((context, processor))
            logger.info(f"Task queued (all workers busy): {context.task_id}, pending={len(self._pending)}")
            self._pending_event.set()
            return None  # caller doesn't need the future

    def _do_submit(self, context: TaskContext, processor: callable) -> Future:
        future = self._executor.submit(self._execute_with_retry, context, processor)
        self._futures[context.task_id] = future
        logger.info(f"Task submitted: {context.task_id}, type={context.task_type}, active={len(self._futures)}")
        return future

    def _execute_with_retry(self, context: TaskContext, processor: callable) -> dict:
        """Execute task with retry logic and status reporting."""
        task_log = bind_task_id(context.task_id)
        task_log.info(f"Task started: type={context.task_type}, doc_id={context.document_id}")

        try:
            for attempt in range(self._retry_max + 1):
                try:
                    result = processor(context)
                    task_log.info(f"Task completed: attempts={attempt + 1}")
                    if self._on_complete:
                        self._on_complete(context, result)
                    return result
                except Exception as e:
                    context.retry_count = attempt
                    if attempt < self._retry_max:
                        delay = self._retry_base_delay * (2**attempt)
                        task_log.warning(f"Task failed (attempt {attempt + 1}): {e}, retry in {delay}s")
                        time.sleep(delay)
                    else:
                        task_log.error(f"Task failed after {self._retry_max + 1} attempts: {e}\n{traceback.format_exc()}")
                        if self._on_failed:
                            self._on_failed(context, str(e))
                        raise TaskException(str(e), task_id=context.task_id, retryable=False)
        finally:
            self._futures.pop(context.task_id, None)
            self._on_task_done(context)

    def get_status(self, task_id: str) -> str | None:
        """Check task status."""
        if task_id not in self._futures:
            return None
        future = self._futures[task_id]
        if future.done():
            return "completed" if not future.exception() else "failed"
        return "running"

    def shutdown(self, wait: bool = True):
        self._drainer_running = False
        self._pending_event.set()
        self._executor.shutdown(wait=wait)
        logger.info("Task dispatcher shutdown")


# Singleton
_dispatcher: TaskDispatcher | None = None


def get_task_dispatcher() -> TaskDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = TaskDispatcher()
    return _dispatcher
