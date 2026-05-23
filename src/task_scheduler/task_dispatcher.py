"""Task dispatch, concurrency control, retry mechanism, status reporting."""
import asyncio
import time
import traceback
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
    task_type: str  # FILE_PROCESS or CHUNK_PROCESS
    document_id: int
    kb_id: int
    data: dict = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)
    retry_count: int = 0


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
        # Callbacks set by communication layer
        self._on_complete: callable | None = None
        self._on_failed: callable | None = None

    def register_callbacks(self, on_complete: callable, on_failed: callable):
        """Register callbacks: on_complete(context, result), on_failed(context, error)."""
        self._on_complete = on_complete
        self._on_failed = on_failed

    def submit(self, context: TaskContext, processor: callable) -> Future:
        """Submit a task for execution. processor(context) -> result dict."""
        if len(self._futures) >= self._max_concurrent:
            logger.warning(f"Task queue full, rejecting: {context.task_id}")
            raise TaskException("Task queue full, try again later", task_id=context.task_id)

        future = self._executor.submit(self._execute_with_retry, context, processor)
        self._futures[context.task_id] = future
        logger.info(f"Task submitted: {context.task_id}, type={context.task_type}, queue={len(self._futures)}")
        return future

    def _execute_with_retry(self, context: TaskContext, processor: callable) -> dict:
        """Execute task with retry logic and status reporting."""
        task_log = bind_task_id(context.task_id)
        task_log.info(f"Task started: type={context.task_type}, doc_id={context.document_id}")

        for attempt in range(self._retry_max + 1):
            try:
                result = processor(context)
                task_log.info(f"Task completed: attempts={attempt + 1}")
                if self._on_complete:
                    self._on_complete(context, result)
                self._futures.pop(context.task_id, None)
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
                    self._futures.pop(context.task_id, None)
                    raise TaskException(str(e), task_id=context.task_id, retryable=False)

    def get_status(self, task_id: str) -> str | None:
        """Check task status."""
        if task_id not in self._futures:
            return None
        future = self._futures[task_id]
        if future.done():
            return "completed" if not future.exception() else "failed"
        return "running"

    def shutdown(self, wait: bool = True):
        self._executor.shutdown(wait=wait)
        logger.info("Task dispatcher shutdown")


# Singleton
_dispatcher: TaskDispatcher | None = None


def get_task_dispatcher() -> TaskDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = TaskDispatcher()
    return _dispatcher
