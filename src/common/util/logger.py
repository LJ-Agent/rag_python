"""Unified logging with Loguru — task_id traceable, aligned with Java logback format."""
import sys
from loguru import logger


def setup_logging(level: str = "INFO", log_format: str | None = None):
    """Configure Loguru with structured format and task_id context."""
    logger.remove()

    if log_format is None:
        log_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{extra[task_id]: <20} | {name}:{function}:{line} | {message}"
        )

    logger.add(
        sys.stdout,
        level=level,
        format=log_format,
        colorize=True,
    )
    return logger


def bind_task_id(task_id: str = ""):
    """Bind task_id to log context for traceability."""
    return logger.bind(task_id=task_id)


def get_logger():
    """Return configured logger instance."""
    return logger
