"""Custom exception hierarchy aligned with Java BusinessException."""


class RAGException(Exception):
    """Base RAG service exception."""

    def __init__(self, code: int, message: str, task_id: str | None = None):
        self.code = code
        self.message = message
        self.task_id = task_id
        super().__init__(message)


class TaskException(RAGException):
    """Task execution exception (retryable)."""

    def __init__(self, message: str, task_id: str | None = None, retryable: bool = True):
        super().__init__(code=500, message=message, task_id=task_id)
        self.retryable = retryable


class AIComputeException(RAGException):
    """AI computation exception (model call failure, embedding error, etc.)."""

    def __init__(self, message: str, task_id: str | None = None):
        super().__init__(code=10501, message=message, task_id=task_id)


class RetrievalException(RAGException):
    """Retrieval service exception — aligned with QA_RETRIEVAL_ERROR."""

    def __init__(self, message: str):
        super().__init__(code=10502, message=message)


class GenerationException(RAGException):
    """Generation service exception — aligned with QA_GENERATION_ERROR."""

    def __init__(self, message: str):
        super().__init__(code=10503, message=message)


class ResourceException(RAGException):
    """Infrastructure resource exception (MinIO, Milvus, Redis unavailable)."""

    def __init__(self, message: str, task_id: str | None = None):
        super().__init__(code=500, message=message, task_id=task_id)


class ValidationException(RAGException):
    """Request validation exception."""

    def __init__(self, message: str):
        super().__init__(code=400, message=message)
