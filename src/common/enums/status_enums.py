"""Document status enum — fully aligned with Java DocumentStatus state machine."""
from enum import Enum


class DocumentStatus(str, Enum):
    """13-state document lifecycle, identical transitions to Java."""

    UPLOADED = "UPLOADED"
    PARSING = "PARSING"
    CLEANING = "CLEANING"
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CHUNKING = "CHUNKING"
    CHUNK_REVIEW = "CHUNK_REVIEW"
    EMBEDDING = "EMBEDDING"
    COMPLETED = "COMPLETED"
    PARSING_FAILED = "PARSING_FAILED"
    CLEANING_FAILED = "CLEANING_FAILED"
    CHUNKING_FAILED = "CHUNKING_FAILED"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"

    @property
    def description(self) -> str:
        return {
            DocumentStatus.UPLOADED: "已上传",
            DocumentStatus.PARSING: "解析中",
            DocumentStatus.CLEANING: "清洗中",
            DocumentStatus.PENDING_REVIEW: "待审核",
            DocumentStatus.APPROVED: "已通过",
            DocumentStatus.REJECTED: "已驳回",
            DocumentStatus.CHUNKING: "分块中",
            DocumentStatus.CHUNK_REVIEW: "待块审核",
            DocumentStatus.EMBEDDING: "向量化中",
            DocumentStatus.COMPLETED: "已完成",
            DocumentStatus.PARSING_FAILED: "解析失败",
            DocumentStatus.CLEANING_FAILED: "清洗失败",
            DocumentStatus.CHUNKING_FAILED: "分块失败",
            DocumentStatus.EMBEDDING_FAILED: "向量化失败",
        }[self]

    @property
    def next_states(self) -> set["DocumentStatus"]:
        """Valid next states from current state."""
        _transitions = {
            DocumentStatus.UPLOADED: {DocumentStatus.PARSING},
            DocumentStatus.PARSING: {DocumentStatus.CLEANING, DocumentStatus.PARSING_FAILED},
            DocumentStatus.CLEANING: {DocumentStatus.PENDING_REVIEW, DocumentStatus.CLEANING_FAILED},
            DocumentStatus.PENDING_REVIEW: {DocumentStatus.APPROVED, DocumentStatus.REJECTED},
            DocumentStatus.APPROVED: {DocumentStatus.CHUNKING},
            DocumentStatus.REJECTED: {DocumentStatus.PARSING},
            DocumentStatus.CHUNKING: {DocumentStatus.EMBEDDING, DocumentStatus.CHUNKING_FAILED},
            DocumentStatus.EMBEDDING: {DocumentStatus.COMPLETED, DocumentStatus.EMBEDDING_FAILED},
            DocumentStatus.PARSING_FAILED: {DocumentStatus.PARSING},
            DocumentStatus.CLEANING_FAILED: {DocumentStatus.CLEANING},
            DocumentStatus.CHUNKING_FAILED: {DocumentStatus.CHUNKING},
            DocumentStatus.EMBEDDING_FAILED: {DocumentStatus.EMBEDDING},
            DocumentStatus.COMPLETED: set(),
        }
        return _transitions.get(self, set())

    def can_transit_to(self, target: "DocumentStatus") -> bool:
        return target in self.next_states

    def is_final(self) -> bool:
        return not self.next_states

    @staticmethod
    def next_after_task_complete(current: "DocumentStatus") -> "DocumentStatus":
        """Map current state to next state when a task reports completion."""
        _mapping = {
            DocumentStatus.UPLOADED: DocumentStatus.PARSING,
            DocumentStatus.PARSING: DocumentStatus.CLEANING,
            DocumentStatus.CLEANING: DocumentStatus.PENDING_REVIEW,
            DocumentStatus.APPROVED: DocumentStatus.CHUNKING,
            DocumentStatus.CHUNKING: DocumentStatus.EMBEDDING,
            DocumentStatus.EMBEDDING: DocumentStatus.COMPLETED,
        }
        if current not in _mapping:
            raise ValueError(f"当前状态不可流转: {current}")
        return _mapping[current]

    @staticmethod
    def failed_state_for(current: "DocumentStatus") -> "DocumentStatus":
        """Map processing state to its corresponding failed state."""
        _mapping = {
            DocumentStatus.PARSING: DocumentStatus.PARSING_FAILED,
            DocumentStatus.CLEANING: DocumentStatus.CLEANING_FAILED,
            DocumentStatus.CHUNKING: DocumentStatus.CHUNKING_FAILED,
            DocumentStatus.EMBEDDING: DocumentStatus.EMBEDDING_FAILED,
        }
        return _mapping.get(current, current)


class TaskType(str, Enum):
    """Task types aligned with Java TaskType enum."""

    FILE_PROCESS = "FILE_PROCESS"
    CHUNK_PROCESS = "CHUNK_PROCESS"
    EMBED_PROCESS = "EMBED_PROCESS"

    @property
    def description(self) -> str:
        return {
            TaskType.FILE_PROCESS: "文件处理",
            TaskType.CHUNK_PROCESS: "分块处理",
            TaskType.EMBED_PROCESS: "向量化入库",
        }[self]


class ReviewResult(str, Enum):
    """Review outcomes aligned with Java ReviewResult."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
