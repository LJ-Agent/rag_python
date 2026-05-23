"""Kafka topic and consumer group constants — fully aligned with Java KafkaConstants."""
from enum import Enum


class KafkaTopics:
    """Kafka topic names, identical to com.rag.common.constant.KafkaConstants."""

    # Java -> Python (task dispatch)
    FILE_PROCESS = "rag-file-process"
    CHUNK_PROCESS = "rag-chunk-process"

    # Python -> Java (status report)
    TASK_COMPLETE = "rag-task-complete"
    TASK_FAILED = "rag-task-failed"

    # Consumer group
    CONSUMER_GROUP = "rag-python-group"


class GrpcConstants:
    """gRPC constants aligned with com.rag.common.constant.GrpcConstants."""

    TIMEOUT_SECONDS = 10
    MAX_RETRIES = 3
    RETRY_DELAY_MS = 1000
    RETRIEVAL_PORT = 50051
    GENERATION_PORT = 50052
