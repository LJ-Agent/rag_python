"""Kafka producer — reports task status back to Java via rag-task-complete / rag-task-failed."""
from kafka import KafkaProducer

from common.config_loader import get_config
from common.constant.kafka_constants import KafkaTopics
from common.exception.exceptions import ResourceException
from common.util.logger import get_logger
from common.util.utils import json_dumps, now_iso
from task_scheduler.task_dispatcher import TaskContext

logger = get_logger()


class StatusKafkaProducer:
    """Reports task completion/failure to Java via Kafka topics."""

    def __init__(self):
        cfg = get_config()["kafka"]
        self._producer = KafkaProducer(
            bootstrap_servers=cfg["bootstrap_servers"],
            value_serializer=lambda v: v.encode("utf-8"),
            acks=cfg["producer"].get("acks", 1),
            retries=cfg["producer"].get("retries", 3),
            linger_ms=cfg["producer"].get("linger_ms", 5),
        )
        self._task_complete_topic = cfg["produce_topics"]["task_complete"]
        self._task_failed_topic = cfg["produce_topics"]["task_failed"]

    def report_complete(self, context: TaskContext, result: dict | None = None):
        """Send task completion notification to Java."""
        message = self._build_message(context, result)
        try:
            self._producer.send(self._task_complete_topic, key=context.task_id, value=message)
            self._producer.flush()
            logger.info(f"Task complete reported: {context.task_id}")
        except Exception as e:
            logger.error(f"Failed to report task complete: {context.task_id} — {e}")
            raise ResourceException(f"Kafka send failed: {e}")

    def report_failed(self, context: TaskContext, error: str):
        """Send task failure notification to Java."""
        message = self._build_message(context, {"error": error})
        try:
            self._producer.send(self._task_failed_topic, key=context.task_id, value=message)
            self._producer.flush()
            logger.warning(f"Task failed reported: {context.task_id} — {error}")
        except Exception as e:
            logger.error(f"Failed to report task failure: {context.task_id} — {e}")

    def _build_message(self, context: TaskContext, result: dict | None = None) -> str:
        """Build Kafka message JSON matching Java KafkaMessage format."""
        msg = {
            "taskId": context.task_id,
            "taskType": context.task_type,
            "documentId": context.document_id,
            "kbId": context.kb_id,
            "data": result or {},
            "createdAt": context.created_at or now_iso(),
        }
        return json_dumps(msg)

    def close(self):
        self._producer.close()


# Singleton
_status_producer: StatusKafkaProducer | None = None


def get_status_producer() -> StatusKafkaProducer:
    global _status_producer
    if _status_producer is None:
        _status_producer = StatusKafkaProducer()
    return _status_producer
