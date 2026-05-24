"""Kafka consumer — listens to rag-file-process and rag-chunk-process topics.
Uses manual partition assignment (instead of consumer groups) for Kafka 4.x KRaft compatibility.
"""
import json
import threading
import time
import traceback

from kafka import KafkaConsumer, TopicPartition, OffsetAndMetadata
from kafka.errors import KafkaError

from common.config_loader import get_config
from common.constant.kafka_constants import KafkaTopics
from common.exception.exceptions import TaskException
from common.util.logger import get_logger
from common.util.utils import json_loads
from communication.kafka_producer.status_producer import get_status_producer
from task_scheduler.task_dispatcher import TaskContext, get_task_dispatcher
from task_scheduler.task_handlers import dispatch_task

logger = get_logger()


class TaskKafkaConsumer:
    """Consumes FILE_PROCESS, CHUNK_PROCESS, EMBED_PROCESS, DOCUMENT_DELETE tasks from Kafka."""

    def __init__(self):
        cfg = get_config()["kafka"]
        self._servers = cfg["bootstrap_servers"]
        self._consumer_group = cfg["consumer_group"]
        self._topics = [
            cfg["consume_topics"]["file_process"],
            cfg["consume_topics"]["chunk_process"],
            cfg["consume_topics"]["embed_process"],
            cfg["consume_topics"]["document_delete"],
        ]
        self._consumer: KafkaConsumer | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._dispatcher = get_task_dispatcher()
        self._status_producer = get_status_producer()
        # Track latest committed offset per partition
        self._committed: dict[tuple, int] = {}
        self._commit_lock = threading.Lock()
        # Register callbacks
        self._dispatcher.register_callbacks(
            on_complete=self._on_task_complete,
            on_failed=self._on_task_failed,
        )

    def start(self):
        """Start the Kafka consumer in a background thread."""
        consumer_cfg = get_config()["kafka"]["consumer"]
        self._consumer = KafkaConsumer(
            bootstrap_servers=self._servers,
            group_id=self._consumer_group,
            auto_offset_reset=consumer_cfg["auto_offset_reset"],
            enable_auto_commit=consumer_cfg["enable_auto_commit"],
            max_poll_records=consumer_cfg["max_poll_records"],
            value_deserializer=lambda m: m.decode("utf-8"),
        )
        # Manual partition assignment for Kafka 4.x KRaft compatibility
        tps = []
        for topic in self._topics:
            partitions = self._consumer.partitions_for_topic(topic)
            if partitions:
                for p in partitions:
                    tps.append(TopicPartition(topic, p))
        if tps:
            self._consumer.assign(tps)
            # After assignment, resume from committed offsets or seek to beginning
            for tp in tps:
                committed = self._consumer.committed(tp)
                if committed is not None:
                    self._consumer.seek(tp, committed)
                else:
                    self._consumer.seek_to_beginning(tp)
            logger.info(f"Assigned {len(tps)} partitions across {len(self._topics)} topics")

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="kafka-consumer")
        self._thread.start()
        logger.info(f"Kafka consumer started: topics={self._topics}, group={self._consumer_group}")

    def _poll_loop(self):
        while self._running:
            try:
                records = self._consumer.poll(timeout_ms=1000)
                for tp, msgs in records.items():
                    for msg in msgs:
                        try:
                            self._process_message(msg.value, msg.topic, msg.partition, msg.offset)
                        except Exception as e:
                            logger.error(f"Message processing error: {e}\n{traceback.format_exc()}")
            except KafkaError as e:
                logger.error(f"Kafka poll error: {e}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Consumer loop error: {e}")
                time.sleep(1)

    def _process_message(self, raw: str, topic: str, partition: int, offset: int):
        """Parse Kafka message JSON and submit to task dispatcher."""
        try:
            msg = json_loads(raw)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {raw[:200]}")
            self._commit_offset(topic, partition, offset)
            return

        task_id = msg.get("taskId", "unknown")
        task_type = msg.get("taskType")
        document_id = msg.get("documentId")
        kb_id = msg.get("kbId")
        data = msg.get("data", {})

        if not task_id or not task_type:
            logger.error(f"Missing required fields: taskId={task_id}, taskType={task_type}")
            self._commit_offset(topic, partition, offset)
            return

        context = TaskContext(
            task_id=task_id,
            task_type=task_type,
            document_id=document_id or 0,
            kb_id=kb_id or 0,
            data=data,
            created_at=msg.get("createdAt", ""),
        )
        context.set_kafka_meta(topic, partition, offset)
        self._dispatcher.submit(context, dispatch_task)
        logger.info(f"Kafka message dispatched: {task_id} -> {task_type}")

    def _on_task_complete(self, context: TaskContext, result: dict):
        """Callback: task succeeded — report status and commit offset."""
        try:
            self._status_producer.report_complete(context, result)
        except Exception:
            pass  # Status reporting is best-effort
        meta = context.get_kafka_meta()
        if meta:
            self._commit_offset(meta["topic"], meta["partition"], meta["offset"])

    def _on_task_failed(self, context: TaskContext, error: str):
        """Callback: task exhausted retries — report failure and commit offset."""
        try:
            self._status_producer.report_failed(context, error)
        except Exception:
            pass
        meta = context.get_kafka_meta()
        if meta:
            self._commit_offset(meta["topic"], meta["partition"], meta["offset"])

    def _commit_offset(self, topic: str, partition: int, offset: int):
        """Commit a specific offset to Kafka. Skips if already committed."""
        key = (topic, partition)
        with self._commit_lock:
            if key in self._committed and self._committed[key] >= offset:
                return
            self._committed[key] = offset
        try:
            tp = TopicPartition(topic, partition)
            meta = OffsetAndMetadata(offset + 1, "", 0)
            self._consumer.commit({tp: meta})
        except Exception as e:
            logger.warning(f"Offset commit failed: {topic}[{partition}] offset={offset} — {e}")

    def stop(self):
        self._running = False
        if self._consumer:
            self._consumer.close()
        logger.info("Kafka consumer stopped")
