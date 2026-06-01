"""RAG Python AI Service — main entry point.

Starts gRPC servers (retrieval + generation) and Kafka consumer.
Fully aligned with Java rag-server communication protocols.
"""
import signal
import sys
import time

from common.config_loader import get_config
from common.util.logger import setup_logging, get_logger
from communication.grpc_server.retrieval_service import create_retrieval_server
from communication.grpc_server.generation_service import create_generation_server
from communication.kafka_consumer.task_consumer import TaskKafkaConsumer
from communication.kafka_producer.status_producer import get_status_producer
from task_scheduler.task_dispatcher import get_task_dispatcher


def _wire_callbacks():
    """Connect task dispatcher to status producer for reporting."""
    dispatcher = get_task_dispatcher()
    producer = get_status_producer()
    dispatcher.register_callbacks(
        on_complete=producer.report_complete,
        on_failed=producer.report_failed,
    )


def main():
    cfg = get_config()
    setup_logging(level=cfg["logging"]["level"], log_format=cfg["logging"]["format"])
    logger = get_logger()

    logger.info("=" * 60)
    logger.info("RAG Python AI Service Starting...")
    logger.info("=" * 60)

    # Wire task completion callbacks
    _wire_callbacks()

    # Start gRPC servers
    retrieval_server = create_retrieval_server()
    generation_server = create_generation_server()

    retrieval_server.start()
    generation_server.start()

    # Start Kafka consumer
    consumer = TaskKafkaConsumer()
    consumer.start()

    # 注册到 Nacos 服务发现
    try:
        from common.nacos_registry import register_service
        register_service("rag-python-service", 50051, {"type": "retrieval"})
        register_service("rag-python-generation", 50052, {"type": "generation"})
    except Exception as e:
        logger.warning(f"Nacos registration skipped: {e}")

    logger.info("All services started successfully")

    # Graceful shutdown handler
    def shutdown(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        consumer.stop()
        get_task_dispatcher().shutdown(wait=False)
        retrieval_server.stop(grace=5)
        generation_server.stop(grace=5)
        get_status_producer().close()
        logger.info("Service stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
