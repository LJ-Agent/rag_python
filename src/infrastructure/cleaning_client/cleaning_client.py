"""gRPC client for RAG-CLEANING service — delegates document cleaning tasks."""

import grpc
from common.config_loader import get_config
from common.util.logger import get_logger
from communication.grpc_server.generated import cleaning_pb2, cleaning_pb2_grpc

logger = get_logger()

_channel: grpc.Channel | None = None
_stub: cleaning_pb2_grpc.CleaningServiceStub | None = None


def _get_stub() -> cleaning_pb2_grpc.CleaningServiceStub:
    """Lazy-init gRPC channel and stub to cleaning service."""
    global _channel, _stub
    if _stub is not None:
        return _stub

    cfg = get_config()["cleaning"]
    endpoint = cfg["grpc_endpoint"]
    if ":" in endpoint:
        host, port_str = endpoint.rsplit(":", 1)
        port = int(port_str)
    else:
        host, port = endpoint, 50056

    _channel = grpc.insecure_channel(
        f"{host}:{port}",
        options=[
            ("grpc.keepalive_time_ms", 30000),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )
    _stub = cleaning_pb2_grpc.CleaningServiceStub(_channel)
    logger.info(f"Cleaning service client connected: {host}:{port}")
    return _stub


def clean_document(
    task_id: str,
    document_id: str,
    kb_id: int,
    tenant_id: str,
    file_name: str,
    file_url: str,
    mime_type: str = "",
    timeout: float = 600.0,
) -> cleaning_pb2.CleaningResponse:
    """Submit a document cleaning request synchronously.

    Args:
        task_id: Unique task ID.
        document_id: Document ID from upstream.
        kb_id: Knowledge base ID.
        tenant_id: Tenant ID.
        file_name: Original file name.
        file_url: MinIO path to the raw file.
        mime_type: MIME type of the file.
        timeout: gRPC call timeout in seconds (default 10min).

    Returns:
        CleaningResponse with status, markdown_url, metadata_url, quality report.
    """
    stub = _get_stub()
    request = cleaning_pb2.CleaningRequest(
        task_id=task_id,
        document_id=str(document_id),
        kb_id=kb_id,
        tenant_id=tenant_id or "default",
        file_name=file_name,
        file_url=file_url,
        mime_type=mime_type,
    )

    logger.info(f"Cleaning request: task={task_id}, doc={document_id}, file={file_name}")

    try:
        response = stub.Clean(request, timeout=timeout)
        status_name = cleaning_pb2.CleaningStatus.Name(response.status)
        logger.info(
            f"Cleaning done: task={task_id}, status={status_name}, "
            f"md={response.markdown_url}, meta={response.metadata_url}"
        )
        return response
    except grpc.RpcError as e:
        logger.error(f"Cleaning gRPC call failed: {e.code()} - {e.details()}")
        raise


def health_check(timeout: float = 5.0) -> bool:
    """Check if the cleaning service is healthy."""
    try:
        stub = _get_stub()
        request = cleaning_pb2.HealthCheckRequest()
        response = stub.HealthCheck(request, timeout=timeout)
        return response.healthy
    except grpc.RpcError:
        return False


def close():
    """Close the gRPC channel."""
    global _channel, _stub
    if _channel:
        _channel.close()
        _channel = None
        _stub = None
