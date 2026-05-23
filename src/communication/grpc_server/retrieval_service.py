"""gRPC RetrievalService — implements retrieval.proto for sync retrieval from Java."""
import time
from concurrent import futures

import grpc

from common.config_loader import get_config
from common.exception.exceptions import RetrievalException
from common.util.logger import get_logger
from ai_core.retrieval.hybrid_retrieval import HybridRetrieval

logger = get_logger()


class RetrievalServiceServicer:
    """gRPC service implementing RetrievalService.Retrieve."""

    def __init__(self):
        self._retrieval = HybridRetrieval()
        cfg = get_config()["retrieval"]
        self._default_top_k = cfg["default_top_k"]
        self._max_top_k = cfg["max_top_k"]
        self._default_threshold = cfg["score_threshold"]

    def Retrieve(self, request, context):
        """Synchronous retrieval endpoint, aligned with Java RetrievalServiceClient."""
        start = time.perf_counter()

        query = request.query
        kb_ids = list(request.kb_ids) if request.kb_ids else []
        top_k = request.top_k if request.top_k > 0 else self._default_top_k
        top_k = min(top_k, self._max_top_k)
        score_threshold = request.score_threshold if request.score_threshold > 0 else self._default_threshold

        if not query.strip():
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("query is required")
            from communication.grpc_server.generated import retrieval_pb2
            return retrieval_pb2.RetrievalResponse()

        try:
            results = self._retrieval.retrieve(query, kb_ids, top_k, score_threshold)
            latency_ms = int((time.perf_counter() - start) * 1000)

            from communication.grpc_server.generated import retrieval_pb2

            chunks = [
                retrieval_pb2.DocumentChunk(
                    chunk_id=str(r.get("chunk_id", "")),
                    document_id=r.get("document_id", 0),
                    document_name=str(r.get("document_name", "")),
                    content=str(r.get("content", "")),
                    score=float(r.get("score", 0.0)),
                    chunk_index=r.get("chunk_index", 0),
                )
                for r in results
            ]

            logger.info(f"gRPC Retrieve: query='{query[:50]}...', kb_ids={kb_ids}, "
                        f"results={len(chunks)}, latency={latency_ms}ms")

            return retrieval_pb2.RetrievalResponse(
                chunks=chunks,
                total_count=len(chunks),
                latency_ms=latency_ms,
            )
        except RetrievalException as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            from communication.grpc_server.generated import retrieval_pb2
            return retrieval_pb2.RetrievalResponse()


def create_retrieval_server() -> grpc.Server:
    """Create and return a gRPC server for RetrievalService."""
    from communication.grpc_server.generated import retrieval_pb2_grpc

    cfg = get_config()["grpc"]
    port = cfg["retrieval"]["port"]
    max_workers = cfg["retrieval"]["max_workers"]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(RetrievalServiceServicer(), server)
    server.add_insecure_port(f"{cfg['server']['host']}:{port}")
    logger.info(f"RetrievalService gRPC server on port {port}")
    return server
