"""gRPC GenerationService — implements generation.proto for streaming/sync generation."""
import asyncio
import queue
import threading
from concurrent import futures

import grpc

from common.config_loader import get_config
from common.exception.exceptions import GenerationException
from common.util.logger import get_logger
from ai_core.generation.qa_generator import QAGenerator

logger = get_logger()


class GenerationServiceServicer:
    """gRPC service implementing GenerationService.Generate + GenerateStream."""

    def __init__(self):
        self._generator = QAGenerator()

    def Generate(self, request, context):
        """Non-streaming generation endpoint."""
        query = request.query
        contexts = list(request.contexts)
        model = request.model or None
        conversation_id = request.conversation_id or None
        params = dict(request.params) if request.params else None

        if not query.strip():
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("query is required")
            from communication.grpc_server.generated import generation_pb2
            return generation_pb2.GenerationResponse()

        try:
            result = self._generator.generate(query, contexts, model, conversation_id, params)

            from communication.grpc_server.generated import generation_pb2

            logger.info(f"gRPC Generate: query='{query[:50]}...', tokens={result['token_count']}")
            return generation_pb2.GenerationResponse(
                content=result["content"],
                is_end=True,
                token_count=result["token_count"],
                finish_reason=result["finish_reason"],
            )
        except GenerationException as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            from communication.grpc_server.generated import generation_pb2
            return generation_pb2.GenerationResponse()

    def GenerateStream(self, request, context):
        """Server-streaming generation endpoint."""
        query = request.query
        contexts = list(request.contexts)
        model = request.model or None
        conversation_id = request.conversation_id or None
        params = dict(request.params) if request.params else None

        if not query.strip():
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("query is required")
            return

        from communication.grpc_server.generated import generation_pb2

        # Bridge async LLM streaming to sync gRPC streaming via a thread+queue
        result_queue: queue.Queue = queue.Queue()

        def run_async_stream():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                async_gen = self._generator.generate_stream(query, contexts, model, conversation_id, params)

                async def consume():
                    async for chunk in async_gen:
                        result_queue.put(chunk)
                    result_queue.put(None)  # sentinel

                loop.run_until_complete(consume())
            except Exception as e:
                logger.error(f"Stream generation error: {e}")
                result_queue.put({"content": "", "is_end": True, "token_count": 0, "finish_reason": "error"})
                result_queue.put(None)

        thread = threading.Thread(target=run_async_stream, daemon=True)
        thread.start()

        try:
            while True:
                try:
                    chunk = result_queue.get(timeout=120)  # 2-minute max wait
                except queue.Empty:
                    break
                if chunk is None:
                    break
                yield generation_pb2.GenerationResponse(
                    content=chunk.get("content", ""),
                    is_end=chunk.get("is_end", False),
                    token_count=chunk.get("token_count", 0),
                    finish_reason=chunk.get("finish_reason", ""),
                )
        except Exception as e:
            logger.error(f"gRPC stream error: {e}")


def create_generation_server() -> grpc.Server:
    """Create and return a gRPC server for GenerationService."""
    from communication.grpc_server.generated import generation_pb2_grpc

    cfg = get_config()["grpc"]
    port = cfg["generation"]["port"]
    max_workers = cfg["generation"]["max_workers"]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    generation_pb2_grpc.add_GenerationServiceServicer_to_server(GenerationServiceServicer(), server)
    server.add_insecure_port(f"{cfg['server']['host']}:{port}")
    logger.info(f"GenerationService gRPC server on port {port}")
    return server
