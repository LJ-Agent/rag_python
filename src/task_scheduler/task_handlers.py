"""Task handler functions — connect Kafka tasks to AI core modules."""
from common.enums.status_enums import TaskType
from common.exception.exceptions import TaskException
from common.util.logger import get_logger
from ai_core.parser.document_parser import DocumentParser
from ai_core.chunker.smart_chunker import SmartChunker
from ai_core.embedder.embedding_service import EmbeddingService
from ai_core.retrieval.hybrid_retrieval import HybridRetrieval
from infrastructure.minio.minio_client import get_minio_client
from infrastructure.bm25.bm25_engine import get_bm25_engine
from .task_dispatcher import TaskContext

logger = get_logger()

_parser = DocumentParser()
_chunker = SmartChunker()
_embedder = EmbeddingService()
_hybrid_retrieval = HybridRetrieval()
_minio = get_minio_client()


def handle_file_process(context: TaskContext) -> dict:
    """Handle FILE_PROCESS task: download from MinIO -> parse -> clean -> write cleaned MD back to MinIO."""
    original_url = context.data.get("originalFileUrl")
    file_name = context.data.get("fileName", "unknown")

    if not original_url:
        raise TaskException("Missing originalFileUrl in task data", task_id=context.task_id)

    # 1. Read original file from MinIO
    file_data = _minio.get_object(original_url)
    # 2. Parse and clean
    cleaned_text = _parser.parse(file_data, file_name)
    # 3. Write cleaned MD back to MinIO
    cleaned_path = original_url.rsplit(".", 1)[0] + "_cleaned.md"
    _minio.put_object(cleaned_path, cleaned_text.encode("utf-8"), "text/markdown")
    logger.info(f"File process done: {file_name} -> {cleaned_path}, {len(cleaned_text)} chars")

    return {
        "cleanedPath": cleaned_path,
        "contentLength": len(cleaned_text),
        "fileName": file_name,
    }


def handle_chunk_process(context: TaskContext) -> dict:
    """Handle CHUNK_PROCESS task: download cleaned MD -> chunk -> embed -> store in Milvus + BM25."""
    cleaned_path = context.data.get("cleanedPath")
    file_name = context.data.get("fileName", "unknown")
    strategy = context.data.get("chunkStrategy", "semantic")

    if not cleaned_path:
        raise TaskException("Missing cleanedPath in task data", task_id=context.task_id)

    # 1. Read cleaned markdown from MinIO
    cleaned_text = _minio.get_object(cleaned_path).decode("utf-8")

    # 2. Chunk
    chunks = _chunker.chunk(cleaned_text, context.document_id, strategy=strategy)

    # 3. Embed and store in Milvus
    chunk_ids = [c.chunk_id for c in chunks]
    chunk_indices = [c.chunk_index for c in chunks]
    contents = [c.content for c in chunks]
    count = _embedder.embed_chunks(chunk_ids, context.document_id, context.kb_id, chunk_indices, contents)

    # 4. Index in BM25 for keyword retrieval
    bm25_docs = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "chunk_index": c.chunk_index,
            "content": c.content,
        }
        for c in chunks
    ]
    _hybrid_retrieval.index_for_bm25(bm25_docs)

    logger.info(f"Chunk process done: {file_name}, {count} chunks indexed")
    return {"chunkCount": count, "fileName": file_name}


# Handler registry
HANDLERS = {
    TaskType.FILE_PROCESS.value: handle_file_process,
    TaskType.CHUNK_PROCESS.value: handle_chunk_process,
}


def dispatch_task(context: TaskContext) -> dict:
    """Route task to the correct handler by task_type."""
    handler = HANDLERS.get(context.task_type)
    if not handler:
        raise TaskException(f"Unknown task type: {context.task_type}", task_id=context.task_id)
    return handler(context)
