"""Task handler functions — connect Kafka tasks to AI core modules."""
from common.config_loader import get_config
from common.enums.status_enums import TaskType
from common.exception.exceptions import TaskException
from common.util.logger import get_logger
from ai_core.parser.document_parser import DocumentParser
from ai_core.chunker.smart_chunker import SmartChunker
from ai_core.embedder.embedding_service import EmbeddingService
from ai_core.retrieval.hybrid_retrieval import HybridRetrieval
from infrastructure.minio.minio_client import get_minio_client
from infrastructure.milvus.milvus_client import get_milvus_client
from infrastructure.bm25.bm25_engine import get_bm25_engine
from .task_dispatcher import TaskContext

logger = get_logger()

_parser = DocumentParser()
_chunker = SmartChunker()
_embedder = EmbeddingService()
_hybrid_retrieval = HybridRetrieval()
_minio = get_minio_client()


def _use_cleaning_service() -> bool:
    """Check if the independent RAG-CLEANING service should be used."""
    cfg = get_config()
    return cfg.get("cleaning", {}).get("enabled", False)


def _clean_via_service(context: TaskContext, original_url: str, file_name: str) -> dict:
    """Delegate document cleaning to RAG-CLEANING gRPC service."""
    from infrastructure.cleaning_client.cleaning_client import clean_document
    from common.util.utils import get_file_extension

    mime_map = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "md": "text/markdown",
        "txt": "text/plain",
    }
    ext = get_file_extension(file_name)
    mime_type = mime_map.get(ext, "application/octet-stream")

    response = clean_document(
        task_id=context.task_id,
        document_id=str(context.document_id),
        kb_id=context.kb_id,
        tenant_id="default",
        file_name=file_name,
        file_url=original_url,
        mime_type=mime_type,
    )

    from communication.grpc_server.generated import cleaning_pb2
    if response.status == cleaning_pb2.FAILED:
        raise TaskException(
            f"Cleaning service failed: {response.error_message}",
            task_id=context.task_id,
        )

    cleaned_path = response.markdown_url
    content_length = 0
    content = b""
    # Try to read the cleaned content to get length
    try:
        content = _minio.get_object(cleaned_path)
        content_length = len(content)
    except Exception:
        pass

    # Safety net: If cleaning service produced too little text (e.g. image-only PDF),
    # fall back to OCR to extract text from the original file
    MIN_VIABLE_CONTENT = 50  # Minimum chars needed for meaningful chunking
    if content_length < MIN_VIABLE_CONTENT:
        ext = get_file_extension(file_name)
        is_image_like = ext in ("pdf", "png", "jpg", "jpeg", "bmp", "tiff", "tif")
        if is_image_like:
            logger.info(
                f"Cleaning service returned only {content_length} chars for {file_name}. "
                f"Attempting OCR fallback for image-based document."
            )
            try:
                ocr_text = _ocr_fallback_direct(original_url, file_name)
                if ocr_text and len(ocr_text) > content_length:
                    # Overwrite cleaned path with OCR result
                    _minio.put_object(cleaned_path, ocr_text.encode("utf-8"), "text/markdown")
                    content_length = len(ocr_text)
                    logger.info(f"OCR fallback succeeded: {content_length} chars extracted")
            except Exception as e:
                logger.warning(f"OCR fallback also failed: {e}")

    logger.info(
        f"File process done (via cleaning service): {file_name} -> {cleaned_path}, "
        f"{content_length} chars, quality={response.quality.overall_score:.2f}"
    )

    return {
        "cleanedPath": cleaned_path,
        "contentLength": content_length,
        "fileName": file_name,
        "qualityScore": response.quality.overall_score,
        "docTitle": response.doc_meta.title,
        "pageCount": response.doc_meta.page_count,
        "wordCount": response.doc_meta.word_count,
    }


def _clean_via_builtin(context: TaskContext, original_url: str, file_name: str) -> dict:
    """Use built-in DocumentParser (legacy fallback). Falls back to OCR for scanned PDFs."""
    file_data = _minio.get_object(original_url)
    try:
        cleaned_text = _parser.parse(file_data, file_name)
    except Exception as e:
        err_msg = str(e)
        if any(kw in err_msg.lower() for kw in ("no text", "scanned", "unsupported", "ocr")):
            logger.info(f"Built-in parser found no text, attempting OCR fallback for {file_name}")
            cleaned_text = _ocr_fallback(file_data, file_name)
        else:
            raise

    cleaned_path = original_url.rsplit(".", 1)[0] + "_cleaned.md"
    _minio.put_object(cleaned_path, cleaned_text.encode("utf-8"), "text/markdown")
    logger.info(f"File process done (built-in): {file_name} -> {cleaned_path}, {len(cleaned_text)} chars")
    return {
        "cleanedPath": cleaned_path,
        "contentLength": len(cleaned_text),
        "fileName": file_name,
    }


def _ocr_fallback(file_data: bytes, file_name: str) -> str:
    """Fallback OCR for scanned/image PDFs using pdf2image + pytesseract/easyocr."""
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    images = []

    if ext == "pdf":
        try:
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(file_data, dpi=200, fmt="png")
            logger.info(f"PDF converted to {len(images)} images for OCR")
        except ImportError:
            raise AIComputeException("pdf2image not available for OCR fallback")
    elif ext in ("png", "jpg", "jpeg", "bmp", "tiff", "tif"):
        images = [file_data]
    else:
        raise AIComputeException(f"OCR fallback not supported for .{ext} files")

    if not images:
        raise AIComputeException("OCR fallback produced no images")

    texts = []
    for i, img in enumerate(images):
        page_text = ""
        try:
            import pytesseract
            from PIL import Image
            from io import BytesIO
            img_data = img if isinstance(img, bytes) else _image_to_bytes(img)
            pil_img = Image.open(BytesIO(img_data)).convert("L")
            page_text = pytesseract.image_to_string(pil_img, lang="chi_sim+eng")
        except ImportError:
            raise AIComputeException("pytesseract not available for OCR — check Tesseract installation")

        if page_text.strip():
            texts.append(page_text.strip())

    result = "\n\n".join(texts)
    if not result.strip():
        raise AIComputeException("OCR fallback produced no text — document may be blank or unsupported")
    logger.info(f"OCR extracted {len(result)} chars from {len(images)} pages")
    return result


def _image_to_bytes(img) -> bytes:
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _ocr_fallback_direct(original_url: str, file_name: str) -> str:
    """Direct OCR fallback — downloads original file from MinIO and runs OCR on it.

    Used when the cleaning service produces too little text (e.g. pure-image PDFs).
    Supports PDF (via pdf2image) and image formats (png/jpg/bmp/tiff).
    """
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    file_data = _minio.get_object(original_url)

    images = []
    if ext == "pdf":
        try:
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(file_data, dpi=200, fmt="png")
            logger.info(f"OCR fallback: PDF converted to {len(images)} page images")
        except ImportError:
            raise TaskException("pdf2image not available for OCR fallback")
    elif ext in ("png", "jpg", "jpeg", "bmp", "tiff", "tif"):
        images = [file_data]
    else:
        return ""  # Not an image-based format, skip OCR

    if not images:
        return ""

    texts = []
    for i, img in enumerate(images):
        page_text = ""
        try:
            import pytesseract
            from PIL import Image
            from io import BytesIO
            img_data = img if isinstance(img, bytes) else _image_to_bytes(img)
            pil_img = Image.open(BytesIO(img_data)).convert("L")
            page_text = pytesseract.image_to_string(pil_img, lang="chi_sim+eng")
        except Exception:
            page_text = ""
        # Fallback: EasyOCR if pytesseract produced nothing
        if not page_text.strip():
            try:
                import easyocr
                import numpy as np
                from PIL import Image
                from io import BytesIO
                img_data = img if isinstance(img, bytes) else _image_to_bytes(img)
                reader_local = easyocr.Reader(["ch_sim", "en"], gpu=False)
                pil_img = Image.open(BytesIO(img_data))
                arr = np.array(pil_img)
                results = reader_local.readtext(arr)
                page_text = "\n".join(r[1] for r in results)
            except Exception:
                continue

        if page_text.strip():
            texts.append(page_text.strip())

    result = "\n\n".join(texts)
    if result.strip():
        logger.info(f"OCR fallback extracted {len(result)} chars from {len(images)} pages")
    return result


def handle_file_process(context: TaskContext) -> dict:
    """Handle FILE_PROCESS task: clean document via RAG-CLEANING service (or built-in fallback)."""
    original_url = context.data.get("originalFileUrl")
    file_name = context.data.get("fileName", "unknown")

    if not original_url:
        raise TaskException("Missing originalFileUrl in task data", task_id=context.task_id)

    if _use_cleaning_service():
        try:
            return _clean_via_service(context, original_url, file_name)
        except Exception as e:
            logger.warning(f"Cleaning service failed, falling back to built-in parser: {e}")
            return _clean_via_builtin(context, original_url, file_name)
    else:
        return _clean_via_builtin(context, original_url, file_name)


def handle_chunk_process(context: TaskContext) -> dict:
    """Handle CHUNK_PROCESS task: download cleaned MD -> chunk only (no embed).
    Returns chunk list for Java to save to DB before human review."""
    cleaned_path = context.data.get("cleanedPath")
    file_name = context.data.get("fileName", "unknown")
    strategy = context.data.get("chunkStrategy", "semantic")
    chunk_config = context.data.get("chunkConfig")

    if not cleaned_path:
        raise TaskException("Missing cleanedPath in task data", task_id=context.task_id)

    # 1. Read cleaned markdown from MinIO
    cleaned_text = _minio.get_object(cleaned_path).decode("utf-8")

    # 2. Chunk only (no embedding, no Milvus)
    config = _parse_chunk_config(chunk_config) if chunk_config else {}
    chunks = _chunker.chunk(cleaned_text, context.document_id, strategy=strategy, **config)

    # 3. Build chunk data list for Java callback
    chunk_list = [
        {
            "chunkId": c.chunk_id,
            "chunkIndex": c.chunk_index,
            "content": c.content,
            "level": c.level,
            "parentId": c.parent_id or "",
            "charCount": len(c.content),
        }
        for c in chunks
    ]

    logger.info(f"Chunk process done: {file_name}, {len(chunks)} chunks (no embed)")
    return {"chunkCount": len(chunks), "fileName": file_name, "chunks": chunk_list}


def handle_embed_process(context: TaskContext) -> dict:
    """Handle EMBED_PROCESS task: embed chunks -> store in Milvus + BM25 (chunks already reviewed)."""
    chunks_data = context.data.get("chunks", [])
    file_name = context.data.get("fileName", "unknown")

    if not chunks_data:
        raise TaskException("Missing chunks in task data", task_id=context.task_id)

    chunk_ids = [c.get("chunkId", c.get("chunk_id", "")) for c in chunks_data]
    chunk_indices = [c.get("chunkIndex", c.get("chunk_index", 0)) for c in chunks_data]
    contents = [c.get("content", "") for c in chunks_data]

    # 1. Embed and store in Milvus
    count = _embedder.embed_chunks(chunk_ids, context.document_id, context.kb_id, chunk_indices, contents)

    # 2. Index in BM25 for keyword retrieval
    bm25_docs = [
        {
            "chunk_id": c.get("chunkId", c.get("chunk_id", "")),
            "document_id": context.document_id,
            "chunk_index": c.get("chunkIndex", c.get("chunk_index", 0)),
            "content": c.get("content", ""),
        }
        for c in chunks_data
    ]
    _hybrid_retrieval.index_for_bm25(bm25_docs)

    logger.info(f"Embed process done: {file_name}, {count} chunks indexed")
    return {"chunkCount": count, "fileName": file_name}


def _parse_chunk_config(raw) -> dict:
    """Parse chunk config from Kafka message (string or dict) into kwargs for SmartChunker."""
    import json
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    if not isinstance(raw, dict):
        return {}
    # Map JSON keys to chunker parameter names
    return {k: v for k, v in raw.items() if v is not None}


def handle_document_delete(context: TaskContext) -> dict:
    """Handle DOCUMENT_DELETE task: remove vectors from Milvus and index from BM25."""
    doc_id = context.document_id
    milvus = get_milvus_client()
    bm25 = get_bm25_engine()

    milvus.delete_by_document(doc_id)
    bm25.remove_document(doc_id)

    logger.info(f"Document cleanup done: docId={doc_id}, milvus_deleted, bm25_removed")
    return {"documentId": doc_id, "status": "cleaned"}


# Handler registry
HANDLERS = {
    TaskType.FILE_PROCESS.value: handle_file_process,
    TaskType.CHUNK_PROCESS.value: handle_chunk_process,
    TaskType.EMBED_PROCESS.value: handle_embed_process,
    "DOCUMENT_DELETE": handle_document_delete,
}


def dispatch_task(context: TaskContext) -> dict:
    """Route task to the correct handler by task_type."""
    handler = HANDLERS.get(context.task_type)
    if not handler:
        raise TaskException(f"Unknown task type: {context.task_type}", task_id=context.task_id)
    return handler(context)
