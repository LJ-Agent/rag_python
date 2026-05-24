"""Vector embedding service — batch text vectorization + Milvus storage."""
from common.config_loader import get_config
from common.exception.exceptions import AIComputeException
from common.util.logger import get_logger
from infrastructure.llm.llm_adapter import get_llm_adapter
from infrastructure.milvus.milvus_client import get_milvus_client

logger = get_logger()


class EmbeddingService:
    """Compute embeddings for chunks and manage vector lifecycle in Milvus."""

    def __init__(self):
        self._llm = get_llm_adapter()
        self._milvus = get_milvus_client()
        cfg = get_config()["llm"]
        self._use_local = cfg.get("use_local_embedding", False) or not cfg.get("api_key")
        self._batch_size = 20

    def embed_chunks(
        self,
        chunk_ids: list[str],
        document_id: int,
        kb_id: int,
        chunk_indices: list[int],
        contents: list[str],
    ) -> int:
        """Batch embed chunks and store in Milvus. Returns count stored."""
        all_ids = []
        for i in range(0, len(contents), self._batch_size):
            batch_contents = contents[i : i + self._batch_size]
            if self._use_local:
                embeddings = self._llm.embed_local(batch_contents)
            else:
                embeddings = self._llm.embed_texts(batch_contents)

            batch_ids = chunk_ids[i : i + self._batch_size]
            batch_indices = chunk_indices[i : i + self._batch_size]
            pks = self._milvus.insert_vectors(
                chunk_ids=batch_ids,
                document_id=document_id,
                kb_id=kb_id,
                chunk_indices=batch_indices,
                contents=batch_contents,
                embeddings=embeddings,
            )
            all_ids.extend(pks)
            logger.info(f"Embedded batch {i // self._batch_size + 1}: {len(batch_ids)} chunks")

        return len(all_ids)

    def delete_document_vectors(self, document_id: int):
        """Remove all vectors for a document (called on doc delete/offline)."""
        self._milvus.delete_by_document(document_id)

    def update_document_vectors(
        self,
        document_id: int,
        kb_id: int,
        chunk_ids: list[str],
        chunk_indices: list[int],
        contents: list[str],
    ):
        """Full refresh: delete old vectors, insert new ones."""
        self._milvus.delete_by_document(document_id)
        return self.embed_chunks(chunk_ids, document_id, kb_id, chunk_indices, contents)
