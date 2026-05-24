"""Milvus vector database client — vector CRUD, similarity search, collection management."""
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient as PyMilvusClient,
    connections,
    utility,
)

from common.config_loader import get_config
from common.exception.exceptions import ResourceException
from common.util.logger import get_logger

logger = get_logger()


class MilvusVectorClient:
    """Milvus client for vector storage and retrieval, per-knowledge-base isolation."""

    def __init__(self):
        cfg = get_config()["milvus"]
        self._host = cfg["host"]
        self._port = cfg["port"]
        self._collection_name = cfg["collection_name"]
        self._dim = cfg["dim"]
        self._index_type = cfg["index_type"]
        self._metric_type = cfg["metric_type"]
        self._nlist = cfg["nlist"]
        self._connect()
        self._init_collection()

    def _connect(self):
        try:
            connections.connect(host=self._host, port=str(self._port))
            logger.info(f"Milvus connected: {self._host}:{self._port}")
        except Exception as e:
            raise ResourceException(f"Milvus connection failed: {e}")

    def _init_collection(self):
        if not utility.has_collection(self._collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="document_id", dtype=DataType.INT64),
                FieldSchema(name="kb_id", dtype=DataType.INT64),
                FieldSchema(name="chunk_index", dtype=DataType.INT32),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
            ]
            schema = CollectionSchema(fields, description="RAG document chunks")
            collection = Collection(self._collection_name, schema)

            index_params = {
                "index_type": self._index_type,
                "metric_type": self._metric_type,
                "params": {"nlist": self._nlist},
            }
            collection.create_index("embedding", index_params)
            collection.load()
            logger.info(f"Milvus collection created: {self._collection_name}")
        else:
            collection = Collection(self._collection_name)
            collection.load()
            logger.info(f"Milvus collection loaded: {self._collection_name}")

    @property
    def _collection(self) -> Collection:
        return Collection(self._collection_name)

    def insert_vectors(
        self,
        chunk_ids: list[str],
        document_id: int,
        kb_id: int,
        chunk_indices: list[int],
        contents: list[str],
        embeddings: list[list[float]],
    ) -> list[int]:
        """Batch insert vectors with metadata. Returns primary keys."""
        data = [chunk_ids, [document_id] * len(chunk_ids), [kb_id] * len(chunk_ids), chunk_indices, contents, embeddings]
        result = self._collection.insert(data)
        self._collection.flush()
        logger.info(f"Milvus inserted {len(chunk_ids)} vectors for doc_id={document_id}")
        return result.primary_keys

    def search(
        self,
        query_vector: list[float],
        kb_ids: list[int],
        top_k: int = 5,
        score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Vector similarity search within specified knowledge bases.

        score_threshold is treated as a similarity threshold (0~1, higher = better).
        Internally converted to L2 distance for filtering.
        """
        search_params = {"metric_type": self._metric_type, "params": {"nprobe": 16}}
        expr = f"kb_id in {kb_ids}" if kb_ids else None
        # Retrieve extra candidates for post-filtering
        results = self._collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=max(top_k * 2, 10),
            expr=expr,
            output_fields=["chunk_id", "document_id", "kb_id", "chunk_index", "content"],
        )
        hits = []
        for result in results[0]:
            # L2 metric: smaller distance = more similar
            # Convert L2 distance to similarity score (0~1, higher = better)
            # For normalized vectors: L2 ∈ [0, 2], similarity = 1 - L2/2
            similarity = 1.0 - result.distance / 2.0
            if similarity >= score_threshold:
                hits.append({
                    "chunk_id": result.entity.get("chunk_id"),
                    "document_id": result.entity.get("document_id"),
                    "chunk_index": result.entity.get("chunk_index"),
                    "content": result.entity.get("content"),
                    "score": round(similarity, 4),
                })
        return hits

    def delete_by_document(self, document_id: int):
        """Delete all vectors for a given document_id."""
        expr = f"document_id == {document_id}"
        self._collection.delete(expr)
        self._collection.flush()
        logger.info(f"Milvus deleted vectors for doc_id={document_id}")

    def delete_by_kb(self, kb_id: int):
        """Delete all vectors for a given knowledge base."""
        expr = f"kb_id == {kb_id}"
        self._collection.delete(expr)
        self._collection.flush()
        logger.info(f"Milvus deleted vectors for kb_id={kb_id}")

    def count(self) -> int:
        return self._collection.num_entities

    def close(self):
        connections.disconnect("default")


# Singleton
_milvus_client: MilvusVectorClient | None = None


def get_milvus_client() -> MilvusVectorClient:
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusVectorClient()
    return _milvus_client
