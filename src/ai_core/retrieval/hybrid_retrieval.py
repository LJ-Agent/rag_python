"""Hybrid retrieval — vector search + BM25 keyword search, weighted fusion + rerank."""

from common.config_loader import get_config
from common.exception.exceptions import RetrievalException
from common.util.logger import get_logger
from infrastructure.bm25.bm25_engine import get_bm25_engine
from infrastructure.llm.llm_adapter import get_llm_adapter
from infrastructure.milvus.milvus_client import get_milvus_client

logger = get_logger()


class HybridRetrieval:
    """Multi-route retrieval: vector + BM25, with weighted fusion and score filtering."""

    def __init__(self):
        cfg = get_config()["retrieval"]
        llm_cfg = get_config()["llm"]
        self._milvus = get_milvus_client()
        self._bm25 = get_bm25_engine()
        self._llm = get_llm_adapter()
        self._vector_weight = cfg["vector_weight"]
        self._bm25_weight = cfg["bm25_weight"]
        self._score_threshold = cfg["score_threshold"]
        self._rerank_top_k = cfg["rerank_top_k"]
        self._use_local_embedding = llm_cfg.get("use_local_embedding", False)

    def retrieve(
        self,
        query: str,
        kb_ids: list[int],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """Execute hybrid retrieval and return ranked results."""
        threshold = score_threshold if score_threshold is not None else self._score_threshold
        try:
            # 1. Vector retrieval (use local or remote embedding)
            if self._use_local_embedding:
                query_embedding = self._llm.embed_local([query])[0]
            else:
                query_embedding = self._llm.embed_single(query)
            vector_results = self._milvus.search(
                query_vector=query_embedding,
                kb_ids=kb_ids,
                top_k=top_k * 2,  # retrieve more for fusion
                score_threshold=threshold,
            )
            # 2. BM25 keyword retrieval
            bm25_results = self._bm25.search(query, top_k=top_k * 2)

            # 3. Weighted fusion (RRF: Reciprocal Rank Fusion), results already sorted by RRF
            fused = self._fuse_results(vector_results, bm25_results)

            result = fused[:top_k]

            sample_scores = [f"{r.get('score',0):.3f}" for r in result[:3]]
            logger.info(
                f"Hybrid retrieval: query='{query[:50]}...', "
                f"kb_ids={kb_ids}, vector={len(vector_results)}, "
                f"bm25={len(bm25_results)}, fused={len(result)}, "
                f"top_scores={sample_scores}"
            )
            return result
        except Exception as e:
            raise RetrievalException(f"Retrieval failed: {e}")

    def _fuse_results(self, vector_results: list[dict], bm25_results: list[dict]) -> list[dict]:
        """Fuse vector and BM25 using weighted RRF for ranking, keep raw similarity scores."""
        rrf_scores: dict[str, float] = {}
        raw_scores: dict[str, float] = {}
        content_map: dict[str, dict] = {}

        # Vector results — track both RRF rank score and raw Milvus similarity
        for rank, r in enumerate(vector_results):
            cid = r["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + self._vector_weight / (60 + rank + 1)
            raw_scores[cid] = r["score"]  # real Milvus similarity (0~1, higher = better)
            content_map[cid] = r

        # BM25 results
        for rank, r in enumerate(bm25_results):
            cid = r["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + self._bm25_weight / (60 + rank + 1)
            if cid not in content_map:
                raw_scores[cid] = r["score"]  # real BM25 score (0~1, higher = better)
                content_map[cid] = r

        fused = []
        for chunk_id in rrf_scores:
            info = content_map[chunk_id]
            info["score"] = round(raw_scores[chunk_id], 4)  # real similarity, not RRF
            fused.append(info)

        # Sort by RRF score to keep correct cross-modal ranking
        fused.sort(key=lambda x: rrf_scores[x["chunk_id"]], reverse=True)
        return fused

    def index_for_bm25(self, documents: list[dict]):
        """Index documents in local BM25 engine (called after chunking+embedding)."""
        self._bm25.add_documents(documents)
        logger.info(f"BM25 indexed: {len(documents)} documents, total={self._bm25.doc_count}")

    def remove_from_bm25(self, document_id: int):
        """Remove document from BM25 index."""
        self._bm25.remove_document(document_id)
