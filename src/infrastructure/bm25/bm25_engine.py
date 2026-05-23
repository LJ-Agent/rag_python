"""BM25 keyword retrieval engine — jieba tokenization + BM25 ranking."""
from collections import defaultdict

import jieba
import numpy as np


class BM25Engine:
    """Local BM25 retrieval engine for keyword-based recall."""

    def __init__(self):
        self._corpus: list[list[str]] = []
        self._documents: list[dict] = []  # parallel list with metadata
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}
        self._k1: float = 1.5
        self._b: float = 0.75
        self._built: bool = False

    def add_documents(self, documents: list[dict]):
        """Index documents. Each doc: {chunk_id, document_id, content, chunk_index, ...}."""
        for doc in documents:
            tokens = self._tokenize(doc["content"])
            self._corpus.append(tokens)
            self._documents.append(doc)
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        return [w for w in jieba.cut(text) if w.strip()]

    def _build_index(self):
        self._doc_len = [len(doc) for doc in self._corpus]
        self._avgdl = sum(self._doc_len) / max(len(self._corpus), 1)
        self._idf = self._compute_idf()
        self._built = True

    def _compute_idf(self) -> dict[str, float]:
        n = len(self._corpus)
        df = defaultdict(int)
        for doc in self._corpus:
            for word in set(doc):
                df[word] += 1
        return {
            word: np.log((n - freq + 0.5) / (freq + 0.5) + 1.0)
            for word, freq in df.items()
        }

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        doc_tokens = self._corpus[doc_idx]
        doc_len = self._doc_len[doc_idx]
        tf = defaultdict(int)
        for t in doc_tokens:
            tf[t] += 1
        score = 0.0
        for token in query_tokens:
            idf = self._idf.get(token, 0.0)
            freq = tf.get(token, 0)
            score += idf * (freq * (self._k1 + 1)) / (freq + self._k1 * (1 - self._b + self._b * doc_len / self._avgdl))
        return score

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve top_k documents by BM25 score."""
        if not self._built:
            return []
        query_tokens = self._tokenize(query)
        scores = [(self._score(query_tokens, i), i) for i in range(len(self._corpus))]
        scores = [(s, i) for s, i in scores if s > 0]
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        max_score = scores[0][0] if scores else 1.0
        for s, idx in scores[:top_k]:
            doc = dict(self._documents[idx])
            doc["score"] = s / max_score  # normalize
            results.append(doc)
        return results

    def remove_document(self, document_id: int):
        """Remove all chunks belonging to a document."""
        indices = [i for i, d in enumerate(self._documents) if d.get("document_id") == document_id]
        for i in reversed(indices):
            del self._corpus[i]
            del self._documents[i]
        self._build_index()

    def clear(self):
        self._corpus.clear()
        self._documents.clear()
        self._doc_len.clear()
        self._idf.clear()
        self._built = False

    @property
    def doc_count(self) -> int:
        return len(self._corpus)


# Singleton
_bm25_engine: BM25Engine | None = None


def get_bm25_engine() -> BM25Engine:
    global _bm25_engine
    if _bm25_engine is None:
        _bm25_engine = BM25Engine()
    return _bm25_engine
