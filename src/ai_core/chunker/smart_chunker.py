"""Smart text chunker — fixed, hierarchical, and semantic chunking strategies."""
import re
from dataclasses import dataclass, field
from typing import Callable

from common.config_loader import get_config
from common.util.logger import get_logger

logger = get_logger()


@dataclass
class Chunk:
    """Chunk metadata, aligned with Java DocumentChunk + entity fields."""

    chunk_id: str
    document_id: int
    content: str
    chunk_index: int
    level: int = 0  # heading level for hierarchical chunks
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)


class SmartChunker:
    """Adaptive chunking with three strategies: fixed-size, hierarchical, semantic."""

    def __init__(self):
        cfg = get_config()["chunk"]
        self._default_size = cfg["default_size"]
        self._overlap = cfg["overlap"]
        self._min_size = cfg["min_chunk_size"]
        self._max_size = cfg.get("max_chunk_size", 60000)
        self._strategy = cfg.get("strategy", "semantic")

    def chunk(self, text: str, document_id: int, strategy: str | None = None, **config) -> list[Chunk]:
        """Chunk text and return ordered list of Chunk objects.

        Accepts per-task config overrides via kwargs:
        - chunk_size: int (used by fixed, recursive, semantic)
        - chunk_overlap: int (used by fixed, recursive)
        - separators: list[str] (used by recursive)
        - similarity_threshold: float (used by semantic)
        - min_chunk_size: int (used by all)
        - max_chunk_size: int (used by all)
        - topic_sensitivity: float (used by topic)
        - max_chunk_length: int (used by topic)
        - coarse_chunk_threshold: int (used by hybrid)
        - fine_chunk_size: int (used by hybrid)
        - fine_chunk_overlap: int (used by hybrid)
        """
        strategy = strategy or self._strategy
        # Apply per-task config overrides
        task_size = config.get("chunk_size") or config.get("fine_chunk_size") or self._default_size
        task_overlap = config.get("chunk_overlap") or config.get("fine_chunk_overlap") or self._overlap
        task_min = config.get("min_chunk_size") or self._min_size
        task_max = config.get("max_chunk_size") or config.get("max_chunk_length") or self._max_size
        task_separators = config.get("separators", ["\n\n", "\n", "。", "。"])
        task_similarity = config.get("similarity_threshold", 0.5)
        task_topic_sensitivity = config.get("topic_sensitivity", 0.7)
        task_coarse_threshold = config.get("coarse_chunk_threshold", 2000)

        chunker: Callable[[str, int], list[Chunk]] = {
            "fixed": self._fixed_chunk,
            "hierarchical": self._hierarchical_chunk,
            "semantic": self._semantic_chunk,
            "recursive": self._recursive_chunk,
            "topic": self._topic_chunk,
            "hybrid": self._hybrid_chunk,
        }[strategy]

        # Store per-task params for use by chunk methods
        self._task_size = task_size
        self._task_overlap = task_overlap
        self._task_min = task_min
        self._task_max = task_max
        self._task_separators = task_separators
        self._task_similarity = task_similarity
        self._task_topic_sensitivity = task_topic_sensitivity
        self._task_coarse_threshold = task_coarse_threshold

        chunks = chunker(text, document_id)
        # Enforce max chunk size (Milvus VarChar limit: 65535)
        chunks = self._enforce_max_size(chunks)
        # Filter out chunks that are too short
        chunks = [c for c in chunks if len(c.content.strip()) >= self._task_min]
        logger.info(f"Chunking [{strategy}]: doc_id={document_id}, {len(chunks)} chunks")
        return chunks

    def _enforce_max_size(self, chunks: list[Chunk]) -> list[Chunk]:
        """Split any chunk exceeding max_chunk_size into smaller pieces."""
        result = []
        for c in chunks:
            if len(c.content) <= self._task_max:
                result.append(c)
            else:
                sub_texts = self._split_long_text(c.content)
                for i, sub in enumerate(sub_texts):
                    result.append(Chunk(
                        chunk_id=f"{c.chunk_id}_p{i}",
                        document_id=c.document_id,
                        content=sub,
                        chunk_index=c.chunk_index * 1000 + i,
                        level=c.level,
                        parent_id=c.parent_id,
                        metadata=c.metadata,
                    ))
        return result

    def _split_long_text(self, text: str) -> list[str]:
        """Split long text into pieces <= max_size, preferring paragraph/sentence breaks."""
        pieces = []
        while len(text) > self._task_max:
            split_at = self._task_max
            para_break = text.rfind("\n\n", 0, self._task_max)
            if para_break > self._task_max // 2:
                split_at = para_break + 2
            else:
                sent_break = text.rfind("。", 0, self._task_max)
                if sent_break > self._task_max // 2:
                    split_at = sent_break + 1
            pieces.append(text[:split_at].strip())
            text = text[split_at:].strip()
        if text.strip():
            pieces.append(text.strip())
        return pieces

    def _fixed_chunk(self, text: str, document_id: int) -> list[Chunk]:
        """Fixed-size chunking with overlap, avoiding mid-sentence breaks."""
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self._task_size, len(text))
            if end < len(text):
                break_point = text.rfind("。", start, end)
                if break_point == -1:
                    break_point = text.rfind("\n", start, end)
                if break_point > max(start, 0):
                    end = break_point + 1
            content = text[max(start, 0):end].strip()
            chunk_id = f"chunk_{document_id}_{idx}"
            chunks.append(Chunk(chunk_id=chunk_id, document_id=document_id, content=content, chunk_index=idx))
            # Advance start, ensuring forward progress
            next_start = end - self._task_overlap
            if next_start <= max(start, 0):
                break
            start = next_start
            idx += 1
        return chunks

    def _recursive_chunk(self, text: str, document_id: int) -> list[Chunk]:
        """Recursive character chunking — split by separators in priority order, then by size."""
        import re as _re
        chunks = []
        idx = 0
        self._recursive_split(text, document_id, self._task_separators, chunks, idx, 0)
        return chunks

    def _recursive_split(self, text: str, doc_id: int, separators: list[str],
                         result: list, idx_start: int, depth: int) -> int:
        """Recursively split text using separator hierarchy."""
        idx = idx_start
        if depth >= len(separators) or len(text) <= self._task_size:
            chunk_id = f"chunk_{doc_id}_{idx}"
            result.append(Chunk(chunk_id=chunk_id, document_id=doc_id, content=text.strip(), chunk_index=idx))
            return idx + 1
        sep = separators[depth]
        if not sep:
            chunk_id = f"chunk_{doc_id}_{idx}"
            result.append(Chunk(chunk_id=chunk_id, document_id=doc_id, content=text.strip(), chunk_index=idx))
            return idx + 1
        parts = text.split(sep)
        buffer = ""
        for part in parts:
            combined = buffer + (sep if buffer else "") + part
            if len(combined) > self._task_size and buffer:
                chunk_id = f"chunk_{doc_id}_{idx}"
                result.append(Chunk(chunk_id=chunk_id, document_id=doc_id, content=buffer.strip(), chunk_index=idx))
                buffer = part
                idx += 1
            else:
                buffer = combined
        if buffer.strip():
            idx = self._recursive_split(buffer.strip(), doc_id, separators, result, idx, depth + 1)
        return idx

    def _hierarchical_chunk(self, text: str, document_id: int) -> list[Chunk]:
        """Header-based hierarchical chunking — splits on markdown headings."""
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        sections = []
        last_pos = 0
        last_level = 0
        for m in heading_pattern.finditer(text):
            if last_pos < m.start():
                sections.append((last_level, text[last_pos : m.start()].strip()))
            last_level = len(m.group(1))
            last_pos = m.end()
        if last_pos < len(text):
            sections.append((last_level, text[last_pos:].strip()))

        chunks = []
        for idx, (level, content) in enumerate(sections):
            if not content.strip():
                continue
            if len(content) > self._task_size * 2:
                sub = self._fixed_chunk(content, document_id)
                for s in sub:
                    s.level = level
                chunks.extend(sub)
            else:
                chunk_id = f"chunk_{document_id}_{idx}"
                chunks.append(Chunk(chunk_id=chunk_id, document_id=document_id, content=content, chunk_index=idx, level=level))
        return chunks

    def _semantic_chunk(self, text: str, document_id: int) -> list[Chunk]:
        """Semantic chunking — split on double-newline (paragraphs), merge short ones."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        buffer = ""
        idx = 0
        for para in paragraphs:
            if len(buffer) + len(para) > self._task_size and buffer:
                chunk_id = f"chunk_{document_id}_{idx}"
                chunks.append(Chunk(chunk_id=chunk_id, document_id=document_id, content=buffer.strip(), chunk_index=idx))
                buffer = para
                idx += 1
            else:
                buffer += ("\n\n" if buffer else "") + para
        if buffer.strip():
            chunk_id = f"chunk_{document_id}_{idx}"
            chunks.append(Chunk(chunk_id=chunk_id, document_id=document_id, content=buffer.strip(), chunk_index=idx))
        return chunks

    def _topic_chunk(self, text: str, document_id: int) -> list[Chunk]:
        """Topic-based chunking — detect topic shifts via paragraph similarity heuristics."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return []
        chunks = []
        buffer = paragraphs[0]
        idx = 0
        for i in range(1, len(paragraphs)):
            similarity = self._paragraph_similarity(buffer, paragraphs[i])
            if similarity < self._task_topic_sensitivity and len(buffer) > self._task_min:
                chunk_id = f"chunk_{document_id}_{idx}"
                chunks.append(Chunk(chunk_id=chunk_id, document_id=document_id, content=buffer.strip(), chunk_index=idx))
                buffer = paragraphs[i]
                idx += 1
            elif len(buffer) + len(paragraphs[i]) > self._task_max:
                chunk_id = f"chunk_{document_id}_{idx}"
                chunks.append(Chunk(chunk_id=chunk_id, document_id=document_id, content=buffer.strip(), chunk_index=idx))
                buffer = paragraphs[i]
                idx += 1
            else:
                buffer += "\n\n" + paragraphs[i]
        if buffer.strip():
            chunk_id = f"chunk_{document_id}_{idx}"
            chunks.append(Chunk(chunk_id=chunk_id, document_id=document_id, content=buffer.strip(), chunk_index=idx))
        return chunks

    def _paragraph_similarity(self, a: str, b: str) -> float:
        """Estimate paragraph similarity via shared word ratio (simple Jaccard-like)."""
        words_a = set(a)
        words_b = set(b)
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def _hybrid_chunk(self, text: str, document_id: int) -> list[Chunk]:
        """Hybrid chunking — coarsely split by structure, then refine with recursive chunking."""
        # Phase 1: Coarse structural split
        coarse_chunks = self._hierarchical_chunk(text, document_id)
        # Phase 2: Refine chunks that exceed threshold
        final = []
        for c in coarse_chunks:
            if len(c.content) > self._task_coarse_threshold:
                sub_chunks = self._fixed_chunk(c.content, document_id)
                for sc in sub_chunks:
                    sc.parent_id = c.chunk_id
                    sc.level = c.level
                final.extend(sub_chunks)
            else:
                final.append(c)
        # Re-index
        for i, c in enumerate(final):
            c.chunk_index = i
            c.chunk_id = f"chunk_{document_id}_{i}"
        return final
