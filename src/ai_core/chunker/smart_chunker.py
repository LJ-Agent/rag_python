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

    def chunk(self, text: str, document_id: int, strategy: str | None = None) -> list[Chunk]:
        """Chunk text and return ordered list of Chunk objects."""
        strategy = strategy or self._strategy
        chunker: Callable[[str, int], list[Chunk]] = {
            "fixed": self._fixed_chunk,
            "hierarchical": self._hierarchical_chunk,
            "semantic": self._semantic_chunk,
        }[strategy]
        chunks = chunker(text, document_id)
        # Enforce max chunk size (Milvus VarChar limit: 65535)
        chunks = self._enforce_max_size(chunks)
        # Filter out chunks that are too short
        chunks = [c for c in chunks if len(c.content.strip()) >= self._min_size]
        logger.info(f"Chunking [{strategy}]: doc_id={document_id}, {len(chunks)} chunks")
        return chunks

    def _enforce_max_size(self, chunks: list[Chunk]) -> list[Chunk]:
        """Split any chunk exceeding max_chunk_size into smaller pieces."""
        result = []
        for c in chunks:
            if len(c.content) <= self._max_size:
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
        while len(text) > self._max_size:
            split_at = self._max_size
            # Try paragraph break
            para_break = text.rfind("\n\n", 0, self._max_size)
            if para_break > self._max_size // 2:
                split_at = para_break + 2
            else:
                # Try sentence break (Chinese period)
                sent_break = text.rfind("。", 0, self._max_size)
                if sent_break > self._max_size // 2:
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
            end = min(start + self._default_size, len(text))
            if end < len(text):
                # Try to break at sentence boundary
                break_point = text.rfind("。", start, end)
                if break_point == -1:
                    break_point = text.rfind("\n", start, end)
                if break_point > start:
                    end = break_point + 1
            content = text[start:end].strip()
            chunk_id = f"chunk_{document_id}_{idx}"
            chunks.append(Chunk(chunk_id=chunk_id, document_id=document_id, content=content, chunk_index=idx))
            start = end - self._overlap
            idx += 1
        return chunks

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
            if len(content) > self._default_size * 2:
                # Sub-split oversized sections
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
            if len(buffer) + len(para) > self._default_size and buffer:
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
