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
        # Filter out chunks that are too short
        chunks = [c for c in chunks if len(c.content.strip()) >= self._min_size]
        logger.info(f"Chunking [{strategy}]: doc_id={document_id}, {len(chunks)} chunks")
        return chunks

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
