"""Multi-format document parser + text cleaner — PDF, MD, Word, TXT."""
import io
from pathlib import Path

from pypdf import PdfReader

from common.exception.exceptions import AIComputeException
from common.util.logger import get_logger
from common.util.utils import clean_text, get_file_extension

logger = get_logger()

SUPPORTED_EXTENSIONS = {"pdf", "md", "txt", "docx"}


class DocumentParser:
    """Parse documents into raw text, then clean to standardized Markdown."""

    def parse(self, file_data: bytes, filename: str) -> str:
        """Parse raw file bytes into clean text. Dispatches by file extension."""
        ext = get_file_extension(filename)
        if ext not in SUPPORTED_EXTENSIONS:
            raise AIComputeException(f"Unsupported file type: {ext}")

        parser_method = {
            "pdf": self._parse_pdf,
            "md": self._parse_md,
            "txt": self._parse_txt,
            "docx": self._parse_docx,
        }[ext]

        try:
            raw_text = parser_method(file_data)
            return self._clean_to_markdown(raw_text, ext)
        except Exception as e:
            raise AIComputeException(f"Document parse failed: {filename} — {e}")

    def _parse_pdf(self, data: bytes) -> str:
        reader = PdfReader(io.BytesIO(data))
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        if not texts:
            raise AIComputeException("PDF extraction produced no text (possibly scanned image)")
        return "\n\n".join(texts)

    def _decode_text(self, data: bytes, label: str) -> str:
        """Try multiple encodings to decode file bytes, avoiding mojibake."""
        for encoding in ["utf-8", "gbk", "gb2312", "latin-1"]:
            try:
                text = data.decode(encoding)
                logger.debug(f"{label}: decoded as {encoding}")
                return text
            except (UnicodeDecodeError, LookupError):
                continue
        # Last resort: UTF-8 with replacement
        logger.warning(f"{label}: all encodings failed, using utf-8 with replace")
        return data.decode("utf-8", errors="replace")

    def _parse_md(self, data: bytes) -> str:
        return self._decode_text(data, "MD")

    def _parse_txt(self, data: bytes) -> str:
        return self._decode_text(data, "TXT")

    def _parse_docx(self, data: bytes) -> str:
        from docx import Document

        doc = Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    def _clean_to_markdown(self, text: str, source_format: str) -> str:
        """Normalize text to clean Markdown format."""
        if source_format == "md":
            return text  # already markdown, keep as-is
        # Basic cleaning
        text = clean_text(text)
        # Ensure it's valid for downstream processing
        if not text.strip():
            raise AIComputeException("Document content is empty after cleaning")
        return text
