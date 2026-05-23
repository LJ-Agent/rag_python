"""Utility functions aligned with Java util classes (Md5Util, etc.)."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def md5_file(file_path: str | Path) -> str:
    """Compute MD5 digest of a file, identical to Java Md5Util."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def md5_bytes(data: bytes) -> str:
    """Compute MD5 digest of bytes."""
    return hashlib.md5(data).hexdigest()


def json_dumps(obj: Any) -> str:
    """Serialize to JSON string with default encoding."""

    def default(o):
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    return json.dumps(obj, ensure_ascii=False, default=default)


def json_loads(s: str) -> Any:
    """Deserialize JSON string."""
    return json.loads(s)


def now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format (aligned with Java)."""
    return datetime.now(timezone.utc).isoformat()


def format_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_file_extension(filename: str) -> str:
    """Extract lowercase file extension."""
    return Path(filename).suffix.lower().lstrip(".")


def clean_text(text: str) -> str:
    """Basic text cleaning: remove excessive whitespace, normalize newlines."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)
    return text.strip()


def truncate_text(text: str, max_len: int = 200) -> str:
    """Truncate text to max_len with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def validate_file_type(filename: str, supported: set[str] | None = None) -> bool:
    if supported is None:
        supported = {"pdf", "md", "txt", "docx"}
    ext = get_file_extension(filename)
    return ext in supported
