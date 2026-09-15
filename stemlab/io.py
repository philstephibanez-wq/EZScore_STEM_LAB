from __future__ import annotations

import hashlib
from pathlib import Path

from .config import CACHE_DIR


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".wma"}:
        return suffix
    return ".bin"


def write_uploaded_source(uploaded) -> tuple[str, Path]:
    data = uploaded.getvalue()
    audio_hash = sha256_bytes(data)
    work_dir = CACHE_DIR / audio_hash
    work_dir.mkdir(parents=True, exist_ok=True)
    source = work_dir / f"source{safe_suffix(uploaded.name)}"
    if not source.exists() or source.stat().st_size != len(data):
        source.write_bytes(data)
    return audio_hash, source
