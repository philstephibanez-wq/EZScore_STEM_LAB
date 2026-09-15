from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from stemlab.config import SPEECH_CACHE_DIRNAME


def _speech_cache_path(work_dir: Path, model_name: str, source_name: str = "original") -> Path:
    safe_model = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(model_name))
    safe_source = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(source_name))
    root = work_dir / SPEECH_CACHE_DIRNAME
    root.mkdir(parents=True, exist_ok=True)

    # Backward compatibility: vocals cache keeps the historical filename.
    if safe_source in {"vocals", "demucs-vocals"}:
        return root / f"whisper_{safe_model}.json"
    return root / f"whisper_{safe_source}_{safe_model}.json"


def load_speech_cache(
    work_dir: Path,
    model_name: str,
    source_name: str = "original",
) -> dict[str, Any] | None:
    path = _speech_cache_path(work_dir, model_name, source_name)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_speech_cache(
    work_dir: Path,
    model_name: str,
    payload: dict[str, Any],
    source_name: str = "original",
) -> Path:
    path = _speech_cache_path(work_dir, model_name, source_name)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


@lru_cache(maxsize=3)
def load_whisper_model(model_name: str):
    import whisper
    return whisper.load_model(model_name)


def whisper_available() -> bool:
    try:
        import whisper  # noqa: F401
        return True
    except Exception:
        return False


def torch_device_label() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"CUDA · {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    return "CPU"


def extract_whisper_words(result: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in result.get("segments", []) or []:
        for item in segment.get("words", []) or []:
            text = str(item.get("word", "") or "").strip()
            if not text:
                continue
            start = round(float(item.get("start", 0.0) or 0.0), 3)
            end = round(float(item.get("end", start) or start), 3)
            words.append({"start": start, "end": end, "text": text})
    return words


def transcribe_audio(
    audio_path: Path,
    *,
    model_name: str,
    source_name: str,
) -> dict[str, Any]:
    """Transcribe one source while preserving original-audio time coordinates."""
    import torch

    model = load_whisper_model(model_name)
    result = model.transcribe(
        str(audio_path),
        task="transcribe",
        word_timestamps=True,
        fp16=bool(torch.cuda.is_available()),
        verbose=False,
    )
    return {
        "engine": "openai-whisper",
        "model": str(model_name),
        "source": str(source_name),
        "language": str(result.get("language", "") or ""),
        "text": str(result.get("text", "") or "").strip(),
        "words": extract_whisper_words(result),
    }


def transcribe_vocals(vocals_path: Path, *, model_name: str) -> dict[str, Any]:
    return transcribe_audio(
        vocals_path,
        model_name=model_name,
        source_name="demucs-vocals",
    )


def _is_lexical_text(text: str) -> bool:
    import re
    return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]", str(text or "")))


def latest_cached_speech_payload(
    work_dir: Path,
    source_name: str = "original",
) -> dict[str, Any] | None:
    """Select the best cached speech timeline for one explicit source.

    R10 invariant:
    - lyrics source = original uploaded audio;
    - Demucs vocals are NOT silently substituted for lyrics.
    """
    root = work_dir / SPEECH_CACHE_DIRNAME
    if not root.is_dir():
        return None

    wanted = str(source_name or "original").strip().lower()
    candidates = []

    for path in root.glob("whisper_*.json"):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload_source = str(payload.get("source", "") or "").strip().lower()

            # Historical R8/R9 caches had no explicit source and were produced
            # from vocals.wav. They must never override an R10 original cache.
            if wanted == "original":
                if payload_source not in {"original", "source", "audio-original"}:
                    continue
            elif wanted in {"vocals", "demucs-vocals"}:
                if payload_source not in {"vocals", "demucs-vocals", ""}:
                    continue
            elif payload_source != wanted:
                continue

            words = list(payload.get("words", []) or [])
            valid_words = []
            latest_end = 0.0
            lexical_words = 0
            lexical_end = 0.0

            for item in words:
                text = str(item.get("text", "") or "").strip()
                start = float(item.get("start", 0.0) or 0.0)
                end = float(item.get("end", start) or start)
                if not text or end <= start:
                    continue

                valid_words.append({
                    "start": start,
                    "end": end,
                    "text": text,
                })
                latest_end = max(latest_end, end)

                if _is_lexical_text(text):
                    lexical_words += 1
                    lexical_end = max(lexical_end, end)

            if not valid_words:
                continue

            candidates.append({
                "path": path,
                "payload": payload,
                "valid_words": valid_words,
                "latest_end": latest_end,
                "lexical_end": lexical_end,
                "word_count": len(valid_words),
                "lexical_words": lexical_words,
                "mtime": path.stat().st_mtime_ns,
            })
        except Exception:
            continue

    if not candidates:
        return None

    # Prefer real lexical coverage, not trailing "..." hallucinations.
    candidates.sort(
        key=lambda item: (
            float(item["lexical_end"]),
            int(item["lexical_words"]),
            int(item["word_count"]),
            int(item["mtime"]),
        ),
        reverse=True,
    )

    best = candidates[0]
    payload = dict(best["payload"])
    payload["words"] = best["valid_words"]
    payload["_cache_path"] = str(best["path"])
    payload["_coverage_end"] = float(best["lexical_end"])
    payload["_raw_end"] = float(best["latest_end"])
    payload["_usable_word_count"] = int(best["word_count"])
    payload["_lexical_word_count"] = int(best["lexical_words"])
    return payload


def latest_cached_words(
    work_dir: Path,
    source_name: str = "original",
) -> list[dict[str, Any]]:
    payload = latest_cached_speech_payload(work_dir, source_name=source_name)
    return list((payload or {}).get("words", []) or [])
