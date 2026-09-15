from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import librosa
import numpy as np


PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

_MAJOR_TEMPLATE = np.zeros(12, dtype=float)
_MAJOR_TEMPLATE[[0, 4, 7]] = [1.0, 0.82, 0.88]

_MINOR_TEMPLATE = np.zeros(12, dtype=float)
_MINOR_TEMPLATE[[0, 3, 7]] = [1.0, 0.82, 0.88]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _chord_templates() -> list[tuple[str, np.ndarray]]:
    result = []
    for root in range(12):
        result.append((PC_NAMES[root], np.roll(_MAJOR_TEMPLATE, root)))
        result.append((PC_NAMES[root] + "m", np.roll(_MINOR_TEMPLATE, root)))
    return result


_TEMPLATES = _chord_templates()


def _estimate_chord(chroma: np.ndarray, bass_chroma: np.ndarray | None = None) -> tuple[str, float]:
    """Estimate a pragmatic major/minor chord from chroma.

    `other` dominates. `bass` is a weak root-support cue only, specifically so
    bass passing notes cannot dictate the chord.
    """
    c = np.asarray(chroma, dtype=float)
    if not np.any(np.isfinite(c)) or float(np.sum(c)) <= 1e-8:
        return "N", 0.0

    c = np.nan_to_num(c, nan=0.0)
    c = c / max(1e-8, float(np.max(c)))

    best_name = "N"
    best_score = -1.0

    for name, template in _TEMPLATES:
        score = _cosine(c, template)

        if bass_chroma is not None:
            b = np.asarray(bass_chroma, dtype=float)
            if np.any(np.isfinite(b)) and float(np.sum(b)) > 1e-8:
                b = np.nan_to_num(b, nan=0.0)
                b /= max(1e-8, float(np.max(b)))
                root_name = name[:-1] if name.endswith("m") else name
                root = PC_NAMES.index(root_name)
                score += 0.10 * float(b[root])

        if score > best_score:
            best_name = name
            best_score = score

    # Deliberately conservative: low template agreement becomes N.
    if best_score < 0.47:
        return "N", float(best_score)
    return best_name, float(best_score)


def analyze_harmony(
    *,
    other_path: Path,
    bass_path: Path,
    drums_path: Path,
    beats_per_bar: int = 4,
    sr: int = 22050,
) -> dict[str, Any]:
    """Build a measure-level harmonic timeline from the separated stems."""
    y_other, sr = librosa.load(str(other_path), sr=sr, mono=True)
    y_bass, _ = librosa.load(str(bass_path), sr=sr, mono=True)
    y_drums, _ = librosa.load(str(drums_path), sr=sr, mono=True)

    hop = 512

    tempo, beat_frames = librosa.beat.beat_track(
        y=y_drums,
        sr=sr,
        hop_length=hop,
        trim=False,
    )
    tempo = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0

    if beat_frames is None or len(beat_frames) < max(8, beats_per_bar * 2):
        # Fallback to mixed rhythmic information from `other`.
        tempo, beat_frames = librosa.beat.beat_track(
            y=y_other,
            sr=sr,
            hop_length=hop,
            trim=False,
        )
        tempo = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0

    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    chroma_other = librosa.feature.chroma_cqt(
        y=y_other,
        sr=sr,
        hop_length=hop,
    )
    chroma_bass = librosa.feature.chroma_cqt(
        y=y_bass,
        sr=sr,
        hop_length=hop,
    )

    beat_chords = []
    for i, frame0 in enumerate(beat_frames):
        frame1 = (
            beat_frames[i + 1]
            if i + 1 < len(beat_frames)
            else min(chroma_other.shape[1], frame0 + max(1, int(sr / hop / max(1.0, tempo / 60.0))))
        )
        frame0 = max(0, min(int(frame0), chroma_other.shape[1] - 1))
        frame1 = max(frame0 + 1, min(int(frame1), chroma_other.shape[1]))

        c = np.mean(chroma_other[:, frame0:frame1], axis=1)

        b0 = min(frame0, chroma_bass.shape[1] - 1)
        b1 = max(b0 + 1, min(frame1, chroma_bass.shape[1]))
        b = np.mean(chroma_bass[:, b0:b1], axis=1)

        chord, confidence = _estimate_chord(c, b)
        beat_chords.append({
            "index": i,
            "time": float(beat_times[i]),
            "chord": chord,
            "confidence": round(confidence, 4),
        })

    measures = []
    bpb = max(2, int(beats_per_bar))
    for start in range(0, len(beat_chords), bpb):
        group = beat_chords[start:start + bpb]
        if len(group) < max(2, bpb // 2):
            break

        tokens = [str(item["chord"]) for item in group]
        # Collapse only consecutive duplicates for motif comparison readability.
        compact = []
        for token in tokens:
            if not compact or compact[-1] != token:
                compact.append(token)

        t0 = float(group[0]["time"])
        if start + bpb < len(beat_chords):
            t1 = float(beat_chords[start + bpb]["time"])
        elif len(group) >= 2:
            dt = max(0.15, float(group[-1]["time"]) - float(group[-2]["time"]))
            t1 = float(group[-1]["time"]) + dt
        else:
            t1 = t0 + 1.0

        measures.append({
            "measure": len(measures) + 1,
            "time_start": t0,
            "time_end": t1,
            "beat_chords": tokens,
            "pattern": "|".join(compact),
        })

    return {
        "tempo": round(tempo, 3),
        "beats_per_bar": bpb,
        "beat_count": len(beat_chords),
        "measure_count": len(measures),
        "beats": beat_chords,
        "measures": measures,
    }


def _measure_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, str(a), str(b), autojunk=False).ratio()


def _phrase_similarity(patterns: list[str], a0: int, b0: int, length: int) -> float:
    if length <= 0:
        return 0.0
    return float(np.mean([
        _measure_similarity(patterns[a0 + k], patterns[b0 + k])
        for k in range(length)
    ]))


def detect_harmonic_motifs(
    measures: list[dict[str, Any]],
    *,
    min_measures: int = 6,
    max_measures: int = 20,
    threshold: float = 0.72,
    max_results: int = 12,
) -> list[dict[str, Any]]:
    """Find repeated multi-measure chord progressions, not short riffs."""
    patterns = [str(m.get("pattern", "") or "") for m in measures]
    n = len(patterns)
    results = []

    if n < min_measures * 2:
        return results

    max_len = min(int(max_measures), n // 2)
    for length in range(int(min_measures), max_len + 1):
        for a0 in range(0, n - length + 1):
            for b0 in range(a0 + length, n - length + 1):
                score = _phrase_similarity(patterns, a0, b0, length)
                if score < threshold:
                    continue

                results.append({
                    "a_measure_start": a0 + 1,
                    "a_measure_end": a0 + length,
                    "b_measure_start": b0 + 1,
                    "b_measure_end": b0 + length,
                    "a_time_start": float(measures[a0]["time_start"]),
                    "a_time_end": float(measures[a0 + length - 1]["time_end"]),
                    "b_time_start": float(measures[b0]["time_start"]),
                    "b_time_end": float(measures[b0 + length - 1]["time_end"]),
                    "length": length,
                    "similarity": round(score, 4),
                })

    results.sort(
        key=lambda item: (
            item["similarity"],
            item["length"],
        ),
        reverse=True,
    )

    selected = []
    for item in results:
        duplicate = False
        for old in selected:
            if (
                abs(item["a_measure_start"] - old["a_measure_start"]) <= 2
                and abs(item["b_measure_start"] - old["b_measure_start"]) <= 2
            ):
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(item)
        if len(selected) >= max_results:
            break

    return selected


def _normalize_lyric_text(value: str) -> str:
    value = str(value or "").lower()
    value = value.replace("’", "'")
    value = re.sub(r"[^a-zà-ÿ0-9'\s-]+", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def lyrics_phrases(
    words: list[dict[str, Any]],
    *,
    gap_seconds: float = 1.15,
    max_phrase_seconds: float = 14.0,
) -> list[dict[str, Any]]:
    """Group Whisper words into phrase-like timed chunks."""
    clean = []
    for item in words or []:
        text = str(item.get("text", "") or "").strip()
        start = float(item.get("start", 0.0) or 0.0)
        end = float(item.get("end", start) or start)
        if not text or end <= start:
            continue
        clean.append({"text": text, "start": start, "end": end})

    phrases = []
    current = []

    def close():
        nonlocal current
        if not current:
            return
        text = " ".join(x["text"] for x in current).strip()
        normalized = _normalize_lyric_text(text)
        if normalized:
            phrases.append({
                "index": len(phrases),
                "time_start": float(current[0]["start"]),
                "time_end": float(current[-1]["end"]),
                "text": text,
                "normalized": normalized,
                "word_count": len(current),
            })
        current = []

    for item in clean:
        if current:
            gap = float(item["start"]) - float(current[-1]["end"])
            duration = float(item["end"]) - float(current[0]["start"])
            if gap >= gap_seconds or duration > max_phrase_seconds:
                close()
        current.append(item)
    close()
    return phrases


def _text_similarity(a: str, b: str) -> float:
    a = _normalize_lyric_text(a)
    b = _normalize_lyric_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def detect_lyric_repeats(
    phrases: list[dict[str, Any]],
    *,
    threshold: float = 0.72,
) -> list[dict[str, Any]]:
    """Find repeated lyric phrases, useful as chorus evidence."""
    matches = []
    for i, a in enumerate(phrases):
        if int(a.get("word_count", 0)) < 4:
            continue
        for j in range(i + 1, len(phrases)):
            b = phrases[j]
            if int(b.get("word_count", 0)) < 4:
                continue
            score = _text_similarity(a["normalized"], b["normalized"])
            if score < threshold:
                continue
            matches.append({
                "a_index": i,
                "b_index": j,
                "a_time_start": float(a["time_start"]),
                "a_time_end": float(a["time_end"]),
                "b_time_start": float(b["time_start"]),
                "b_time_end": float(b["time_end"]),
                "text_a": a["text"],
                "text_b": b["text"],
                "similarity": round(score, 4),
            })
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(0.001, min(a1 - a0, b1 - b0))
    return float(np.clip(overlap / denom, 0.0, 1.0))


def combine_structure_candidates(
    harmonic_motifs: list[dict[str, Any]],
    lyric_repeats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fuse harmony-first structure evidence with repeated lyrics.

    Harmony remains primary. Lyrics strengthen a harmonic recurrence; they do
    not create a block on their own here.
    """
    result = []
    for motif in harmonic_motifs:
        best_lyrics = None
        best_lyric_score = 0.0

        for lyr in lyric_repeats:
            oa = _interval_overlap(
                motif["a_time_start"], motif["a_time_end"],
                lyr["a_time_start"], lyr["a_time_end"],
            )
            ob = _interval_overlap(
                motif["b_time_start"], motif["b_time_end"],
                lyr["b_time_start"], lyr["b_time_end"],
            )
            aligned = min(oa, ob)
            if aligned <= 0.15:
                continue

            score = float(lyr["similarity"]) * aligned
            if score > best_lyric_score:
                best_lyric_score = score
                best_lyrics = lyr

        harmonic_score = float(motif["similarity"])
        lyric_score = float(best_lyrics["similarity"]) if best_lyrics else 0.0

        # Explicitly harmony-first.
        combined = 0.72 * harmonic_score + 0.28 * lyric_score

        structural_type = "progression répétée"
        if best_lyrics and lyric_score >= 0.78:
            structural_type = "refrain probable"

        row = dict(motif)
        row.update({
            "harmonic_score": round(harmonic_score, 4),
            "lyric_score": round(lyric_score, 4),
            "combined_score": round(combined, 4),
            "type": structural_type,
            "lyric_match": best_lyrics,
        })
        result.append(row)

    result.sort(
        key=lambda x: (x["combined_score"], x["length"]),
        reverse=True,
    )
    return result


def _measure_interval_text(
    words: list[dict[str, Any]],
    t0: float,
    t1: float,
) -> str:
    selected = []
    for item in words or []:
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        start = float(item.get("start", 0.0) or 0.0)
        end = float(item.get("end", start) or start)
        if end >= t0 and start <= t1:
            selected.append(text)
    text = " ".join(selected)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    return re.sub(r"\s+", " ", text).strip()


def _section_harmonic_similarity(
    measures: list[dict[str, Any]],
    a0: int,
    a1: int,
    b0: int,
    b1: int,
) -> float:
    pa = [str(m.get("pattern", "") or "") for m in measures[a0:a1]]
    pb = [str(m.get("pattern", "") or "") for m in measures[b0:b1]]
    n = min(len(pa), len(pb))
    if n < 2:
        return 0.0
    scores = [_measure_similarity(pa[i], pb[i]) for i in range(n)]
    base = float(np.mean(scores))
    length_penalty = n / max(len(pa), len(pb))
    return base * length_penalty


def build_visual_blocks(
    measures: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    words: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Create a readable non-destructive block proposal.

    This is only a visual/editorial preview:
    - no audio or word timestamp is moved;
    - boundaries come from strong repeated harmonic phrases;
    - nearby competing boundaries are compacted;
    - similar sections share the same A/B/C family.
    """
    if not measures:
        return []

    n = len(measures)
    strong = [
        c for c in (candidates or [])
        if float(c.get("combined_score", c.get("similarity", 0.0)) or 0.0) >= 0.64
        and int(c.get("length", 0) or 0) >= 6
    ][:8]

    boundary_scores: dict[int, float] = {0: 1.0, n: 1.0}

    for c in strong:
        score = float(c.get("combined_score", c.get("similarity", 0.0)) or 0.0)
        for key in ("a_measure_start", "b_measure_start"):
            pos = max(0, min(n, int(c.get(key, 1)) - 1))
            boundary_scores[pos] = max(boundary_scores.get(pos, 0.0), score)
        for key in ("a_measure_end", "b_measure_end"):
            pos = max(0, min(n, int(c.get(key, 0))))
            boundary_scores[pos] = max(boundary_scores.get(pos, 0.0), score)

    # Compact nearby boundaries so the preview does not fragment into mini-blocks.
    items = sorted(boundary_scores.items())
    compact: list[tuple[int, float]] = []
    min_gap = 5

    for pos, score in items:
        if not compact:
            compact.append((pos, score))
            continue
        ppos, pscore = compact[-1]
        if pos - ppos < min_gap and pos not in (0, n):
            if score > pscore and ppos not in (0, n):
                compact[-1] = (pos, score)
        else:
            compact.append((pos, score))

    boundaries = sorted({p for p, _ in compact} | {0, n})

    # Avoid tiny head/tail.
    if len(boundaries) >= 3 and boundaries[1] < 4:
        boundaries.pop(1)
    if len(boundaries) >= 3 and n - boundaries[-2] < 4:
        boundaries.pop(-2)

    sections = []
    representatives: list[tuple[int, int]] = []

    for i in range(len(boundaries) - 1):
        a0 = boundaries[i]
        a1 = boundaries[i + 1]
        if a1 <= a0:
            continue

        best_idx = None
        best_score = 0.0
        for idx, (r0, r1) in enumerate(representatives):
            score = _section_harmonic_similarity(measures, a0, a1, r0, r1)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None and best_score >= 0.72:
            cluster_idx = best_idx
        else:
            cluster_idx = len(representatives)
            representatives.append((a0, a1))

        label = chr(ord("A") + (cluster_idx % 26))
        first = measures[a0]
        last = measures[a1 - 1]
        t0 = float(first.get("time_start", 0.0))
        t1 = float(last.get("time_end", t0))

        chord_patterns = [
            str(m.get("pattern", "") or "")
            for m in measures[a0:a1]
        ]

        sections.append({
            "index": len(sections),
            "name": f"Bloc {label}",
            "cluster": label,
            "measure_start": int(first.get("measure", a0 + 1)),
            "measure_end": int(last.get("measure", a1)),
            "time_start": t0,
            "time_end": t1,
            "harmonic_repeat": round(float(best_score), 4),
            "lyrics": _measure_interval_text(words or [], t0, t1),
            "chord_patterns": chord_patterns,
            "visual_only": True,
        })

    return sections
