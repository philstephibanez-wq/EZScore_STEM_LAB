"""Rhythm-facing API.

Current validated R8 engine performs beat tracking inside analyze_harmony().
This module is the stable migration boundary for the next extraction step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .structure_engine import analyze_harmony as _analyze_harmony


def analyze_rhythm_and_harmony(
    *,
    other_path: Path,
    bass_path: Path,
    drums_path: Path,
    beats_per_bar: int = 4,
) -> dict[str, Any]:
    return _analyze_harmony(
        other_path=other_path,
        bass_path=bass_path,
        drums_path=drums_path,
        beats_per_bar=beats_per_bar,
    )
