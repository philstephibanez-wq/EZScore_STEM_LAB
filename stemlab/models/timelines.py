"""Shared timeline contracts for Stem Lab and future EZScore integration.

All timestamps are expressed in seconds relative to the ORIGINAL uploaded audio.
No analysis module is allowed to move another module's timestamps.
"""

from __future__ import annotations

from typing import TypedDict


class TimedWord(TypedDict):
    start: float
    end: float
    text: str


class BeatChord(TypedDict):
    index: int
    time: float
    chord: str
    confidence: float


class MeasureHarmony(TypedDict):
    measure: int
    time_start: float
    time_end: float
    beat_chords: list[str]
    pattern: str


class VisualBlock(TypedDict, total=False):
    index: int
    name: str
    cluster: str
    measure_start: int
    measure_end: int
    time_start: float
    time_end: float
    harmonic_repeat: float
    lyrics: str
    chord_patterns: list[str]
    visual_only: bool
    includes_preroll: bool
    includes_postroll: bool
