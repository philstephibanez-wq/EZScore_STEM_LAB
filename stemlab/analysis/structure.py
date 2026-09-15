"""Multimodal structure fusion.

Important invariant: this module proposes visual/editorial blocks only.
It never changes chord, lyric or audio timestamps.
"""

from .structure_engine import build_visual_blocks, combine_structure_candidates

__all__ = ["combine_structure_candidates", "build_visual_blocks"]
