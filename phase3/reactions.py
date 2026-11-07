"""Reaction shots: cut to the whole group while a laugh/reaction is *measured* in the audio
and several faces are *measured* in the picture. Otherwise nothing is added."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from phase2.faces import FaceBox

from .config import ReactionCfg
from .timeline import Seg

LaughWindow = Tuple[float, float, float]      # source seconds + score 0..1


def plan_reaction_windows(laughter: Sequence[LaughWindow], samples: Sequence[Tuple[float, Sequence[FaceBox]]],
                          clip_start: float, clip_end: float, cfg: ReactionCfg) -> List[Seg]:
    """Clip-relative windows. Requires >= min_faces in >= min_face_ratio of the samples inside."""
    if not cfg.enabled or not samples:
        return []
    out: List[Seg] = []
    for a, b, sc in laughter:
        if sc < cfg.min_laughter_score or b <= clip_start or a >= clip_end:
            continue
        s = max(a, clip_start) - clip_start
        e = min(b, clip_end) - clip_start
        e = min(e, s + cfg.max_window)
        if e - s < cfg.min_window:
            continue
        inside = [boxes for t, boxes in samples if s - 0.25 <= t <= e + 0.25]
        if not inside:
            continue
        multi = sum(1 for boxes in inside if len(boxes) >= cfg.min_faces)
        if multi / len(inside) >= cfg.min_face_ratio:
            out.append((round(s, 3), round(e, 3)))
    merged: List[Seg] = []
    for a, b in sorted(out):
        if merged and a <= merged[-1][1] + 0.3:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged
