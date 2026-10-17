from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import FramingCfg
from .faces import FaceBox


@dataclass
class Subject:
    t: float
    cx: Optional[float]
    cy: Optional[float]
    track_id: Optional[int]


class _Track:
    def __init__(self, tid: int, cx: float, cy: float):
        self.id = tid
        self.cx = cx
        self.cy = cy
        self.last_t = 0.0
        self.area = 1.0
        self.mouth = 0.0


def _speaking(t: float, speech: Sequence[Tuple[float, float]]) -> bool:
    return any(a <= t <= b for a, b in speech)


def track_subject(samples: Sequence[Tuple[float, Sequence[FaceBox]]], src_w: float,
                  cfg: FramingCfg, speech: Sequence[Tuple[float, float]] = (),
                  cuts: Sequence[float] = ()) -> List[Subject]:
    """Associate faces across samples and pick one subject per sample.
    Subject choice uses face size plus a mouth-motion proxy while transcript
    speech is active; switching needs a margin and a hold time (hysteresis).
    cx is None when no face is visible."""
    tracks: List[_Track] = []
    nid = 0
    current: Optional[_Track] = None
    hold_until = -1.0
    prev_t: Optional[float] = None
    out: List[Subject] = []

    for t, boxes in samples:
        if prev_t is not None and any(prev_t < c <= t for c in cuts):
            tracks.clear()
            current = None
            hold_until = -1.0
        prev_t = t
        seen: List[_Track] = []
        used = set()
        for b in sorted(boxes, key=lambda b: -(b.w * b.h)):
            bx, by = b.x + b.w / 2, b.y + b.h / 2
            best, bd = None, float("inf")
            for tr in tracks:
                if tr.id in used or t - tr.last_t > cfg.track_ttl:
                    continue
                d = math.hypot(bx - tr.cx, by - tr.cy)
                if d < bd:
                    best, bd = tr, d
            if best is not None and bd <= cfg.match_distance * src_w:
                tr = best
            else:
                tr = _Track(nid, bx, by)
                nid += 1
                tracks.append(tr)
            used.add(tr.id)
            tr.cx, tr.cy, tr.last_t = bx, by, t
            tr.area = b.w * b.h
            tr.mouth = 0.6 * tr.mouth + 0.4 * b.mouth
            seen.append(tr)
        tracks[:] = [tr for tr in tracks if t - tr.last_t <= cfg.track_ttl]

        if not seen:
            out.append(Subject(t, None, None, None))
            continue
        speaking = _speaking(t, speech)
        max_area = max(tr.area for tr in seen) or 1.0

        def score(tr: _Track) -> float:
            return tr.area / max_area + tr.mouth * (1.0 if speaking else 0.3)

        best_tr = max(seen, key=score)
        if current is not None and current in seen and best_tr is not current:
            if t < hold_until or score(best_tr) < score(current) * cfg.switch_ratio:
                best_tr = current
        if best_tr is not current:
            current = best_tr
            hold_until = t + cfg.hold_seconds
        out.append(Subject(t, current.cx, current.cy, current.id))
    return out
