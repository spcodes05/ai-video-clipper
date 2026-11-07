"""Emphasis-driven zoom planning (no random zooms) + FFmpeg expression builder."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from phase2.boundaries import Word, is_sentence_end

from .config import ZoomCfg
from .timeline import Seg

INTENSIFIERS = {"never", "always", "insane", "crazy", "literally", "honestly", "wow", "seriously",
                "unbelievable", "worst", "best", "ridiculous", "shocked", "wild", "hilarious"}
_NORM = re.compile(r"[^\w']+")


@dataclass(frozen=True)
class ZoomEvent:
    start: float
    end: float
    scale: float
    reason: str = ""
    score: float = 0.0


@dataclass
class Sentence:
    words: List[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def split_sentences(words: Sequence[Word], gap: float = 0.9) -> List[Sentence]:
    out: List[Sentence] = []
    cur: List[Word] = []
    for w in sorted(words, key=lambda w: w.start):
        if cur and (w.start - cur[-1].end > gap):
            out.append(Sentence(cur))
            cur = []
        cur.append(w)
        if is_sentence_end(w):
            out.append(Sentence(cur))
            cur = []
    if cur:
        out.append(Sentence(cur))
    return out


def sentence_score(i: int, sents: Sequence[Sentence], energy_db: Optional[np.ndarray],
                   energy_ref: float, laughter: Sequence[Seg]) -> Tuple[float, str]:
    s = sents[i]
    score, why = 0.0, []
    txt = s.words[-1].text.strip()
    if txt.endswith("!"):
        score += 1.0
        why.append("exclamation")
    elif txt.endswith("?"):
        score += 0.3
    toks = {_NORM.sub("", w.text.lower()) for w in s.words}
    if toks & INTENSIFIERS:
        score += 0.4
        why.append("intensifier")
    if i > 0 and len(s.words) <= 9 and len(sents[i - 1].words) >= 8 and s.start - sents[i - 1].end >= 0.4:
        score += 1.0
        why.append("punchline")
    if any(s.end - 0.2 <= a <= s.end + 1.6 for a, _ in laughter):
        score += 1.3
        why.append("laugh-follows")
    if energy_db is not None and len(energy_db) and energy_ref > -90:
        a, b = int(s.start / 0.02), max(int(s.end / 0.02), int(s.start / 0.02) + 1)
        seg = energy_db[a:b]
        if len(seg) and float(np.percentile(seg, 90)) > energy_ref + 3.0:
            score += 0.8
            why.append("loud")
    return score, "+".join(why) or "none"


def plan_zoom(words: Sequence[Word], duration: float, cfg: ZoomCfg,
              energy_db: Optional[np.ndarray] = None, laughter: Sequence[Seg] = (),
              blocked: Sequence[Seg] = ()) -> List[ZoomEvent]:
    """Pick a few emphasis moments. `blocked` windows (e.g. group reaction shots) never zoom."""
    if not cfg.enabled or cfg.emphasis_scale <= cfg.normal_scale + 1e-6 or not words:
        return []
    sents = split_sentences(words)
    ref = float(np.percentile(energy_db, 75)) if energy_db is not None and len(energy_db) else -90.0
    scored: List[Tuple[float, int, str]] = []
    for i in range(len(sents)):
        sc, why = sentence_score(i, sents, energy_db, ref, laughter)
        if sc >= cfg.min_score:
            scored.append((sc, i, why))
    events: List[ZoomEvent] = []
    used = 0.0
    budget = cfg.max_fraction * duration
    for sc, i, why in sorted(scored, key=lambda t: (-t[0], t[1])):
        s = sents[i]
        # zoom on the emphasised tail of the sentence, not the whole thing
        n_tail = max(1, min(len(s.words), 6))
        a = max(0.0, s.words[-n_tail].start - 0.15)
        b = min(duration, max(s.end + 0.35, a + cfg.min_event_seconds))
        if b - a > cfg.max_event_seconds:
            a = b - cfg.max_event_seconds
        if b - a < cfg.min_event_seconds:
            continue
        if any(a < y and b > x for x, y in blocked):
            continue
        if any(a < e.end + cfg.min_spacing and b > e.start - cfg.min_spacing for e in events):
            continue
        if used + (b - a) > budget:
            continue
        scale = cfg.maximum_scale if (sc >= 2.3 and "laugh-follows" in why) else cfg.emphasis_scale
        events.append(ZoomEvent(round(a, 3), round(b, 3), min(scale, cfg.maximum_scale), why, round(sc, 2)))
        used += b - a
    return sorted(events, key=lambda e: e.start)


def zoom_at(events: Sequence[ZoomEvent], t: float, trans: float, normal: float = 1.0) -> float:
    """Python twin of zoom_expression (used by tests and face-safety checks)."""
    z = normal
    for e in events:
        a = min(max((t - e.start) / trans, 0.0), 1.0)
        b = min(max((e.end - t) / trans, 0.0), 1.0)
        m = min(a, b)
        z += (e.scale - normal) * m * m * (3 - 2 * m)
    return z


def zoom_expression(events: Sequence[ZoomEvent], trans: float, normal: float = 1.0) -> str:
    """Smoothstep-eased zoom factor as an FFmpeg expression of `t` (events must not overlap)."""
    if not events:
        return f"{normal:.4f}"
    terms = []
    for e in events:
        m = (f"min(clip((t-{e.start:.3f})/{trans:.3f},0,1),"
             f"clip(({e.end:.3f}-t)/{trans:.3f},0,1))")
        terms.append(f"{e.scale - normal:.4f}*({m})*({m})*(3-2*({m}))")
    return f"{normal:.4f}+" + "+".join(terms)


def max_safe_scale(boxes: Sequence[Tuple[float, float, float, float]], crop_x: float, crop_y: float,
                   cw: int, ch: int, margin: float = 0.06) -> float:
    """Largest zoom keeping every face (x, y, w, h in source px) that lies in the crop visible."""
    cx, cy = crop_x + cw / 2, crop_y + ch / 2
    worst = 0.0
    for x, y, w, h in boxes:
        fx, fy = x + w / 2, y + h / 2
        if not (crop_x <= fx <= crop_x + cw and crop_y <= fy <= crop_y + ch):
            continue
        ex = max(abs(x - cx), abs(x + w - cx)) / (cw / 2)
        ey = max(abs(y - cy), abs(y + h - cy)) / (ch / 2)
        worst = max(worst, ex, ey)
    if worst <= 0:
        return 99.0
    return (1.0 - margin) / worst
