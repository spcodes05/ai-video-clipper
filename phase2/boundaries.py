from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import ClipCfg

_STRIP = "\"')]”’"
_END = (".", "?", "!", "…")


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass
class Candidate:
    start: float
    end: float
    score: float = 0.0
    reasons: Dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


def is_sentence_end(w: Word) -> bool:
    return w.text.strip().rstrip(_STRIP).endswith(_END)


def _is_start(words: Sequence[Word], i: int, gap: float) -> bool:
    if i == 0:
        return True
    return is_sentence_end(words[i - 1]) or (words[i].start - words[i - 1].end) > gap


def refine_boundaries(words: Sequence[Word], start: float, end: float,
                      clip: ClipCfg) -> Optional[Candidate]:
    """Snap a rough [start, end] to sentence boundaries within duration limits.
    Returns None when no natural boundary pair fits (never pads)."""
    if not words:
        return None
    starts = [w.start for i, w in enumerate(words) if _is_start(words, i, clip.sentence_gap)]
    ends = [w.end for w in words if is_sentence_end(w)]

    back = [s for s in starts if start - clip.start_lookback <= s <= start]
    fwd = [s for s in starts if start < s <= start + 3.0]
    if back:
        new_start = max(back)
    elif fwd:
        new_start = min(fwd)
    else:
        return None

    lo = new_start + clip.min_seconds
    hi = new_start + clip.max_seconds - clip.lead_in - clip.tail_out
    after = [e for e in ends if max(end, lo) <= e <= min(end + clip.end_lookahead, hi)]
    if after:
        new_end = min(after)
    else:
        before = [e for e in ends if lo <= e < end and e <= hi]
        if not before:
            return None
        new_end = max(before)

    prev_end = max((w.end for w in words if w.end <= new_start), default=0.0)
    next_start = min((w.start for w in words if w.start >= new_end),
                     default=new_end + clip.tail_out)
    s = max(new_start - clip.lead_in, (prev_end + new_start) / 2)
    e = min(new_end + clip.tail_out, (new_end + next_start) / 2)
    s = max(s, 0.0)
    if e - s < 1.0:
        return None
    return Candidate(s, e)


def overlap_ratio(a: Candidate, b: Candidate) -> float:
    inter = min(a.end, b.end) - max(a.start, b.start)
    if inter <= 0:
        return 0.0
    return inter / max(min(a.duration, b.duration), 1e-9)


def dedupe(cands: Sequence[Candidate], max_overlap: float) -> List[Candidate]:
    kept: List[Candidate] = []
    for c in sorted(cands, key=lambda c: c.score, reverse=True):
        if all(overlap_ratio(c, k) <= max_overlap for k in kept):
            kept.append(c)
    return kept


def within_duration(c: Candidate, clip: ClipCfg) -> bool:
    return clip.min_seconds - 1e-6 <= c.duration <= clip.max_seconds + 1e-6


def compute_keep_segments(words: Sequence[Word], duration: float, max_pause: float,
                          keep_pause: float) -> List[Tuple[float, float]]:
    """Clip-relative segments to keep; only gaps between words longer than
    max_pause are shortened (keep_pause is retained), so words are never cut."""
    ws = sorted(words, key=lambda w: w.start)
    if not ws:
        return [(0.0, duration)]
    cuts: List[Tuple[float, float]] = []
    if ws[0].start > max_pause:
        cuts.append((0.0, ws[0].start - keep_pause / 2))
    for a, b in zip(ws, ws[1:]):
        if b.start - a.end > max_pause:
            cuts.append((a.end + keep_pause / 2, b.start - keep_pause / 2))
    if duration - ws[-1].end > max_pause:
        cuts.append((ws[-1].end + keep_pause / 2, duration))
    keep: List[Tuple[float, float]] = []
    pos = 0.0
    for a, b in cuts:
        if a - pos > 0.05:
            keep.append((pos, a))
        pos = max(pos, b)
    if duration - pos > 0.05:
        keep.append((pos, duration))
    return keep or [(0.0, duration)]


def retime(t: float, keep: Sequence[Tuple[float, float]]) -> float:
    acc = 0.0
    for a, b in keep:
        if t < a:
            return acc
        if t <= b:
            return acc + (t - a)
        acc += b - a
    return acc


def retime_words(words: Sequence[Word], keep: Sequence[Tuple[float, float]]) -> List[Word]:
    return [Word(retime(w.start, keep), retime(w.end, keep), w.text) for w in words]