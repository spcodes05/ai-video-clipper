"""Clip-time <-> output-time mapping once segments have been cut out."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Seg = Tuple[float, float]


def keep_duration(keep: Optional[Sequence[Seg]], total: float) -> float:
    return sum(b - a for a, b in keep) if keep else total


def retime(t: float, keep: Optional[Sequence[Seg]]) -> float:
    if not keep:
        return t
    acc = 0.0
    for a, b in keep:
        if t < a:
            return acc
        if t <= b:
            return acc + (t - a)
        acc += b - a
    return acc


def map_interval(a: float, b: float, keep: Optional[Sequence[Seg]]) -> Optional[Seg]:
    """Map [a,b] to output time; if cuts split it, keep the longest surviving piece."""
    if not keep:
        return (a, b)
    best: Optional[Seg] = None
    acc = 0.0
    for s, e in keep:
        lo, hi = max(a, s), min(b, e)
        if hi - lo > 1e-6:
            piece = (acc + lo - s, acc + hi - s)
            if best is None or piece[1] - piece[0] > best[1] - best[0]:
                best = piece
        acc += e - s
    return best


def complement(cuts: Sequence[Seg], duration: float, min_len: float = 0.05) -> List[Seg]:
    """Merge cut spans and return the segments to keep."""
    merged: List[List[float]] = []
    for a, b in sorted((max(0.0, a), min(duration, b)) for a, b in cuts if b > a):
        if merged and a <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    keep: List[Seg] = []
    pos = 0.0
    for a, b in merged:
        if a - pos > min_len:
            keep.append((pos, a))
        pos = max(pos, b)
    if duration - pos > min_len:
        keep.append((pos, duration))
    return keep or [(0.0, duration)]


def is_identity(keep: Optional[Sequence[Seg]], duration: float) -> bool:
    return not keep or (len(keep) == 1 and keep[0][0] <= 0.01 and abs(keep[0][1] - duration) < 0.01)


def retime_keyframes(kf: Sequence[Tuple[float, float]], keep: Optional[Sequence[Seg]]
                     ) -> List[Tuple[float, float]]:
    """Move (t, x) crop keyframes onto the cut timeline. Each kept segment gets its own
    start/end samples (linear interpolation), so the path is continuous inside a segment
    and steps (1 ms) at the joins, exactly like the video itself."""
    if not keep:
        return list(kf)

    def at(t: float) -> float:
        if t <= kf[0][0]:
            return kf[0][1]
        for (t0, x0), (t1, x1) in zip(kf, kf[1:]):
            if t < t1:
                return x0 + (x1 - x0) * ((t - t0) / (t1 - t0) if t1 > t0 else 0.0)
        return kf[-1][1]

    out: List[Tuple[float, float]] = []
    acc = 0.0
    for a, b in keep:
        pts = [(a, at(a))] + [(t, x) for t, x in kf if a < t < b] + [(b, at(b))]
        for i, (t, x) in enumerate(pts):
            ot = acc + (t - a)
            if i == len(pts) - 1 and len(keep) > 1:
                ot -= 1e-3
            if out and ot <= out[-1][0]:
                ot = out[-1][0] + 1e-3
            out.append((ot, x))
        acc += b - a
    return out


def limit_cut_density(cuts: Sequence[Seg], duration: float, per_10s: float) -> List[Seg]:
    """Pacing guard: keep at most per_10s cuts per 10 seconds (the longest cuts win)."""
    allowed = max(1, int(per_10s * duration / 10.0 + 0.5))
    if len(cuts) <= allowed:
        return list(cuts)
    best = sorted(cuts, key=lambda c: c[0] - c[1])[:allowed]
    return sorted(best)


def snap_cuts_to_quiet(cuts: Sequence[Seg], db, ms: int, frame: float = 0.02,
                       limits: Optional[Sequence[Seg]] = None) -> List[Seg]:
    """Move each cut edge to the quietest 20 ms frame within +-ms, never leaving its allowed span
    (`limits` = the silent/gap region the cut lives in), so joins land in silence instead of on a
    waveform peak."""
    if db is None or len(db) == 0 or ms <= 0:
        return list(cuts)
    n = max(1, int(ms / (frame * 1000)))
    out: List[Seg] = []
    for a, b in cuts:
        def best(t: float, lo: float, hi: float) -> float:
            i = int(round(t / frame))
            cand = [(float(db[j]), abs(j - i), j) for j in range(max(0, i - n), min(len(db), i + n + 1))
                    if lo <= j * frame <= hi]
            return min(cand)[2] * frame if cand else t
        lim = next(((x, y) for x, y in (limits or []) if x - 1e-6 <= a and b <= y + 1e-6), (a - 0.1, b + 0.1))
        na, nb = best(a, max(lim[0], a - 0.06), min(a + 0.06, b - 0.1)), best(b, max(b - 0.06, a + 0.1), min(lim[1], b + 0.06))
        out.append((na, nb) if nb - na >= 0.08 else (a, b))
    return out
