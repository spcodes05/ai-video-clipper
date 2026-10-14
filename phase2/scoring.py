from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from .config import DEFAULT_WEIGHTS


def normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def opening_strength(text: str) -> float:
    """Weak heuristic for the first sentence; one factor among several."""
    first = re.split(r"(?<=[.?!])\s", text.strip(), maxsplit=1)[0]
    score = 0.0
    if first.endswith("?"):
        score += 0.4
    if re.search(r"\d", first):
        score += 0.2
    if 4 <= len(first.split()) <= 25:
        score += 0.3
    if re.match(r"^(and|but|so|because|um|uh)\b", first.lower()):
        score -= 0.3
    return max(0.0, min(1.0, score + 0.1))


def completeness(starts_at_sentence: bool, ends_at_sentence: bool) -> float:
    return 0.5 * float(starts_at_sentence) + 0.5 * float(ends_at_sentence)


def combine(factors: Sequence[Dict[str, float]],
            weights: Dict[str, float] = DEFAULT_WEIGHTS
            ) -> List[Tuple[float, Dict[str, float]]]:
    """Min-max normalises each factor across candidates, then returns
    (weighted score 0..1, per-factor contribution) for each candidate."""
    names = [n for n in weights if weights[n] > 0 and any(n in f for f in factors)]
    total_w = sum(weights[n] for n in names) or 1.0
    norm = {n: normalize([f.get(n, 0.0) for f in factors]) for n in names}
    out = []
    for i in range(len(factors)):
        parts = {n: weights[n] / total_w * norm[n][i] for n in names}
        out.append((sum(parts.values()), parts))
    return out