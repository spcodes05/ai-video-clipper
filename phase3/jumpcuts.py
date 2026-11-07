"""Conservative transcript-driven jump cuts: filler words and function-word stutters."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence, Set, Tuple

from phase2.boundaries import Word

from .config import JumpCutCfg
from .timeline import Seg

FILLERS = {"um", "uh", "uhm", "umm", "uhh", "er", "erm", "ah", "ahh", "hmm", "mm", "mmm", "hm"}
# "no no no" / "ha ha" are comedy, so only stutter-prone function words are ever collapsed.
STUTTER_WORDS = {"the", "a", "an", "i", "and", "so", "but", "to", "of", "it", "that", "we",
                 "you", "is", "in", "this", "he", "she", "they", "my", "your", "if", "was"}
_NORM = re.compile(r"[^\w']+")


def norm(text: str) -> str:
    return _NORM.sub("", text.lower())


@dataclass
class JumpCutPlan:
    cuts: List[Seg] = field(default_factory=list)
    removed_idx: Set[int] = field(default_factory=set)
    reasons: List[Tuple[str, str, float]] = field(default_factory=list)   # (kind, word, time)


def plan_jump_cuts(words: Sequence[Word], duration: float, cfg: JumpCutCfg,
                   laughter: Sequence[Seg] = ()) -> JumpCutPlan:
    ws = list(words)
    plan = JumpCutPlan()
    if not cfg.enabled or len(ws) < 3:
        return plan
    cand: List[Tuple[int, str]] = []
    for i, w in enumerate(ws):
        n = norm(w.text)
        if cfg.remove_fillers and n in FILLERS and (w.end - w.start) <= cfg.max_filler_seconds:
            cand.append((i, "filler"))
        elif (cfg.remove_repeats and i + 1 < len(ws) and n in STUTTER_WORDS
              and norm(ws[i + 1].text) == n and ws[i + 1].start - w.end <= 0.3
              and (w.end - w.start) <= 0.45 and not w.text.strip().endswith((".", "?", "!"))):
            cand.append((i, "repeat"))
    budget = cfg.max_cut_ratio * duration
    total = 0.0
    last_end = -1e9
    for i, kind in cand:
        w = ws[i]
        if any(a - 1.0 < w.end and b + 0.5 > w.start for a, b in laughter):
            continue                                   # near a laugh: it may be part of the joke
        if kind == "repeat" and (ws[i + 1].text.strip().endswith(("!", "?")) or
                                 (i + 2 < len(ws) and ws[i + 2].text.strip().endswith("!"))):
            continue                                   # emphatic repetition is delivery, not a stutter
        prev_end = ws[i - 1].end if i > 0 else 0.0
        next_start = ws[i + 1].start if i + 1 < len(ws) else duration
        a = max(w.start - 0.03, (prev_end + w.start) / 2)
        b = min(w.end + 0.03, (w.end + next_start) / 2)
        if kind == "repeat":                   # also take the gap up to the surviving word
            b = min(next_start, w.end + 0.3) if next_start - w.end <= 0.3 else b
        if b - a < 0.08 or a - last_end < cfg.min_segment or total + (b - a) > budget:
            continue
        if i > 0 and ws[i - 1].start > a - cfg.min_segment and last_end < 0:
            pass
        if a < cfg.min_segment * 0.5:          # would leave a sliver at the very start
            continue
        plan.cuts.append((a, b))
        plan.removed_idx.add(i)
        plan.reasons.append((kind, w.text, round(w.start, 2)))
        last_end = b
        total += b - a
    return plan
