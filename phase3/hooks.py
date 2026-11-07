"""Rule-based opening hooks built ONLY from words actually spoken in the clip (or from
verified signals such as a detected laugh). Nothing is invented."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from phase2.boundaries import Word

from .config import HookCfg
from .timeline import Seg
from .zoom import split_sentences

HOOK_WORDS = {"never", "worst", "best", "crazy", "insane", "secret", "wrong", "actually", "honestly",
              "thought", "wait", "can't", "cannot", "unbelievable", "shocked", "terrible",
              "hilarious", "ridiculous", "nobody", "everyone", "mistake", "truth", "joking", "lied"}
_STRIP = re.compile(r"[\"“”‘’()\[\]{}]")


@dataclass(frozen=True)
class Hook:
    text: str
    source: str       # quote | template:laughter | manual
    score: float = 0.0


def _clean(s: str) -> str:
    s = _STRIP.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(",;:- ")


def _fit(words: List[str], max_words: int) -> str:
    cut = len(words) > max_words
    txt = " ".join(words[:max_words]).rstrip(",;:-. ")
    if cut:
        txt += "…"
    return txt


def generate_hook(words: Sequence[Word], laughter: Sequence[Seg], cfg: HookCfg,
                  laughter_scores: Sequence[float] = ()) -> Optional[Hook]:
    if not cfg.enabled:
        return None
    if cfg.text:
        return Hook(_clean(cfg.text).upper(), "manual", 99.0)
    if not words:
        return None
    sents = split_sentences(words)
    cands: List[Hook] = []
    horizon = words[0].start + max(25.0, 0.5 * (words[-1].end - words[0].start))
    for s in sents:
        if s.start > horizon:
            break
        toks = [_clean(w.text) for w in s.words if _clean(w.text)]
        if not 3 <= len(toks) <= 14:
            continue
        low = {t.lower().strip(".,!?…") for t in toks}
        last = s.words[-1].text.strip()
        score = 0.0
        score += 1.2 if last.endswith("?") else 0.0
        score += 1.0 if last.endswith("!") else 0.0
        score += 0.8 * len(low & HOOK_WORDS)
        score += 0.4 if any(c.isdigit() for c in " ".join(toks)) else 0.0
        score -= 0.05 * abs(len(toks) - 6)
        score -= 0.002 * (s.start - words[0].start)
        txt = _fit(toks, cfg.max_words)
        if last.endswith(("?", "!")) and not txt.endswith(("?", "!", "…")):
            txt += last[-1]
        cands.append(Hook(txt.upper(), "quote", score))
    best_quote = max(cands, key=lambda h: h.score) if cands else None
    strong_laugh = bool(laughter) and (not laughter_scores or max(laughter_scores) >= 0.5)
    if cfg.mode == "template" or (cfg.mode == "auto" and strong_laugh
                                  and (best_quote is None or best_quote.score < 0.8)):
        if strong_laugh:
            return Hook("WAIT FOR THE REACTION", "template:laughter", 1.0)
    if best_quote is not None:
        return best_quote
    toks = [_clean(w.text) for w in words[:cfg.max_words + 2] if _clean(w.text)]
    return Hook(_fit(toks, cfg.max_words).upper(), "quote", 0.0) if toks else None
