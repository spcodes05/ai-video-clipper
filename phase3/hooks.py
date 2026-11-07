"""Opening hooks.

A hook is shown only when it genuinely adds something:
  1. LLM suggestion (optional advisor), strictly validated against the transcript, else
  2. verified rule-based curiosity: a question asked LATER in the clip, or "wait for the
     reaction" when a laugh was actually measured.
No candidate -> no hook. Nothing is invented and a hook never repeats words the viewer is
about to read in the captions."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from phase2.boundaries import Word

from .config import HookCfg
from .timeline import Seg
from .zoom import split_sentences

SMALL = {"a", "an", "the", "of", "in", "on", "to", "and", "but", "or", "for", "at", "by", "with", "is"}
BANNED = ("you won't believe", "you wont believe", "shocking", "must see", "must watch", "blow your mind",
          "went viral", "gone viral", "number one", "100%", "gone wrong", "will change your life")
_STRIP = re.compile(r"[\"“”‘’()\[\]{}*_]")
_TOK = re.compile(r"[^\w']+")


@dataclass(frozen=True)
class Hook:
    text: str
    source: str       # llm | quote | template:laughter | manual
    score: float = 0.0


def tokens(text: str) -> List[str]:
    return [t for t in (_TOK.sub("", w.lower()) for w in text.split()) if t]


def clean_text(s: str) -> str:
    s = _STRIP.sub("", s).replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(",;:- ")


def apply_case(text: str, mode: str) -> str:
    if mode == "upper":
        return text.upper()
    if mode == "sentence":
        t = text.lower()
        return t[:1].upper() + t[1:]
    if mode == "title":
        out = []
        for i, w in enumerate(text.split()):
            lw = w.lower()
            out.append(lw if (i and lw in SMALL) else lw[:1].upper() + lw[1:])
        return " ".join(out)
    return text


def spoken_in_window(words: Sequence[Word], seconds: float) -> Set[str]:
    return {t for w in words if w.start < seconds for t in tokens(w.text)}


def validate_hook_text(text: str, words: Sequence[Word], cfg: HookCfg) -> Optional[str]:
    """Return the cleaned text if it is acceptable, else None."""
    t = clean_text(text or "")
    if not t:
        return None
    toks = t.split()
    if not 2 <= len(toks) <= cfg.max_words:
        return None
    low = t.lower()
    if any(b in low for b in BANNED):
        return None
    letters = [c for c in t if c.isalpha()]
    if len(letters) >= 6 and sum(c.isupper() for c in letters) / len(letters) > 0.8 and len(toks) > 3:
        t = apply_case(t, "sentence")                 # shouting -> calm sentence case
    spoken = {x for w in words for x in tokens(w.text)}
    for i, w in enumerate(toks):                      # names / numbers must come from the transcript
        core = _TOK.sub("", w)
        if core and (any(c.isdigit() for c in core) or (i and core[0].isupper() and core.lower() not in SMALL
                                                         and core.lower() not in COMMON_CAPS)):
            if core.lower() not in spoken:
                return None
    # do not repeat a whole spoken sentence, nor what the captions show while the hook is up
    htok = set(tokens(t))
    for s in split_sentences(words):
        st = set(tokens(s.text))
        if st and len(htok & st) / max(len(htok | st), 1) >= 0.8:
            return None
    early = spoken_in_window(words, cfg.duration)
    if htok and len(htok & early) / len(htok) >= 0.7:
        return None
    return t


COMMON_CAPS = {"i", "i'm", "i'll", "i've", "i'd", "wait", "why", "what", "who", "how", "when", "where",
               "nobody", "everyone", "this", "that", "he", "she", "they", "it", "his", "her", "their", "until",
               "you", "hear", "answer", "conversation", "got", "was", "not", "ready", "for", "expected",
               "response", "reaction", "weird", "ever", "still", "never", "even", "then", "here", "now"}


def generate_hook(words: Sequence[Word], laughter: Sequence[Seg], cfg: HookCfg,
                  laughter_scores: Sequence[float] = (), advisor_hook: Optional[str] = None
                  ) -> Optional[Hook]:
    if not cfg.enabled:
        return None
    if cfg.text:
        return Hook(apply_case(clean_text(cfg.text), "auto" if cfg.text_case == "auto" else cfg.text_case),
                    "manual", 99.0)
    if not words:
        return None
    if advisor_hook:
        ok = validate_hook_text(advisor_hook, words, cfg)
        if ok:
            return Hook(ok, "llm", 3.0)

    cands: List[Hook] = []
    strong_laugh = bool(laughter) and (not laughter_scores or max(laughter_scores) >= 0.5)
    if cfg.mode in ("auto", "template") and strong_laugh:
        cands.append(Hook("Wait For The Reaction", "template:laughter", 1.6))
    if cfg.mode in ("auto", "quote"):
        t0 = words[0].start
        early = spoken_in_window(words, cfg.duration)
        for s in split_sentences(words):
            toks = [clean_text(w.text) for w in s.words if clean_text(w.text)]
            if not s.words[-1].text.strip().endswith("?") or not 3 <= len(toks) <= cfg.max_words:
                continue
            if s.start - t0 < cfg.duration + 0.5:        # asked right away: captions already show it
                continue
            txt = " ".join(toks).rstrip(".,")
            if len({t for x in toks for t in tokens(x)} & early) / max(len(toks), 1) >= 0.5:
                continue
            cands.append(Hook(txt, "quote", 1.2 + (0.2 if s.start - t0 < 0.5 * (words[-1].end - t0) else 0.0)))
    cands = [c for c in cands if c.score >= cfg.min_score]
    if not cands:
        return None
    best = max(cands, key=lambda h: h.score)
    return Hook(apply_case(best.text, "auto" if cfg.text_case == "auto" else cfg.text_case), best.source, best.score)
