"""Optional LLM advisor: one small JSON call per clip for (a) a curiosity hook and (b) the moments
that deserve a zoom. Cached on disk, never required, never trusted blindly (outputs are validated)."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from phase2.boundaries import Word

from .zoom import split_sentences

log = logging.getLogger("phase3.advisor")

SYSTEM = (
    "You are a short-form video editor's assistant. You receive the transcript of ONE clip with "
    "timestamps in seconds (relative to the clip start). Reply with JSON only:\n"
    '{"hook": string|null, "moments": [{"start": number, "end": number, "kind": '
    '"punchline|statement|reaction|emotion", "strength": 1|2|3}]}\n'
    "Hook rules: at most {max_words} words, grammatical, creates curiosity, describes what the clip "
    "is really about, does NOT reveal the punchline, does NOT quote a spoken sentence, does NOT "
    "invent people, names, numbers or events, no clickbait claims. Use null if no honest hook exists.\n"
    "Moments: at most 4 timestamps where a subtle zoom would help (a delivered punchline, an important "
    "statement, a reaction, an emotional beat). Do not mark ordinary sentences. Use only times that "
    "exist in the transcript."
)


@dataclass
class Moment:
    start: float
    end: float
    kind: str
    strength: int


@dataclass
class AdvisorResult:
    hook: Optional[str] = None
    moments: List[Moment] = field(default_factory=list)
    used: bool = False
    error: Optional[str] = None


def _transcript(words: Sequence[Word]) -> str:
    return "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in split_sentences(words))


def parse_reply(raw: str, duration: float, max_words: int) -> AdvisorResult:
    d = json.loads(raw)
    if not isinstance(d, dict):
        raise ValueError("reply is not an object")
    hook = d.get("hook")
    hook = hook.strip() if isinstance(hook, str) and hook.strip() else None
    moments: List[Moment] = []
    for m in (d.get("moments") or [])[:4]:
        try:
            a, b = float(m["start"]), float(m["end"])
            kind = str(m.get("kind", "statement"))
            st = int(m.get("strength", 1))
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= a < b <= duration + 0.5) or b - a > 6 or kind not in ("punchline", "statement", "reaction", "emotion"):
            continue
        moments.append(Moment(a, min(b, duration), kind, max(1, min(3, st))))
    return AdvisorResult(hook=hook, moments=moments, used=True)


class EditAdvisor:
    def __init__(self, provider, cache_dir: Optional[Path] = None, want_hook: bool = True,
                 want_moments: bool = True):
        self.provider = provider
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.want_hook, self.want_moments = want_hook, want_moments

    def advise(self, words: Sequence[Word], duration: float, max_words: int, title: str = "",
               has_laughter: bool = False) -> AdvisorResult:
        if not words:
            return AdvisorResult()
        user = (f"Clip length: {duration:.1f}s. Laughter detected: {'yes' if has_laughter else 'no'}.\n"
                f"Local title guess: {title}\nTranscript:\n{_transcript(words)}")
        system = SYSTEM.replace("{max_words}", str(max_words))
        key = hashlib.sha256((system + user).encode("utf-8")).hexdigest()[:16]
        cache = self.cache_dir / f"advisor_{key}.json" if self.cache_dir else None
        try:
            if cache and cache.exists():
                raw = cache.read_text(encoding="utf-8")
            else:
                raw = self.provider.complete_json(system, user)
                if cache:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(raw, encoding="utf-8")
            res = parse_reply(raw, duration, max_words)
        except Exception as exc:  # noqa: BLE001 - advisor is optional; never break a render
            log.warning("LLM advisor unavailable for this clip (%s); using local heuristics", exc)
            return AdvisorResult(error=str(exc))
        if not self.want_hook:
            res.hook = None
        if not self.want_moments:
            res.moments = []
        return res
