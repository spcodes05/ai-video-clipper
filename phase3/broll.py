"""Optional B-roll: planning + real compositing for a USER-SUPPLIED folder of clips.

How matching works (and its limit): a clip is used only when a spoken word equals a word in the
file name (`coffee.mp4` for "coffee"). That is exact keyword matching, not semantic understanding,
so it never inserts anything you did not name yourself. Stock-footage APIs are not included
(cost + relevance cannot be guaranteed offline).

Safety rules: never in the first 1.5 s / last 1 s, never during a reaction/laugh window, at most
`max_inserts`, `max_seconds` each, `min_gap` apart, and only inside a single kept segment. B-roll
covers the foreground panel only; the original audio keeps playing untouched (sync preserved)."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from phase2.boundaries import Word

from .config import BrollCfg

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm")
STOPWORDS = {"about", "after", "again", "because", "before", "could", "should", "their", "there", "these",
             "thing", "things", "those", "which", "while", "would", "really", "going", "think"}


@dataclass(frozen=True)
class BrollInsert:
    start: float          # clip-relative seconds
    end: float
    path: str
    keyword: str


class BrollProvider(ABC):
    @abstractmethod
    def find(self, keyword: str) -> Optional[str]:
        """Return a local video path for the keyword, or None."""


class NullBrollProvider(BrollProvider):
    def find(self, keyword: str) -> Optional[str]:
        return None


class FolderBrollProvider(BrollProvider):
    def __init__(self, folder: str):
        self.files = [p for p in Path(folder).glob("*") if p.suffix.lower() in VIDEO_EXT]

    def find(self, keyword: str) -> Optional[str]:
        k = keyword.lower()
        for p in sorted(self.files):
            if k in re.sub(r"[^a-z0-9]+", " ", p.stem.lower()).split():
                return str(p)
        return None


def get_provider(cfg: BrollCfg) -> BrollProvider:
    if cfg.enabled and cfg.provider == "folder" and cfg.folder and Path(cfg.folder).is_dir():
        return FolderBrollProvider(cfg.folder)
    return NullBrollProvider()


def plan_broll(words: Sequence[Word], provider: BrollProvider, cfg: BrollCfg, duration: float,
               blocked: Sequence[tuple] = ()) -> List[BrollInsert]:
    if not cfg.enabled or cfg.max_inserts <= 0:
        return []
    out: List[BrollInsert] = []
    for w in words:
        if len(out) >= cfg.max_inserts:
            break
        key = re.sub(r"[^a-z0-9]", "", w.text.lower())
        if len(key) < cfg.min_word_chars or key in STOPWORDS:
            continue
        a = max(0.0, w.start - 0.1)
        b = a + min(cfg.max_seconds, max(1.0, w.end - w.start + 1.2))
        if a < 1.5 or b > duration - 1.0:
            continue
        if any(a < y and b > x for x, y in blocked):
            continue
        if out and a < out[-1].end + cfg.min_gap:
            continue
        p = provider.find(key)
        if p:
            out.append(BrollInsert(round(a, 3), round(b, 3), p, key))
    return out
