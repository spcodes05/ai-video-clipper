"""Optional B-roll. Only the interface and an offline folder matcher are provided.

Why not more: reliable B-roll needs (a) a relevance model so unrelated stock footage is never
inserted and (b) a guarantee that the speaker's expression is not covered. Neither can be
guaranteed with the local-only tooling here, so inserts are PLANNED and written to the clip
report but NOT burned into the video. The renderer works identically with B-roll off."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from phase2.boundaries import Word

from .config import BrollCfg


@dataclass(frozen=True)
class BrollInsert:
    start: float
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
    """Matches a keyword against file names in a user-supplied folder (no network, no cost)."""

    def __init__(self, folder: str):
        self.files = [p for p in Path(folder).glob("*") if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm")]

    def find(self, keyword: str) -> Optional[str]:
        k = keyword.lower()
        for p in self.files:
            if k in re.sub(r"[^a-z0-9]+", " ", p.stem.lower()).split():
                return str(p)
        return None


def get_provider(cfg: BrollCfg) -> BrollProvider:
    if cfg.enabled and cfg.provider == "folder" and cfg.folder and Path(cfg.folder).is_dir():
        return FolderBrollProvider(cfg.folder)
    return NullBrollProvider()


def plan_broll(words: Sequence[Word], provider: BrollProvider, cfg: BrollCfg,
               blocked: Sequence[tuple] = ()) -> List[BrollInsert]:
    if not cfg.enabled:
        return []
    out: List[BrollInsert] = []
    for w in words:
        if len(out) >= cfg.max_inserts:
            break
        key = re.sub(r"[^a-z0-9]", "", w.text.lower())
        if len(key) < 4 or any(a < w.end and b > w.start for a, b in blocked):
            continue
        p = provider.find(key)
        if p:
            out.append(BrollInsert(w.start, min(w.start + 2.0, w.end + 1.5), p, key))
    return out
