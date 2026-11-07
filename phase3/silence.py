"""Audio-based silence detection + word-safe silence removal (keep-segment planning)."""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from phase2.boundaries import Word, is_sentence_end
from phase2.ffmpeg_utils import FFMPEG

from .config import SilenceCfg
from .timeline import Seg, complement, limit_cut_density, snap_cuts_to_quiet

log = logging.getLogger("phase3.silence")
SR = 16000
FRAME = 0.02


def read_audio(src, start: float, duration: float, sr: int = SR) -> Optional[np.ndarray]:
    """Mono float32 samples of [start, start+duration] via one FFmpeg pipe, or None."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src),
           "-t", f"{duration:.3f}", "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    try:
        p = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError:
        return None
    if p.returncode != 0 or not p.stdout:
        return None
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def frame_db(x: np.ndarray, sr: int = SR, frame: float = FRAME) -> np.ndarray:
    n = max(1, int(sr * frame))
    k = len(x) // n
    if k == 0:
        return np.array([-120.0])
    f = x[: k * n].reshape(k, n)
    rms = np.sqrt(np.mean(f * f, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-9)


def effective_threshold(db: np.ndarray, cfg: SilenceCfg) -> float:
    """Absolute threshold, raised above the noise floor when the recording is noisy."""
    thr = cfg.threshold_db
    if cfg.adaptive and len(db) >= 10:
        floor = float(np.percentile(db, 10))
        thr = max(thr, min(floor + 8.0, thr + 12.0))
    return thr


def detect_silences(db: np.ndarray, thr: float, min_dur: float, frame: float = FRAME) -> List[Seg]:
    """Runs of frames below `thr` lasting >= min_dur (clip seconds)."""
    out: List[Seg] = []
    run: Optional[int] = None
    for i, v in enumerate(db):
        if v < thr:
            run = i if run is None else run
        elif run is not None:
            if (i - run) * frame >= min_dur:
                out.append((run * frame, i * frame))
            run = None
    if run is not None and (len(db) - run) * frame >= min_dur:
        out.append((run * frame, len(db) * frame))
    return out


@dataclass
class SilencePlan:
    keep: List[Seg]
    cuts: List[Seg] = field(default_factory=list)
    removed: float = 0.0
    preserved_comedic: int = 0
    threshold_db: float = 0.0


def _pause_class(prev: Word, nxt: Word, nxt_sentence_words: int, laughter: Sequence[Seg],
                 cfg: SilenceCfg) -> str:
    """normal | comedic | dramatic.

    dramatic: a held beat right before a payoff (a laugh within 3 s, or a short exclamation/question) -
    never cut down. comedic: setup -> beat -> short line, or any pause touching a laugh."""
    gap = nxt.start - prev.end
    laugh_soon = any(nxt.start - 0.2 <= a <= nxt.start + 3.0 for a, _ in laughter)
    if gap >= cfg.dramatic_pause_min and laugh_soon:
        return "dramatic"
    if any(a < nxt.start + 0.5 and b > prev.end - 0.3 for a, b in laughter):
        return "comedic"
    ends = is_sentence_end(prev) or prev.text.strip().endswith(("...", "—", "-"))
    if ends and nxt_sentence_words <= 8:
        return "comedic"
    return "normal"


def _sentence_len_from(words: Sequence[Word], i: int) -> int:
    n = 0
    for w in words[i:]:
        n += 1
        if is_sentence_end(w):
            break
    return n


def validate_keep(words: Sequence[Word], keep: Sequence[Seg], tol: float = 0.01) -> List[Word]:
    """Words that are cut through (not fully inside a single kept segment)."""
    bad = []
    for w in words:
        if not any(a - tol <= w.start and w.end <= b + tol for a, b in keep):
            bad.append(w)
    return bad


def plan_silence(words: Sequence[Word], duration: float, silences: Optional[Sequence[Seg]],
                 cfg: SilenceCfg, laughter: Sequence[Seg] = (),
                 extra_cuts: Sequence[Seg] = (), thr: float = 0.0,
                 cut_ok: Sequence[Word] = (), db=None) -> SilencePlan:
    """Cut only the part of a between-word gap that the audio confirms is silent.

    * speech is padded by `padding` on both sides (words are never touched);
    * `keep_pause` of natural pause survives every cut (`comedic_keep` for comic timing);
    * laugh/reaction windows and non-silent noise inside a gap are never cut;
    * total removal is capped at `max_cut_ratio` (longest gaps are cut first);
    * the result is verified: no word may straddle a cut."""
    ws = sorted(words, key=lambda w: w.start)
    if not ws:
        return SilencePlan([(0.0, duration)], threshold_db=thr)

    gaps: List[Tuple[float, float, object]] = []         # (gap_start, gap_end, pause class)
    if ws[0].start > cfg.minimum_silence_duration:
        gaps.append((0.0, ws[0].start, "normal"))
    comedic_n = 0
    for i in range(len(ws) - 1):
        a, b = ws[i], ws[i + 1]
        if b.start - a.end >= cfg.minimum_silence_duration:
            cls = _pause_class(a, b, _sentence_len_from(ws, i + 1), laughter, cfg) \
                if cfg.preserve_comedic_pauses else "normal"
            gaps.append((a.end, b.start, cls))
    if duration - ws[-1].end > cfg.minimum_silence_duration:
        gaps.append((ws[-1].end, duration, "normal"))

    cuts: List[Tuple[float, Seg]] = []                    # (cut length, span)
    for g0, g1, cls in gaps:
        com = cls in ("comedic", "dramatic")
        keep_total = (cfg.dramatic_pause_keep if cls == "dramatic"
                      else cfg.comedic_keep if cls == "comedic" else cfg.keep_pause)
        keep_total = max(keep_total, 2 * cfg.padding)
        head = g0 == 0.0 and ws[0].start == g1
        tail = g1 == duration and ws[-1].end == g0
        lo = g0 if head else g0 + keep_total / 2
        hi = g1 if tail else g1 - keep_total / 2
        if head:
            hi = g1 - cfg.padding
        if tail:
            lo = g0 + cfg.padding
        if hi - lo < 0.12:
            continue
        if silences is not None:   # restrict to audio-confirmed silence, shrunk by the padding (None = no audio data)
            pieces = [(max(lo, s0 + cfg.padding / 2), min(hi, s1 - cfg.padding / 2))
                      for s0, s1 in silences if s1 > lo and s0 < hi]
        else:
            pieces = [(lo, hi)]
        for p0, p1 in pieces:
            if any(a < p1 and b > p0 for a, b in laughter):
                continue
            if p1 - p0 >= 0.12:
                cuts.append((p1 - p0, (p0, p1)))
        if com and any(abs(c[1][0] - lo) < 1e-6 or c[1][0] >= lo for c in cuts):
            comedic_n += 1

    budget = cfg.max_cut_ratio * duration
    chosen: List[Seg] = []
    total = 0.0
    for length, span in sorted(cuts, key=lambda c: -c[0]):
        room = budget - total
        if room < 0.12:
            continue
        if length > room:                       # shorten the cut to the remaining budget
            span = (span[0], span[0] + room)
            length = room
        chosen.append(span)
        total += length
    chosen = snap_cuts_to_quiet(chosen, db, cfg.snap_ms, limits=silences) if silences else chosen
    chosen += [c for c in extra_cuts]
    chosen = limit_cut_density(chosen, duration, cfg.max_cuts_per_10s)
    keep = complement(chosen, duration)

    if len(keep) > cfg.max_segments:                      # too choppy: drop the smallest cuts
        spans = sorted(chosen, key=lambda s: s[1] - s[0], reverse=True)[: cfg.max_segments - 1]
        keep = complement(spans, duration)
        chosen = spans
    # safety net: never cut through a word
    protected = [w for w in ws if w not in set(cut_ok)]
    for _ in range(len(chosen) + 1):
        bad = validate_keep(protected, keep)
        if not bad:
            break
        w = bad[0]
        chosen = [c for c in chosen if not (c[0] < w.end and c[1] > w.start)]
        keep = complement(chosen, duration)
    removed = duration - sum(b - a for a, b in keep)
    return SilencePlan(keep, sorted(chosen), removed, comedic_n, thr)
