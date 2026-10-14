from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .ffmpeg_utils import FFMPEG, run


def _np():
    try:
        import numpy as np
        return np
    except ImportError as e:
        raise RuntimeError("numpy is required for audio analysis: pip install numpy") from e


@dataclass
class AudioAnalysis:
    hop: float
    db: "object"  # numpy array of per-frame RMS in dB


def decode_audio(path, sr: int = 16000):
    np = _np()
    p = run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "pipe:1"], "audio-decode")
    return np.frombuffer(p.stdout, dtype="<i2")


def frame_db(x, sr: int, hop_s: float = 0.02) -> AudioAnalysis:
    np = _np()
    hop = int(sr * hop_s)
    n = len(x) // hop
    db = np.empty(n, dtype=np.float32)
    scale = 32768.0 if x.dtype.kind == "i" else 1.0
    for i in range(0, n, 4096):
        j = min(n, i + 4096)
        blk = x[i * hop:j * hop].astype(np.float32).reshape(-1, hop) / scale
        rms = np.sqrt((blk ** 2).mean(axis=1))
        db[i:j] = 20 * np.log10(rms + 1e-9)
    return AudioAnalysis(hop_s, db)


def analyze_samples(x, sr: int) -> AudioAnalysis:
    return frame_db(x, sr)


def analyze_video_audio(path, sr: int = 16000) -> AudioAnalysis:
    return frame_db(decode_audio(path, sr), sr)


def modulation_scores(a: AudioAnalysis, win_s: float = 1.5, step_s: float = 0.5
                      ) -> List[Tuple[float, float, float]]:
    """Heuristic: energy-envelope modulation in the 3-7 Hz band (rhythmic bursts,
    as in laughter). Syllabic speech also modulates here, so use laughter_scores()
    which discounts windows covered by transcribed speech. Needs tuning per source."""
    np = _np()
    w = int(round(win_s / a.hop))
    s = int(round(step_s / a.hop))
    db = np.asarray(a.db)
    if len(db) < w:
        return []
    env = 10 ** (db / 20)
    freqs = np.fft.rfftfreq(w, a.hop)
    band = (freqs >= 3) & (freqs <= 7)
    total = (freqs >= 0.5) & (freqs <= 20)
    window = np.hanning(w)
    gate = np.percentile(db, 30)
    out = []
    for i in range(0, len(env) - w + 1, s):
        seg = env[i:i + w]
        if db[i:i + w].mean() < gate:
            score = 0.0
        else:
            spec = np.abs(np.fft.rfft((seg - seg.mean()) * window)) ** 2
            score = float(spec[band].sum() / (spec[total].sum() + 1e-12))
            score *= float(min(1.0, (seg.std() / (seg.mean() + 1e-9)) * 2))
        out.append((i * a.hop, i * a.hop + win_s, score))
    return out


def laughter_scores(a: AudioAnalysis, speech: Sequence[Tuple[float, float]],
                    win_s: float = 1.5, step_s: float = 0.5) -> List[Tuple[float, float, float]]:
    out = []
    for t0, t1, sc in modulation_scores(a, win_s, step_s):
        cov = sum(max(0.0, min(t1, b) - max(t0, a_)) for a_, b in speech) / (t1 - t0)
        out.append((t0, t1, sc * (1.0 - min(1.0, cov))))
    return out


def candidate_signals(a: AudioAnalysis, laugh: Sequence[Tuple[float, float, float]],
                      start: float, end: float) -> Dict[str, float]:
    """Delivery = loudness level + variation relative to the whole recording
    (an observable audio characteristic, not an emotion claim)."""
    np = _np()
    db = np.asarray(a.db)
    i0, i1 = int(start / a.hop), max(int(end / a.hop), int(start / a.hop) + 1)
    seg = db[i0:i1]
    if len(seg) == 0:
        return {"laughter": 0.0, "delivery": 0.0}
    p10, p90 = np.percentile(db, 10), np.percentile(db, 90)
    level = float(np.clip((seg.mean() - p10) / (p90 - p10 + 1e-9), 0, 1))
    var = float(np.clip(seg.std() / (db.std() + 1e-9), 0, 1))
    peak = max((s for t0, t1, s in laugh if t1 > start and t0 < end), default=0.0)
    return {"laughter": float(peak), "delivery": 0.6 * level + 0.4 * var}