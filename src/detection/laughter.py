"""Audio laughter / reaction windows.

Heuristic: rhythmic 3-7 Hz energy bursts in stretches NOT covered by transcribed speech
(see phase2.audio_analysis). It is a weak signal: scores are normalised to the 95th
percentile of the whole video, and the raw windows are dumped to analysis/laughter.json
so they can be checked by ear.
"""
import logging
import wave
from pathlib import Path

import numpy as np

from src.models.schemas import Transcript

log = logging.getLogger(__name__)

Window = tuple[float, float, float]  # (start, end, score 0..1)


def laughter_windows(wav_path: Path, transcript: Transcript) -> list[Window]:
    try:
        from phase2.audio_analysis import analyze_samples, laughter_scores
    except ImportError as exc:
        log.warning("Laughter detection unavailable (%s)", exc)
        return []
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        x = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    speech = [(s.start, s.end) for s in transcript.segments]
    raw = laughter_scores(analyze_samples(x, sr), speech)
    if not raw:
        return []
    ref = float(np.percentile([sc for _, _, sc in raw], 95))
    if ref <= 1e-9:
        return []
    return [(t0, t1, min(1.0, sc / ref)) for t0, t1, sc in raw]


def peak_in(windows: list[Window], start: float, end: float) -> float:
    return max((sc for t0, t1, sc in windows if t1 > start and t0 < end), default=0.0)
