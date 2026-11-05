import wave
from pathlib import Path

import numpy as np

from src.models.schemas import AudioFeatures
from src.utils.errors import VideoInputError

FRAME_S = 0.1  # RMS resolution in seconds


class AudioAnalyzer:
    """
    Lightweight loudness/speech-activity features from per-frame RMS.

    NOTE: energy is NOT treated as proof of humor. It is just one feature.
    Real laughter/cheer detection is a Phase 2 replacement for this class.
    """

    def __init__(self, rms: np.ndarray):
        self.rms = np.asarray(rms, dtype=np.float32)
        if self.rms.size == 0:
            raise VideoInputError("Audio track is empty or could not be decoded.")
        self.ref = max(float(np.percentile(self.rms, 99)), 1e-6)
        floor = float(np.percentile(self.rms, 10))
        self.speech_threshold = max(floor * 2.0, self.ref * 0.08)
        self.is_speech = self.rms > self.speech_threshold

    @classmethod
    def from_wav(cls, wav_path: Path) -> "AudioAnalyzer":
        chunks: list[np.ndarray] = []
        with wave.open(str(wav_path), "rb") as wf:
            hop = int(wf.getframerate() * FRAME_S)
            while True:
                raw = wf.readframes(hop * 600)
                if not raw:
                    break
                x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                n = len(x) // hop
                if n == 0:
                    break
                chunks.append(np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1)))
        return cls(np.concatenate(chunks) if chunks else np.array([], dtype=np.float32))

    def features(self, start: float, end: float) -> AudioFeatures:
        a = max(0, int(start / FRAME_S))
        b = max(a + 1, int(end / FRAME_S))
        seg = self.rms[a:b]
        if seg.size == 0:
            return AudioFeatures()
        speech_ratio = float(self.is_speech[a:b].mean())
        return AudioFeatures(
            energy=float(min(seg.mean() / self.ref, 1.0)),
            peak_energy=float(min(seg.max() / self.ref, 1.0)),
            speech_ratio=speech_ratio,
            silence_ratio=1.0 - speech_ratio,
        )

    def sustained_until(self, start: float, limit: float, ratio: float, max_gap: float = 0.2) -> float:
        """End time of a loud run (laugh, applause, reaction) that begins right at `start`.

        Returns `start` when nothing loud follows. Never goes past `limit`. Brief dips up to
        `max_gap` seconds are bridged so a laugh with gaps counts as one reaction.
        """
        a = max(0, int(start / FRAME_S))
        b = max(a, int(limit / FRAME_S))
        thr = self.ref * ratio
        allowed_quiet = max(1, round(max_gap / FRAME_S))
        last, quiet = start, 0
        for i, v in enumerate(self.rms[a:b]):
            if v >= thr:
                last, quiet = (a + i + 1) * FRAME_S, 0
            else:
                quiet += 1
                if quiet > allowed_quiet:
                    break
        return min(last, limit)
