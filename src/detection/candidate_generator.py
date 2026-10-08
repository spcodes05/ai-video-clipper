import logging
import re

import numpy as np

from src.detection.audio_analyzer import AudioAnalyzer
from src.models.schemas import CandidateClip, Scene, Transcript, TranscriptSegment
from src.pipeline.config import CandidateConfig
from src.utils.timeutils import overlap_ratio

log = logging.getLogger(__name__)

WORD_RE = re.compile(r"[A-Za-z']+")

EMOTION_WORDS = frozenset(
    """amazing awesome incredible insane crazy wild unbelievable shocking terrible horrible worst best
    hate love afraid scared terrified angry furious excited hilarious ridiculous disgusting beautiful
    heartbreaking stupid genius never always nobody everyone dead die killed fired millions billion
    secret truth lie lied wrong omg wow""".split()
)
LAUGH_TOKENS = frozenset({"haha", "hahaha", "lol", "laughter", "laughing", "laughs"})
HOOK_OPENERS = (
    "imagine", "here's the thing", "the craziest", "you won't believe", "listen", "wait",
    "let me tell you", "the worst", "the best", "what if", "why ", "how ", "nobody", "everyone",
)
FRAGMENT_STARTERS = frozenset({"and", "but", "because", "or", "which", "then", "that"})


class CandidateGenerator:
    """Cheap, local, multi-signal candidate window generator (no LLM)."""

    def __init__(self, cfg: CandidateConfig, audio: AudioAnalyzer, scenes: list[Scene]):
        self.cfg = cfg
        self.audio = audio
        self.scenes = scenes
        self._scene_starts = np.array([s.start for s in scenes]) if scenes else np.array([])
        self._scene_ends = np.array([s.end for s in scenes]) if scenes else np.array([])

    # ------------------------------------------------------------------ public
    def generate(self, transcript: Transcript) -> list[CandidateClip]:
        segs = [s for s in transcript.segments if s.text.strip()]
        windows = []
        for i in range(len(segs)):
            for j in range(i, len(segs)):
                dur = segs[j].end - segs[i].start
                if dur > self.cfg.max_duration:
                    break
                if dur < self.cfg.min_duration:
                    continue
                windows.append(self._score_window(segs, i, j))

        kept = self._suppress(windows)
        kept.sort(key=lambda w: w["start"])
        candidates = [self._to_candidate(idx, w) for idx, w in enumerate(kept, start=1)]
        log.info("Generated %d candidates from %d raw windows", len(candidates), len(windows))
        return candidates

    # ------------------------------------------------------------------ scoring
    def _score_window(self, segs: list[TranscriptSegment], i: int, j: int) -> dict:
        window = segs[i : j + 1]
        start, end = window[0].start, window[-1].end
        dur = end - start
        text = " ".join(s.text.strip() for s in window)
        tokens = [t.lower() for t in WORD_RE.findall(text)]

        wps = len(tokens) / dur
        density = min(wps / 3.0, 1.0)
        if wps > 5.0:
            density *= 0.6

        emotion_hits = sum(t in EMOTION_WORDS for t in tokens) + text.count("!")
        emotion = min(emotion_hits / 4.0, 1.0)

        qa = 1.0 if any(s.text.strip().endswith("?") for s in window[:-1]) else 0.0

        last = window[-1].text.strip()
        gap_after = segs[j + 1].start - window[-1].end if j + 1 < len(segs) else 1.0
        punch = 0.0
        if len(last.split()) <= 12:
            punch += 0.4
        if gap_after >= 0.4:
            punch += 0.3
        if last.endswith(("!", "?")) or any(t in LAUGH_TOKENS for t in tokens[-6:]):
            punch += 0.3
        punch = min(punch, 1.0)

        first = window[0].text.strip()
        first_lower = first.lower()
        if "?" in first:
            hook = 1.0
        elif first_lower.startswith(HOOK_OPENERS):
            hook = 0.8
        elif any(t in EMOTION_WORDS for t in WORD_RE.findall(first_lower)):
            hook = 0.5
        else:
            hook = 0.0

        feats = self.audio.features(start, end)
        audio_sig = 0.5 * min(feats.energy / 0.35, 1.0) + 0.5 * feats.peak_energy
        speech = min(feats.speech_ratio / 0.8, 1.0)

        scene = 0.0
        if self._scene_starts.size:
            scene += 0.5 * float(np.min(np.abs(self._scene_starts - start)) <= 1.0)
            scene += 0.5 * float(np.min(np.abs(self._scene_ends - end)) <= 1.0)

        signals = {
            "density": density,
            "emotion": emotion,
            "qa": qa,
            "punchline": punch,
            "hook": hook,
            "audio": audio_sig,
            "scene": scene,
            "speech": speech,
        }
        score = sum(self.cfg.signal_weights.get(k, 0.0) * v for k, v in signals.items())

        max_gap = max((window[k + 1].start - window[k].end for k in range(len(window) - 1)), default=0.0)
        if max_gap > 3.0:
            score -= 0.2
        first_token = tokens[0] if tokens else ""
        if first_token in FRAGMENT_STARTERS:
            score -= 0.1

        return {
            "start": start,
            "end": end,
            "text": text,
            "score": max(score, 0.0),
            "signals": {k: round(v, 3) for k, v in signals.items()},
            "features": feats,
        }

    # ------------------------------------------------------------------ selection
    def _suppress(self, windows: list[dict]) -> list[dict]:
        """Greedy non-max suppression so we don't send near-duplicate windows to the LLM."""
        kept: list[dict] = []
        for w in sorted(windows, key=lambda x: x["score"], reverse=True):
            if all(
                overlap_ratio(w["start"], w["end"], k["start"], k["end"]) < self.cfg.overlap_suppression
                for k in kept
            ):
                kept.append(w)
                if len(kept) >= self.cfg.max_candidates:
                    break
        return kept

    def _to_candidate(self, idx: int, w: dict) -> CandidateClip:
        scene_ids = [s.id for s in self.scenes if s.end > w["start"] and s.start < w["end"]]
        return CandidateClip(
            id=f"candidate_{idx:02d}",
            start=round(w["start"], 3),
            end=round(w["end"], 3),
            transcript=w["text"],
            scene_ids=scene_ids,
            audio_features=w["features"],
            heuristic_score=round(w["score"], 4),
            signals=w["signals"],
        )