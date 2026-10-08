"""End-to-end smoke test: real FFmpeg render, fake LLM, no Whisper, no API key."""
import json
import re
import shutil
import subprocess

import numpy as np
import pytest

from src.analysis.boundaries import refine_boundaries
from src.analysis.llm_analyzer import LLMAnalyzer
from src.analysis.scoring import rank_clips, select_clips
from src.detection.audio_analyzer import AudioAnalyzer
from src.detection.candidate_generator import CandidateGenerator
from src.llm.base import LLMProvider
from src.models.schemas import Scene, Transcript, TranscriptSegment, WordTimestamp
from src.pipeline.config import AppConfig
from src.utils.ffmpeg import probe_video
from src.video.reframer import get_reframer
from src.video.renderer import ClipRenderer

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

DURATION = 40


class FakeProvider(LLMProvider):
    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        items = []
        pattern = r"### (candidate_\d+)\ncandidate range: ([\d.]+)s - ([\d.]+)s"
        for cid, start, end in re.findall(pattern, user_prompt):
            items.append(
                {
                    "candidate_id": cid,
                    "humor": 70, "hook": 80, "surprise": 60, "emotion": 60,
                    "shareability": 65, "visual_interest": 50,
                    "context_completeness": 90, "payoff": 70, "overall": 72,
                    "recommended_start": float(start), "recommended_end": float(end),
                    "title": "Fake title", "reasoning": "Fake reasoning.",
                    "requires_too_much_context": False,
                }
            )
        return json.dumps({"analyses": items})


@pytest.fixture
def sample_video(tmp_path):
    out = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=1280x720:rate=25:duration={DURATION}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={DURATION}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out),
        ],
        check=True,
    )
    return out


def _transcript() -> Transcript:
    segs = []
    for i in range(DURATION // 4):
        s = i * 4.0
        words = [
            WordTimestamp(word=w, start=s + k * 0.8, end=s + k * 0.8 + 0.7)
            for k, w in enumerate(["this", "is", "wild", f"number{i}!"])
        ]
        segs.append(TranscriptSegment(start=s, end=s + 3.5, text=" ".join(w.word for w in words), words=words))
    return Transcript(segments=segs, duration=DURATION)


def test_end_to_end_render(sample_video, tmp_path):
    cfg = AppConfig()
    cfg.candidates.min_duration = 10
    cfg.candidates.max_duration = 25
    cfg.selection.num_clips = 1

    meta = probe_video(sample_video)
    assert meta.has_audio and meta.width == 1280

    transcript = _transcript()
    audio = AudioAnalyzer(np.random.default_rng(1).uniform(0.05, 0.3, DURATION * 10))
    candidates = CandidateGenerator(cfg.candidates, audio, [Scene(id=0, start=0, end=DURATION)]).generate(transcript)
    assert candidates

    analyses = LLMAnalyzer(FakeProvider(), cfg.llm).analyze(candidates, transcript)
    assert len(analyses) == len(candidates)

    ranked = rank_clips(candidates, analyses, cfg.scoring)
    for r in ranked:
        r.final_start, r.final_end = refine_boundaries(
            r.analysis.recommended_start, r.analysis.recommended_end,
            transcript.segments, meta.duration, cfg.boundaries, cfg.candidates.max_duration,
        )
    selected = select_clips(ranked, cfg.selection)
    assert len(selected) == 1

    renderer = ClipRenderer(cfg.video, cfg.captions, get_reframer("center"), tmp_path / "captions")
    out = tmp_path / "clip_01.mp4"
    rendered = renderer.render(sample_video, meta, selected[0], transcript, out)

    assert out.exists() and out.stat().st_size > 0
    result = probe_video(out)
    assert (result.width, result.height) == (1080, 1920)
    assert result.has_audio
    assert abs(result.duration - rendered.duration) < 1.0
    assert (tmp_path / "captions" / "clip_01.ass").exists()