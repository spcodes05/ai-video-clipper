import pytest

from src.analysis.local_analyzer import LocalAnalyzer
from src.analysis.scoring import compute_viral_score
from src.detection.visual_analyzer import VisualAnalyzer
from src.models.schemas import AudioFeatures, CandidateClip, Transcript, TranscriptSegment
from src.pipeline.config import AppConfig, load_config
from src.utils.errors import ConfigError
from src.video.qa import validate_clip

import numpy as np

TEXTS = [
    "and then it happened.",
    "Here is the story.",
    "It was wild.",
    "Nobody expected it!",
    "Then he laughed.",
    "That was the end.",
    "Why did it happen?",
    "and so on",
]


def _transcript() -> Transcript:
    return Transcript(
        segments=[TranscriptSegment(start=i * 4.0, end=i * 4.0 + 3.8, text=t) for i, t in enumerate(TEXTS)]
    )


def _cand(signals=None) -> CandidateClip:
    return CandidateClip(
        id="candidate_01", start=0.0, end=31.8, transcript=" ".join(TEXTS),
        audio_features=AudioFeatures(), signals=signals or {},
    )


def test_fragments_are_trimmed_but_min_duration_kept():
    a = LocalAnalyzer(AppConfig()).analyze([_cand()], _transcript())[0]
    assert a.recommended_start == 4.0
    assert a.recommended_end == pytest.approx(27.8)
    assert a.recommended_end - a.recommended_start >= AppConfig().candidates.min_duration


def test_analysis_is_deterministic():
    sig = {"hook": 0.8, "punchline": 0.6, "audio": 0.4}
    one = LocalAnalyzer(AppConfig()).analyze([_cand(sig)], _transcript())[0]
    two = LocalAnalyzer(AppConfig()).analyze([_cand(sig)], _transcript())[0]
    assert one == two


def test_missing_features_do_not_crash():
    a = LocalAnalyzer(AppConfig(), visual=None).analyze([_cand({})], _transcript())[0]
    assert a.visual_interest == 50.0
    assert "unavailable" in a.reasoning
    assert 0 <= compute_viral_score(a, AppConfig().scoring) <= 100


def test_strong_signals_score_higher():
    weak = LocalAnalyzer(AppConfig()).analyze([_cand({})], _transcript())[0]
    strong = LocalAnalyzer(AppConfig()).analyze(
        [_cand({"hook": 1.0, "punchline": 1.0, "emotion": 1.0, "audio": 1.0, "density": 1.0, "qa": 1.0})],
        _transcript(),
    )[0]
    cfg = AppConfig().scoring
    assert compute_viral_score(strong, cfg) > compute_viral_score(weak, cfg)


def test_title_prefers_question():
    a = LocalAnalyzer(AppConfig()).analyze([_cand()], _transcript())[0]
    assert a.title == "Why did it happen?"


def test_profile_changes_weights():
    cfg = load_config("config.yaml", {"profile": "comedy"})
    assert cfg.scoring.weights.humor == pytest.approx(0.30)
    assert sum(cfg.scoring.weights.model_dump().values()) == pytest.approx(1.0)


def test_unknown_profile_rejected():
    with pytest.raises(ConfigError):
        load_config("config.yaml", {"profile": "nope"})


def test_visual_analyzer_empty_returns_none():
    assert VisualAnalyzer(np.array([])).score(0, 10) is None


def test_visual_analyzer_scores_motion():
    v = VisualAnalyzer(np.array([0.0, 0.0, 10.0, 10.0]), step=1.0)
    assert v.score(2, 4) > v.score(0, 2)


def test_qa_reports_missing_file(tmp_path):
    assert validate_clip(tmp_path / "nope.mp4", 10, 1080, 1920) == ["output file is missing or empty"]