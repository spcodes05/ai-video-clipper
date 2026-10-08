import pytest

from src.analysis.boundaries import refine_boundaries
from src.analysis.scoring import compute_viral_score, select_clips
from src.models.schemas import AudioFeatures, CandidateClip, ClipAnalysis, RankedClip, TranscriptSegment
from src.pipeline.config import BoundaryConfig, ScoringConfig, SelectionConfig, load_config
from src.utils.errors import ConfigError
from src.utils.timeutils import overlap_ratio


def _analysis(**kw) -> ClipAnalysis:
    base = dict(
        candidate_id="c", humor=50, hook=50, surprise=50, emotion=50, shareability=50,
        visual_interest=50, context_completeness=50, payoff=50, overall=50,
        recommended_start=0, recommended_end=30, title="t", reasoning="r",
    )
    base.update(kw)
    return ClipAnalysis(**base)


def _ranked(cid, start, end, score) -> RankedClip:
    cand = CandidateClip(id=cid, start=start, end=end, transcript="x", audio_features=AudioFeatures())
    return RankedClip(
        candidate=cand, analysis=_analysis(candidate_id=cid), viral_score=score,
        final_start=start, final_end=end,
    )


def test_default_weights_sum_to_one():
    assert sum(ScoringConfig().weights.model_dump().values()) == pytest.approx(1.0)


def test_viral_score_formula():
    a = _analysis(hook=100, humor=0, surprise=0, emotion=0, shareability=0, visual_interest=0, context_completeness=0)
    assert compute_viral_score(a, ScoringConfig()) == pytest.approx(20.0)


def test_context_penalty_applied():
    a = _analysis(requires_too_much_context=True)
    assert compute_viral_score(a, ScoringConfig()) == pytest.approx(42.5)


def test_scores_are_clamped():
    assert _analysis(humor=250, hook=-5).humor == 100 and _analysis(hook=-5).hook == 0


def test_overlap_ratio():
    assert overlap_ratio(0, 10, 5, 15) == 0.5
    assert overlap_ratio(0, 10, 10, 20) == 0.0


def test_selection_avoids_overlap():
    ranked = [_ranked("a", 0, 30, 92), _ranked("b", 10, 40, 91), _ranked("c", 100, 130, 80)]
    sel = select_clips(ranked, SelectionConfig(num_clips=2))
    assert [r.candidate.id for r in sel] == ["a", "c"]
    assert [r.clip_number for r in sel] == [1, 2]


def test_selection_allows_overlap_when_enabled():
    ranked = [_ranked("a", 0, 30, 92), _ranked("b", 10, 40, 91)]
    sel = select_clips(ranked, SelectionConfig(num_clips=2, allow_overlap=True))
    assert len(sel) == 2


def test_boundaries_snap_and_stay_in_bounds():
    segs = [
        TranscriptSegment(start=10.0, end=14.0, text="a"),
        TranscriptSegment(start=14.5, end=20.0, text="b"),
        TranscriptSegment(start=22.0, end=30.0, text="c"),
    ]
    s, e = refine_boundaries(10.4, 29.5, segs, 100.0, BoundaryConfig(), 60)
    assert 9.7 <= s <= 10.0 and 30.0 <= e <= 30.5
    s, e = refine_boundaries(10.0, 200.0, segs, 100.0, BoundaryConfig(), 60)
    assert e <= 100.0


def test_invalid_config_rejected():
    with pytest.raises(ConfigError):
        load_config("config.yaml", {"min_duration": 90, "max_duration": 30})