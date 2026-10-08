import numpy as np

from src.detection.audio_analyzer import AudioAnalyzer
from src.detection.candidate_generator import CandidateGenerator
from src.models.schemas import Scene, Transcript, TranscriptSegment
from src.pipeline.config import CandidateConfig
from src.utils.timeutils import overlap_ratio


def _transcript(n=30, seg_len=4.0) -> Transcript:
    segs = [
        TranscriptSegment(
            start=i * seg_len,
            end=(i + 1) * seg_len - 0.2,
            text=f"This is sentence number {i}, and it is pretty wild!" if i % 5 == 0 else f"Sentence {i} continues here.",
        )
        for i in range(n)
    ]
    return Transcript(segments=segs)


def _audio(seconds=130) -> AudioAnalyzer:
    rng = np.random.default_rng(0)
    return AudioAnalyzer(rng.uniform(0.02, 0.3, int(seconds / 0.1)))


def test_audio_features():
    a = AudioAnalyzer(np.array([0.0] * 10 + [1.0] * 10))
    quiet, loud = a.features(0.0, 1.0), a.features(1.0, 2.0)
    assert quiet.silence_ratio == 1.0 and quiet.energy == 0.0
    assert loud.speech_ratio == 1.0 and loud.energy == 1.0


def test_candidates_respect_duration_and_overlap():
    cfg = CandidateConfig(min_duration=15, max_duration=60, max_candidates=10)
    gen = CandidateGenerator(cfg, _audio(), [Scene(id=0, start=0, end=120)])
    cands = gen.generate(_transcript())
    assert 0 < len(cands) <= 10
    assert len({c.id for c in cands}) == len(cands)
    for c in cands:
        assert cfg.min_duration <= c.duration <= cfg.max_duration
    for i, a in enumerate(cands):
        for b in cands[i + 1 :]:
            assert overlap_ratio(a.start, a.end, b.start, b.end) < cfg.overlap_suppression


def test_empty_transcript_yields_no_candidates():
    gen = CandidateGenerator(CandidateConfig(), _audio(), [])
    assert gen.generate(Transcript(segments=[])) == []