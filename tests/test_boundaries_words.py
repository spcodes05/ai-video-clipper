import numpy as np

from src.analysis.boundaries import refine_boundaries
from src.analysis.local_analyzer import LocalAnalyzer
from src.detection.audio_analyzer import FRAME_S, AudioAnalyzer
from src.detection.laughter import peak_in
from src.models.schemas import (AudioFeatures, CandidateClip, Transcript,
                                TranscriptSegment, WordTimestamp)
from src.pipeline.config import AppConfig, BoundaryConfig


def _w(text, a, b):
    return WordTimestamp(word=text, start=a, end=b)


def _segs():
    # one long Whisper segment holding two sentences: snapping to the segment edge would
    # cut mid-thought, word-level snapping finds the sentence end at 14.8
    words = [_w("So", 10.0, 10.3), _w("my", 10.3, 10.5), _w("dog", 10.5, 10.9), _w("left.", 10.9, 11.4),
             _w("Then", 12.0, 12.3), _w("it", 12.3, 12.4), _w("came", 12.4, 12.8), _w("back!", 12.8, 14.8),
             _w("And", 15.0, 15.2), _w("then", 15.2, 15.5)]
    return [TranscriptSegment(start=10.0, end=15.5, text="x", words=words),
            TranscriptSegment(start=20.0, end=22.0, text="y", words=[_w("Okay.", 20.0, 22.0)])]


def test_word_snap_picks_sentence_end_not_segment_end():
    s, e = refine_boundaries(10.0, 15.4, _segs(), 100.0, BoundaryConfig(), 60)
    assert 14.8 <= e <= 15.0  # holds the tail but stops before "And"
    assert s >= 9.75


def test_word_snap_can_be_disabled():
    s, e = refine_boundaries(10.0, 15.4, _segs(), 100.0, BoundaryConfig(word_snap=False), 60)
    assert e > 15.4 - 1e-6


def _audio(loud_from, loud_to, total=30.0):
    rms = np.full(int(total / FRAME_S), 0.01, dtype=np.float32)
    rms[int(loud_from / FRAME_S): int(loud_to / FRAME_S)] = 0.5
    rms[0] = 1.0  # sets the reference level
    return AudioAnalyzer(rms)


def test_laugh_tail_extends_end_when_reaction_follows():
    segs = [TranscriptSegment(start=0.0, end=10.0, text="Joke.", words=[_w("Joke.", 9.0, 10.0)]),
            TranscriptSegment(start=20.0, end=22.0, text="Next.", words=[_w("Next.", 20.0, 22.0)])]
    base = refine_boundaries(0.0, 10.0, segs, 60.0, BoundaryConfig(), 60)
    held = refine_boundaries(0.0, 10.0, segs, 60.0, BoundaryConfig(), 60, audio=_audio(10.0, 11.4))
    assert held[1] > base[1] + 0.5 and held[1] <= 10.0 + 1.5 + 1e-6


def test_laugh_tail_never_crosses_next_speech():
    segs = [TranscriptSegment(start=0.0, end=10.0, text="Joke.", words=[_w("Joke.", 9.0, 10.0)]),
            TranscriptSegment(start=10.6, end=12.0, text="Next.", words=[_w("Next.", 10.6, 12.0)])]
    _, e = refine_boundaries(0.0, 10.0, segs, 60.0, BoundaryConfig(), 60, audio=_audio(10.0, 12.0))
    assert e <= 10.6 + 1e-6


def test_no_extension_without_reaction():
    segs = [TranscriptSegment(start=0.0, end=10.0, text="Joke.", words=[_w("Joke.", 9.0, 10.0)])]
    base = refine_boundaries(0.0, 10.0, segs, 60.0, BoundaryConfig(), 60)
    quiet = refine_boundaries(0.0, 10.0, segs, 60.0, BoundaryConfig(), 60, audio=_audio(25.0, 26.0))
    assert quiet == base


def test_peak_in():
    w = [(0.0, 1.5, 0.2), (5.0, 6.5, 0.9)]
    assert peak_in(w, 4.0, 8.0) == 0.9 and peak_in(w, 2.0, 4.0) == 0.0


def test_laughter_raises_humor_and_payoff():
    segs = [TranscriptSegment(start=i * 4.0, end=i * 4.0 + 3.8, text=t) for i, t in enumerate(
        ["Here is the story.", "It was wild.", "Nobody expected it.", "He sat down.", "Then he fell!",
         "Everyone lost it.", "That was the end.", "Done."])]
    tr = Transcript(segments=segs)
    c = CandidateClip(id="candidate_01", start=0.0, end=31.8, transcript="t", audio_features=AudioFeatures())
    cfg = AppConfig()
    plain = LocalAnalyzer(cfg).analyze([c], tr)[0]
    laughed = LocalAnalyzer(cfg, None, [(10.0, 11.5, 1.0), (31.9, 33.4, 1.0)]).analyze([c], tr)[0]
    assert laughed.humor > plain.humor and laughed.payoff > plain.payoff
