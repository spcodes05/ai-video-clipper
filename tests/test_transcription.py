from src.models.schemas import Transcript, TranscriptSegment
from src.transcription.whisper_service import transcript_to_text
from src.utils.timeutils import format_timestamp


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00.000"
    assert format_timestamp(3725.5) == "01:02:05.500"
    assert format_timestamp(-3) == "00:00:00.000"


def test_transcript_text_preserves_timestamps():
    t = Transcript(segments=[TranscriptSegment(start=0.0, end=4.2, text="Yesterday I went to the store.")])
    assert transcript_to_text(t) == "[00:00:00.000 -> 00:00:04.200] Yesterday I went to the store."