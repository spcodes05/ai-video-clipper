import pytest

from src.captions.ass_renderer import format_ass_time, get_style, render_ass
from src.captions.subtitle_generator import group_words, words_for_clip
from src.models.schemas import TranscriptSegment, WordTimestamp
from src.utils.errors import ConfigError, VideoInputError
from src.utils.ffmpeg import probe_video
from src.video.cutter import build_input_args
from src.video.reframer import compute_center_crop
from src.utils.errors import FFmpegError


def test_center_crop_landscape():
    c = compute_center_crop(1920, 1080, 1080, 1920)
    assert c.height == 1080 and c.width % 2 == 0
    assert abs(c.width / c.height - 9 / 16) < 0.01
    assert c.x == (1920 - c.width) // 2


def test_center_crop_portrait_is_untouched():
    c = compute_center_crop(1080, 1920, 1080, 1920)
    assert (c.x, c.y, c.width, c.height) == (0, 0, 1080, 1920)


def test_probe_missing_file():
    with pytest.raises(VideoInputError, match="not found"):
        probe_video("does/not/exist.mp4")


def test_probe_unsupported_extension(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hi")
    with pytest.raises(VideoInputError, match="Unsupported"):
        probe_video(f)


def test_invalid_clip_range():
    with pytest.raises(FFmpegError):
        build_input_args("a.mp4", 10, 5)


def test_ass_time_format():
    assert format_ass_time(3725.456) == "1:02:05.46"
    assert format_ass_time(0) == "0:00:00.00"


def test_words_are_relative_to_clip_and_grouped():
    seg = TranscriptSegment(
        start=10.0, end=14.0, text="I thought he was joking.",
        words=[WordTimestamp(word=w, start=10.0 + i * 0.6, end=10.0 + i * 0.6 + 0.5) for i, w in enumerate("I thought he was joking.".split())],
    )
    words = words_for_clip([seg], 10.0, 14.0)
    assert words[0].start == 0.0 and len(words) == 5
    lines = group_words(words, 4.0, max_words=3)
    assert [l.text for l in lines] == ["I thought he", "was joking."]
    assert all(lines[i].end <= lines[i + 1].start + 1e-9 for i in range(len(lines) - 1))


def test_segments_without_word_timestamps_are_synthesized():
    seg = TranscriptSegment(start=0.0, end=3.0, text="hello brave new world")
    assert len(words_for_clip([seg], 0.0, 3.0)) == 4


def test_ass_output_contains_events_and_resolution():
    seg = TranscriptSegment(start=0.0, end=2.0, text="hello world")
    lines = group_words(words_for_clip([seg], 0.0, 2.0), 2.0)
    ass = render_ass(lines, get_style("bold"), 1080, 1920)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "Dialogue: 0,0:00:00.00" in ass and "HELLO WORLD" in ass


def test_unknown_style_rejected():
    with pytest.raises(ConfigError):
        get_style("nope")