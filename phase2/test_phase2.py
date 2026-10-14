import shutil
import subprocess
import sys

import pytest

from phase2.boundaries import (Candidate, Word, compute_keep_segments, dedupe,
                               refine_boundaries, retime_words)
from phase2.captions import build_ass, group_words, line_breaks
from phase2.config import CaptionCfg, ClipCfg, ConfigError, FramingCfg, Phase2Cfg
from phase2.crop import (CropPlan, crop_size, effective_size, output_size,
                         smooth_path, thin_keyframes, x_expression)
from phase2.faces import DetectorUnavailable, FaceBox, HaarFaceDetector
from phase2.ffmpeg_utils import (FFmpegError, build_render_command, build_video_filter,
                                 video_info)
from phase2.scoring import combine, normalize
from phase2.tracking import track_subject


def make_words(n=400):
    return [Word(i * 0.5, i * 0.5 + 0.4, f"w{i}" + ("." if i % 10 == 9 else ""))
            for i in range(n)]


# ---- config
def test_config_defaults_valid():
    assert Phase2Cfg.from_config({}).clip.max_seconds == 90


def test_config_rejects_bad_durations():
    with pytest.raises(ConfigError):
        Phase2Cfg.from_config({"phase2": {"clip": {"min_seconds": 100, "max_seconds": 90}}})


def test_config_rejects_unknown_key_and_type():
    with pytest.raises(ConfigError):
        Phase2Cfg.from_config({"phase2": {"clip": {"nope": 1}}})
    with pytest.raises(ConfigError):
        Phase2Cfg.from_config({"phase2": {"render": {"fps": "fast"}}})


def test_config_rejects_odd_resolution_and_color():
    with pytest.raises(ConfigError):
        Phase2Cfg.from_config({"phase2": {"render": {"width": 1081}}})
    with pytest.raises(ConfigError):
        Phase2Cfg.from_config({"phase2": {"captions": {"primary": "red"}}})


# ---- boundaries
def test_refine_natural_boundaries_and_duration():
    clip = ClipCfg()
    c = refine_boundaries(make_words(), 23.3, 80.0, clip)
    assert c is not None
    assert c.start <= 20.0 and c.end >= 84.9
    assert clip.min_seconds <= c.duration <= clip.max_seconds


def test_refine_rejects_instead_of_padding():
    words = make_words(40)  # only 20 s of material
    assert refine_boundaries(words, 2.0, 15.0, ClipCfg()) is None


def test_dedupe_overlapping():
    a, b = Candidate(0, 60, 0.9), Candidate(5, 62, 0.5)
    c = Candidate(200, 260, 0.1)
    kept = dedupe([a, b, c], 0.4)
    assert a in kept and c in kept and b not in kept


def test_keep_segments_only_cut_long_gaps():
    words = [Word(0, 1, "a"), Word(1.1, 2, "b."), Word(6, 7, "c.")]
    keep = compute_keep_segments(words, 7.5, 1.2, 0.4)
    assert len(keep) == 2 and keep[0][1] < 6 and keep[1][0] > 2
    rt = retime_words(words, keep)
    assert rt[2].start < words[2].start


# ---- crop / framing
@pytest.mark.parametrize("w,h,exp", [
    (1920, 1080, (606, 1080)), (1080, 1920, (1080, 1920)),
    (1080, 1080, (606, 1080)), (1080, 2400, (1080, 1920)),
])
def test_crop_size(w, h, exp):
    assert crop_size(w, h, 1080, 1920) == exp


def test_effective_size_rotation():
    assert effective_size(1920, 1080, 90) == (1080, 1920)
    assert effective_size(1920, 1080, 0) == (1920, 1080)


def test_output_size_caps_upscale():
    assert output_size(360, 1080, 1920, 2.0) == (404, 720)
    assert output_size(1080, 1080, 1920, 2.0) == (1080, 1920)


def test_smooth_path_limits_speed_and_jumps_only_on_cut():
    cfg = FramingCfg()
    raw = [(i * 0.5, 400.0 if i < 2 else 1400.0) for i in range(10)]
    p = smooth_path(raw, 1920, 606, cfg)
    for (t0, x0, _), (t1, x1, j) in zip(p, p[1:]):
        assert abs(x1 - x0) <= cfg.max_speed * 606 * (t1 - t0) + 1e-6
        assert not j
    pc = smooth_path(raw, 1920, 606, cfg, cuts=[1.2])
    assert any(j for _, _, j in pc)


def test_tracking_fallback_holds_last_position():
    cfg = FramingCfg()
    box = FaceBox(0, 900, 300, 200, 200)
    samples = [(0.0, [box]), (0.5, [box]), (1.0, []), (1.5, [])]
    subj = track_subject(samples, 1920, cfg)
    assert subj[2].cx is None
    path = smooth_path([(s.t, s.cx) for s in subj], 1920, 606, cfg)
    assert abs(path[-1][1] - path[1][1]) < 1.0


def test_x_expression_formats():
    assert x_expression([(0.0, 100.0)]) == "100.00"
    e = x_expression([(0.0, 0.0), (1.0, 50.0), (2.0, 50.0)])
    assert e.startswith("if(lt(t,") and " " not in e
    kf = thin_keyframes([(0, 0, False), (0.5, 1, False), (1.0, 900, True)], 1.0)
    assert kf[-1] == (1.0, 900)


# ---- captions
def test_caption_grouping_and_wrapping():
    cfg = CaptionCfg()
    words = [Word(i * 0.3, i * 0.3 + 0.25, w) for i, w in enumerate(
        "this is a fairly long sentence with many words today.".split())]
    groups = group_words(words, cfg)
    assert all(len(g) <= cfg.max_words for g in groups)
    assert len(line_breaks(["averyverylongword", "another", "third"], 10, 2)) <= 2


def test_ass_timing_monotonic_and_escaped():
    cfg = CaptionCfg()
    words = [Word(0.0, 0.3, "hello{x}"), Word(0.3, 0.6, "world."), Word(1.0, 1.4, "next")]
    ass = build_ass(words, cfg, 1080, 1920)
    ev = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert ev and "{x}" not in ass
    starts = [l.split(",")[1] for l in ev]
    assert starts == sorted(starts)
    plain = build_ass(words, CaptionCfg(word_highlight=False), 1080, 1920)
    assert plain.count("Dialogue:") == 2


# ---- ffmpeg construction / Windows paths
def test_filter_has_no_absolute_paths_and_cmd_keeps_unicode_spaces():
    plan = CropPlan(606, 1080, "100.00", 0)
    vf = build_video_filter(plan, 1080, 1920, 30, None, "captions.ass")
    assert "subtitles=filename=captions.ass" in vf and "\\" not in vf and "C:" not in vf
    src = r"D:\my videos\पाठ clip.mp4"
    cmd = build_render_command(src, r"D:\out dir\o.mp4", 1.5, 60, vf, None, 20,
                               "medium", "192k", True)
    assert src in cmd and r"D:\out dir\o.mp4" in cmd and cmd.index("-ss") < cmd.index("-i")


# ---- optional dependency absence
def test_detector_unavailable_without_cv2(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(DetectorUnavailable):
        HaarFaceDetector()


# ---- scoring
def test_scoring_normalizes_and_explains():
    assert normalize([1, 1]) == [0.5, 0.5]
    res = combine([{"opening": 1.0, "visual": 0.0}, {"opening": 0.0, "visual": 1.0}],
                  {"opening": 0.7, "visual": 0.3})
    assert res[0][0] > res[1][0] and set(res[0][1]) == {"opening", "visual"}


# ---- audio heuristic
def test_modulated_energy_scores_above_steady():
    np = pytest.importorskip("numpy")
    from phase2.audio_analysis import analyze_samples, laughter_scores
    sr = 16000
    t = np.arange(sr * 10) / sr
    steady = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    burst = (0.3 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)) * 32767).astype("<i2")
    s = max((x[2] for x in laughter_scores(analyze_samples(steady, sr), [])), default=0)
    b = max((x[2] for x in laughter_scores(analyze_samples(burst, sr), [])), default=0)
    assert b > s
    covered = max((x[2] for x in laughter_scores(analyze_samples(burst, sr), [(0, 10)])), default=0)
    assert covered < b


# ---- failure handling and synthetic end-to-end
def test_corrupt_input_raises_clear_error(tmp_path):
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe not installed")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(FFmpegError) as e:
        video_info(bad)
    assert e.value.stage == "probe"


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                    reason="ffmpeg not installed")
def test_end_to_end_synthetic(tmp_path):
    from phase2.pipeline import render_clip
    src = tmp_path / "src clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=1280x720:rate=25:duration=6", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=6", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", str(src)], check=True)
    cfg = Phase2Cfg.from_config({"phase2": {
        "clip": {"min_seconds": 1, "target_seconds": 3, "max_seconds": 10},
        "render": {"width": 540, "height": 960, "preset": "ultrafast", "loudnorm": True},
        "framing": {"enabled": True}}})
    words = [Word(0.2, 0.6, "Hello"), Word(0.7, 1.2, "there."), Word(1.5, 2.0, "Testing"),
             Word(2.1, 2.8, "captions.")]
    res = render_clip(src, tmp_path / "out.mp4", 0.0, 4.0, words, cfg)
    info = video_info(res.output)
    assert (info.width, info.height) == (540, 960) and info.has_audio
    assert abs(info.duration - 4.0) < 0.75