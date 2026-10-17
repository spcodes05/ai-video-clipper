"""Phase 2 integration: config switch, ClipRenderer delegation, timestamps, fallbacks, selection."""
import json
import shutil
import subprocess

import pytest
import yaml

from src.models.schemas import (
    CandidateClip, AudioFeatures, ClipAnalysis, RankedClip, Scene,
    Transcript, TranscriptSegment, WordTimestamp,
)
from src.pipeline.config import AppConfig, load_config
from src.pipeline.orchestrator import Pipeline
from src.pipeline.phase2_bridge import build_phase2_cfg, scene_cuts_from, selection_for
from src.utils.errors import ConfigError, FFmpegError
from src.utils.ffmpeg import probe_video
from src.video.qa import validate_clip
from src.video.reframer import get_reframer
from src.video.renderer import ClipRenderer

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed")


def _cfg(phase2=None, **video):
    data = {"video": {"width": 540, "height": 960, "preset": "ultrafast", **video}}
    if phase2 is not None:
        data["phase2"] = phase2
    return AppConfig.model_validate(data)


def _write_cfg(tmp_path, data):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(p)


def _ranked(start, end, score=50.0, n=0):
    cand = CandidateClip(id=f"c{n}", start=start, end=end, transcript="t",
                         audio_features=AudioFeatures())
    ana = ClipAnalysis(candidate_id=f"c{n}", humor=1, hook=1, surprise=1, emotion=1,
                       shareability=1, visual_interest=1, context_completeness=1, payoff=1,
                       overall=1, recommended_start=start, recommended_end=end,
                       title=f"t{n}", reasoning="r")
    return RankedClip(candidate=cand, analysis=ana, viral_score=score,
                      final_start=start, final_end=end)


def _transcript(duration=12):
    segs = []
    for i in range(duration // 4):
        s = 10.0 + i * 4.0 if duration > 12 else i * 4.0
        words = [WordTimestamp(word=w, start=s + k * 0.8, end=s + k * 0.8 + 0.7)
                 for k, w in enumerate(["This", "is", "wild", f"number{i}."])]
        segs.append(TranscriptSegment(start=s, end=s + 3.5,
                                      text=" ".join(w.word for w in words), words=words))
    return Transcript(segments=segs, duration=duration)


def _make_video(path, seconds=12, rate="25", extra_vf=None, audio=True, rotate=None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate={rate}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    if extra_vf:
        cmd += ["-vf", extra_vf, "-fps_mode", "vfr"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-shortest"] if audio else ["-an"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True)
    return path


# ---------------------------------------------------------------- config switch
def test_phase2_disabled_by_default_and_via_flag(tmp_path):
    p = _write_cfg(tmp_path, {"phase2": {"enabled": False}})
    assert build_phase2_cfg(load_config(p, {})) is None
    assert build_phase2_cfg(load_config(p, {"phase2": True})) is not None
    p2 = _write_cfg(tmp_path, {"phase2": {"enabled": True}})
    assert build_phase2_cfg(load_config(p2, {"phase2": False})) is None


def test_no_phase2_section_means_legacy():
    assert build_phase2_cfg(AppConfig()) is None


def test_enabled_invalid_config_raises_config_error():
    with pytest.raises(ConfigError, match="phase2"):
        build_phase2_cfg(_cfg({"enabled": True, "render": {"fps": 0}}))
    with pytest.raises(ConfigError, match="phase2"):
        build_phase2_cfg(_cfg({"enabled": True, "bogus": 1}))


def test_pipeline_construction_fails_loudly_when_phase2_invalid(tmp_path):
    with pytest.raises(ConfigError):
        Pipeline(_cfg({"enabled": True, "clip": {"min_seconds": 99, "max_seconds": 10}}),
                 tmp_path / "x.mp4")


def test_video_section_is_single_source_for_output_settings():
    p2 = build_phase2_cfg(_cfg({"enabled": True, "render": {"width": 720, "crf": 30}},
                               crf=18, preset="fast", audio_bitrate="128k"))
    assert (p2.render.width, p2.render.height) == (540, 960)
    assert (p2.render.crf, p2.render.preset, p2.render.audio_bitrate) == (18, "fast", "128k")


# ---------------------------------------------------------------- selection / dedupe
def test_selection_overlap_threshold_follows_phase2():
    from src.analysis.scoring import select_clips
    cfg = _cfg({"enabled": True, "clip": {"max_overlap": 0.4}})
    p2 = build_phase2_cfg(cfg)
    a, b = _ranked(0, 40, 90, 1), _ranked(30, 70, 80, 2)     # 25% of the shorter clip
    c = _ranked(20, 60, 70, 3)                               # 50% vs a
    assert [r.candidate.id for r in select_clips([a, b, c], cfg.selection)] == ["c1"]
    sel = select_clips([a, b, c], selection_for(cfg.selection, p2))
    assert [r.candidate.id for r in sel] == ["c1", "c2"]
    assert selection_for(cfg.selection, None) is cfg.selection


# ---------------------------------------------------------------- timestamps
def test_scene_cuts_are_source_times_without_first_scene():
    scenes = [Scene(id=0, start=0, end=30), Scene(id=1, start=30, end=75.5)]
    assert scene_cuts_from(scenes) == [30.0]


def test_renderer_passes_source_timestamps_to_phase2(tmp_path, monkeypatch):
    import phase2.pipeline as p2pipe
    seen = {}

    def fake_render(src, out, start, end, words, cfg, **kw):
        seen.update(start=start, end=end, words=words, kw=kw)
        out.write_bytes(b"x")
        return p2pipe.RenderResult(out, end - start, True, True)

    monkeypatch.setattr(p2pipe, "render_clip", fake_render)
    cfg = _cfg({"enabled": True})
    r = ClipRenderer(cfg.video, cfg.captions, get_reframer("center"), tmp_path / "captions",
                     phase2_cfg=build_phase2_cfg(cfg))
    clip = _ranked(14.0, 22.0)
    out = tmp_path / "clip_01.mp4"
    item = r.render(tmp_path / "src.mp4", None, clip, _transcript(24), out, scene_cuts=[18.2])
    assert (seen["start"], seen["end"]) == (14.0, 22.0) and item.start == 14.0
    assert seen["words"] and all(14.0 - 1e-6 <= w.start and w.end <= 22.0 + 1e-6 for w in seen["words"])
    assert seen["words"][0].start == pytest.approx(14.0, abs=0.01)
    assert seen["kw"]["scene_cuts"] == [18.2]
    assert seen["kw"]["speech_segments"] == [(14.0, 17.5), (18.0, 21.5)]   # source time, in-clip only


def test_phase2_failure_is_reported_not_swallowed(tmp_path, monkeypatch, caplog):
    import phase2.pipeline as p2pipe
    from phase2.ffmpeg_utils import OutputValidationError

    def boom(*a, **k):
        raise OutputValidationError("resolution 1x1")

    monkeypatch.setattr(p2pipe, "render_clip", boom)
    cfg = _cfg({"enabled": True})
    r = ClipRenderer(cfg.video, cfg.captions, get_reframer("center"), tmp_path / "c",
                     phase2_cfg=build_phase2_cfg(cfg))
    with caplog.at_level("ERROR"):
        with pytest.raises(FFmpegError, match="Phase 2 render failed"):
            r.render(tmp_path / "s.mp4", None, _ranked(0, 8), _transcript(), tmp_path / "o.mp4")
    assert "Phase 2 render failed" in caplog.text


def test_legacy_path_untouched_when_disabled(tmp_path, monkeypatch):
    import phase2.pipeline as p2pipe
    monkeypatch.setattr(p2pipe, "render_clip",
                        lambda *a, **k: pytest.fail("phase2 must not run when disabled"))
    cfg = _cfg()
    r = ClipRenderer(cfg.video, cfg.captions, get_reframer("center"), tmp_path / "c")
    assert r.phase2_cfg is None


# ---------------------------------------------------------------- real renders
class _FakeDetector:
    """Stands in for HaarFaceDetector: reports a face on the right side of the frame."""
    def __init__(self, *a, **k):
        import cv2
        self.cv2 = cv2

    def detect(self, frame, t, analysis_width=640, next_frame=None):
        from phase2.faces import FaceBox
        h, w = frame.shape[:2]
        return [FaceBox(t, w * 0.78, h * 0.2, w * 0.1, w * 0.1, 9.0, 0.0)]


@needs_ffmpeg
def test_end_to_end_phase2_through_clip_renderer_with_tracking(tmp_path, monkeypatch):
    import phase2.pipeline as p2pipe
    monkeypatch.setattr(p2pipe, "HaarFaceDetector", _FakeDetector)
    src = _make_video(tmp_path / "src clip ñ.mp4")
    meta = probe_video(src)
    cfg = _cfg({"enabled": True, "captions": {"enabled": True}})
    p2 = build_phase2_cfg(cfg)
    r = ClipRenderer(cfg.video, cfg.captions, get_reframer("center"),
                     tmp_path / "analysis" / "captions", phase2_cfg=p2)
    out = tmp_path / "clips" / "clip_01.mp4"
    item = r.render(src, meta, _ranked(0.0, 8.0), _transcript(), out)
    assert validate_clip(out, item.duration, 540, 960) == []
    dbg = json.loads((tmp_path / "analysis" / "phase2" / "clip_01.analysis.json").read_text("utf-8"))
    assert dbg["samples"] and dbg["keyframes"][-1][1] > (1280 - dbg["crop"]["cw"]) / 2 + 50


@needs_ffmpeg
def test_phase2_without_detector_falls_back_to_center_and_warns(tmp_path, monkeypatch, caplog):
    import phase2.pipeline as p2pipe
    from phase2.faces import DetectorUnavailable

    class Unavailable:
        def __init__(self, *a, **k):
            raise DetectorUnavailable("no detector here")

    monkeypatch.setattr(p2pipe, "HaarFaceDetector", Unavailable)
    src = _make_video(tmp_path / "s.mp4")
    cfg = _cfg({"enabled": True})
    r = ClipRenderer(cfg.video, cfg.captions, get_reframer("center"), tmp_path / "a" / "captions",
                     phase2_cfg=build_phase2_cfg(cfg))
    out = tmp_path / "o.mp4"
    with caplog.at_level("WARNING"):
        item = r.render(src, probe_video(src), _ranked(0.0, 8.0), _transcript(), out)
    assert validate_clip(out, item.duration, 540, 960) == []
    assert "centered crop" in caplog.text


@needs_ffmpeg
def test_variable_frame_rate_source(tmp_path, monkeypatch):
    import phase2.pipeline as p2pipe
    monkeypatch.setattr(p2pipe, "HaarFaceDetector", _FakeDetector)
    src = _make_video(tmp_path / "vfr.mp4", extra_vf="select='mod(n,3)',setpts=N/(25/3)/TB" if False
                      else "select='not(mod(n,4))+not(mod(n,7))'")
    cfg = _cfg({"enabled": True})
    r = ClipRenderer(cfg.video, cfg.captions, get_reframer("center"), tmp_path / "a" / "captions",
                     phase2_cfg=build_phase2_cfg(cfg))
    out = tmp_path / "o.mp4"
    item = r.render(src, probe_video(src), _ranked(0.0, 4.0), _transcript(), out)
    assert validate_clip(out, item.duration, 540, 960, tolerance=1.5) == []


@needs_ffmpeg
def test_rotation_metadata_swaps_effective_size(tmp_path):
    from phase2.ffmpeg_utils import video_info
    base = _make_video(tmp_path / "base.mp4", seconds=2)
    rot = tmp_path / "rot.mp4"
    res = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", "90",
                          "-i", str(base), "-c", "copy", str(rot)], capture_output=True)
    if res.returncode != 0:
        pytest.skip("this ffmpeg cannot write display rotation")
    info = video_info(rot)
    assert info.rotation in (90, 270)


@needs_ffmpeg
def test_caption_fallback_after_styled_attempt_fails(tmp_path, monkeypatch, caplog):
    import phase2.pipeline as p2pipe
    from phase2.boundaries import Word
    real = p2pipe.build_video_filter
    calls = []

    def flaky(plan, w, h, fps, keep, ass_name, fonts_dir=None):
        calls.append(ass_name)
        vf = real(plan, w, h, fps, keep, ass_name, fonts_dir)
        return vf.replace("captions.ass", "missing.ass") if len(calls) == 1 else vf

    monkeypatch.setattr(p2pipe, "build_video_filter", flaky)
    src = _make_video(tmp_path / "s.mp4", seconds=6)
    cfg = _cfg({"enabled": True, "framing": {"enabled": False}})
    p2 = build_phase2_cfg(cfg)
    words = [Word(0.2, 0.6, "Hello"), Word(0.7, 1.2, "there.")]
    with caplog.at_level("WARNING"):
        res = p2pipe.render_clip(src, tmp_path / "o.mp4", 0.0, 4.0, words, p2)
    assert len(calls) == 2 and res.captions_ok is False
    assert "plain captions" in caplog.text
    assert validate_clip(tmp_path / "o.mp4", res.duration, 540, 960) == []


@needs_ffmpeg
def test_real_face_detector_keeps_off_center_face_in_frame(tmp_path):
    """Uses OpenCV's real Haar detector on a real face (skipped without scikit-image/OpenCV 4.x)."""
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    data = pytest.importorskip("skimage.data")
    if not hasattr(cv2, "CascadeClassifier"):
        pytest.skip("OpenCV build without CascadeClassifier")
    canvas = np.zeros((720, 1280, 3), np.uint8) + 40
    canvas[100:612, 700:1212] = data.astronaut()[:, :, ::-1]
    png = tmp_path / "face.png"
    cv2.imwrite(str(png), canvas)
    src = tmp_path / "face.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-framerate", "25",
                    "-i", str(png), "-f", "lavfi", "-i", "sine=frequency=300", "-t", "6", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    cfg = _cfg({"enabled": True})
    r = ClipRenderer(cfg.video, cfg.captions, get_reframer("center"), tmp_path / "a" / "captions",
                     phase2_cfg=build_phase2_cfg(cfg))
    out = tmp_path / "o.mp4"
    item = r.render(src, probe_video(src), _ranked(0.0, 5.0), _transcript(), out)
    assert validate_clip(out, item.duration, 540, 960) == []
    frame = tmp_path / "f.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "2", "-i", str(out), "-frames:v", "1", str(frame)],
                   check=True)
    from phase2.faces import HaarFaceDetector
    assert HaarFaceDetector().detect(cv2.imread(str(frame)), 0.0, 640), "face missing from tracked crop"


@needs_ffmpeg
def test_long_clip_with_moving_face_renders_and_non_caption_errors_are_not_retried(tmp_path, monkeypatch, caplog):
    """60 s clip, face jumping every sample: >100 raw keyframes used to break FFmpeg's crop parser."""
    import phase2.pipeline as p2pipe
    from phase2.faces import FaceBox

    class Jumpy:
        def __init__(self, *a, **k):
            import cv2
            self.cv2 = cv2

        def detect(self, frame, t, analysis_width=640, next_frame=None):
            h, w = frame.shape[:2]
            return [FaceBox(t, w * (0.1 + 0.7 * ((int(t * 2) * 37) % 100) / 100), h * 0.2, w * 0.1, w * 0.1, 9.0, 0.0)]

    monkeypatch.setattr(p2pipe, "HaarFaceDetector", Jumpy)
    src = tmp_path / "long.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=640x360:rate=10:duration=60", "-f", "lavfi", "-i", "sine=frequency=440:duration=60",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    str(src)], check=True)
    cfg = _cfg({"enabled": True, "framing": {"sample_interval": 0.25, "keyframe_interval": 0.25,
                                             "deadzone": 0.0, "smoothing": 1.0, "max_speed": 5.0}})
    p2 = build_phase2_cfg(cfg)
    out = tmp_path / "o.mp4"
    res = p2pipe.render_clip(src, out, 0.0, 60.0, [], p2, debug_dir=tmp_path / "dbg")
    assert validate_clip(out, res.duration, 540, 960, tolerance=1.5) == []
    kf = json.loads((tmp_path / "dbg" / "o.analysis.json").read_text("utf-8"))["keyframes"]
    assert 2 < len(kf) <= 150

    # a crop/filter error must surface immediately instead of triggering the caption retry
    real = p2pipe.build_video_filter
    calls = []
    monkeypatch.setattr(p2pipe, "build_video_filter",
                        lambda *a, **k: calls.append(1) or real(*a, **k).replace("crop=", "crop=bogus", 1))
    from phase2.boundaries import Word
    with caplog.at_level("WARNING"):
        with pytest.raises(Exception) as exc:
            p2pipe.render_clip(src, tmp_path / "o2.mp4", 0.0, 4.0, [Word(0.2, 0.6, "Hi"), Word(0.7, 1.0, "there.")], p2)
    assert type(exc.value).__name__ == "FFmpegError" and len(calls) == 1
    assert "plain captions" not in caplog.text
