import dataclasses
import shutil
import subprocess

import pytest

from phase2.boundaries import Word
from phase2.config import ConfigError, Phase2Cfg
from phase2.ffmpeg_utils import build_blur_fit_filter
from phase2.pipeline import render_clip


def test_filter_keeps_whole_frame_over_blurred_background():
    vf = build_blur_fit_filter(854, 480, 1080, 1920, 30, None, "c.ass", None)
    assert "split=2[a][b]" in vf and "scale=1080:606" in vf  # 854x480 -> 1080 wide, even height
    assert "overlay=(W-w)/2:(H-h)/2" in vf and vf.endswith("format=yuv420p")
    assert "subtitles=filename=c.ass" in vf


def test_filter_applies_silence_removal_before_split():
    vf = build_blur_fit_filter(854, 480, 1080, 1920, 30, [(0.0, 2.0)], None, None)
    assert vf.index("select=") < vf.index("split=2")


def test_config_validation():
    with pytest.raises(ConfigError):
        Phase2Cfg.from_config({"phase2": {"enabled": True, "framing": {"blur_fit": "sometimes"}}})
    with pytest.raises(ConfigError):
        Phase2Cfg.from_config({"phase2": {"enabled": True, "framing": {"min_face_ratio": 2}}})
    cfg = Phase2Cfg.from_config({"phase2": {"enabled": True}})
    assert cfg.framing.blur_fit == "auto" and cfg.framing.min_face_ratio == 0.6


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_render_uses_blur_fit_when_no_faces(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=854x480:rate=30", "-f", "lavfi", "-i", "sine=frequency=440",
                    "-t", "4", "-pix_fmt", "yuv420p", str(src)], check=True)
    cfg = Phase2Cfg.from_config({"phase2": {"enabled": True}})
    cfg = dataclasses.replace(cfg, render=dataclasses.replace(cfg.render, max_upscale=1e9))
    words = [Word(0.5, 1.0, "hello"), Word(1.1, 1.6, "there")]
    debug = tmp_path / "dbg"
    render_clip(src, tmp_path / "o.mp4", 0.0, 3.0, words, cfg, debug_dir=debug)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(tmp_path / "o.mp4")],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "1080,1920"
    assert '"layout": "blur_fit"' in (debug / "o.analysis.json").read_text()
