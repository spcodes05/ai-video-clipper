from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .crop import CropPlan

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BIN", "ffprobe")


class FFmpegError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


class OutputValidationError(RuntimeError):
    pass


def run(cmd: Sequence[str], stage: str, cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    try:
        p = subprocess.run(list(cmd), capture_output=True, cwd=cwd, check=False)
    except FileNotFoundError as e:
        raise FFmpegError(stage, f"executable not found: {cmd[0]} "
                                 "(install FFmpeg and add it to PATH, or set FFMPEG_BIN/FFPROBE_BIN)") from e
    if p.returncode != 0:
        tail = p.stderr.decode("utf-8", "replace").strip().splitlines()[-15:]
        raise FFmpegError(stage, "\n".join(tail) or f"exit code {p.returncode}")
    return p


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    rotation: int
    duration: float
    fps: float
    has_audio: bool


def _fraction(s: Optional[str]) -> float:
    try:
        v = float(Fraction(s)) if s else 0.0
    except (ZeroDivisionError, ValueError):
        v = 0.0
    return v


def video_info(path) -> VideoInfo:
    p = run([FFPROBE, "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", str(path)], "probe")
    try:
        data = json.loads(p.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError as e:
        raise FFmpegError("probe", f"unreadable ffprobe output for {path}") from e
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise FFmpegError("probe", f"no video stream in {path}")
    rot = 0
    tag = (v.get("tags") or {}).get("rotate")
    if tag is not None:
        rot = int(float(tag))
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    fps = _fraction(v.get("avg_frame_rate")) or _fraction(v.get("r_frame_rate")) or 30.0
    dur = float(v.get("duration") or data.get("format", {}).get("duration") or 0.0)
    return VideoInfo(int(v["width"]), int(v["height"]), rot % 360, dur, fps,
                     any(s.get("codec_type") == "audio" for s in streams))


def _between(keep: Sequence[Tuple[float, float]]) -> str:
    return "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in keep)


def build_video_filter(plan: CropPlan, out_w: int, out_h: int, fps: int,
                       keep: Optional[Sequence[Tuple[float, float]]] = None,
                       ass_name: Optional[str] = None,
                       fonts_dir: Optional[str] = None) -> str:
    parts: List[str] = [f"fps={fps}",
                        f"crop={plan.cw}:{plan.ch}:x='{plan.x_expr}':y={plan.y}"]
    if keep:
        parts += [f"select='{_between(keep)}'", "setpts=N/FRAME_RATE/TB"]
    parts += [f"scale={out_w}:{out_h}:flags=lanczos", "setsar=1"]
    if ass_name:
        sub = f"subtitles=filename={ass_name}"
        if fonts_dir:
            sub += f":fontsdir={fonts_dir}"
        parts.append(sub)
    parts.append("format=yuv420p")
    return ",".join(parts)


def build_audio_filter(loudnorm: bool, target_lufs: float, limiter: bool,
                       keep: Optional[Sequence[Tuple[float, float]]] = None) -> Optional[str]:
    parts: List[str] = []
    if keep:
        parts += [f"aselect='{_between(keep)}'", "asetpts=N/SR/TB"]
    if loudnorm:
        parts += [f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11", "aresample=48000"]
    if limiter:
        parts.append("alimiter=limit=0.95")
    return ",".join(parts) or None


def build_render_command(src, out, start: float, duration: float, vf: str,
                         af: Optional[str], crf: int, preset: str,
                         audio_bitrate: str, has_audio: bool) -> List[str]:
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
           "-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "0:a:0"]
    cmd += ["-vf", vf]
    if has_audio:
        if af:
            cmd += ["-af", af]
        cmd += ["-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    return cmd


def validate_output(path, width: int, height: int, expected_duration: float,
                    expect_audio: bool, tol: float = 0.75) -> None:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise OutputValidationError(f"output missing or empty: {p}")
    info = video_info(p)
    if (info.width, info.height) != (width, height):
        raise OutputValidationError(
            f"resolution {info.width}x{info.height}, expected {width}x{height}")
    if abs(info.duration - expected_duration) > tol:
        raise OutputValidationError(
            f"duration {info.duration:.2f}s, expected {expected_duration:.2f}s (±{tol}s)")
    if expect_audio and not info.has_audio:
        raise OutputValidationError("audio stream missing in output")