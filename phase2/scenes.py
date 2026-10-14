from __future__ import annotations

import re
from typing import List, Sequence

from .ffmpeg_utils import FFMPEG, run

_PTS = re.compile(r"pts_time:([0-9.]+)")


def detect_scenes(video, threshold: float = 0.35, fps: float = 2.0) -> List[float]:
    """Sparse scene-cut detection (samples at `fps`) via FFmpeg's scene score."""
    vf = f"fps={fps},scale=320:-2,select='gt(scene,{threshold})',showinfo"
    p = run([FFMPEG, "-hide_banner", "-nostats", "-loglevel", "info", "-i", str(video),
             "-an", "-vf", vf, "-f", "null", "-"], "scene-detect")
    return [float(m.group(1)) for m in _PTS.finditer(p.stderr.decode("utf-8", "replace"))]


def cuts_in(cuts: Sequence[float], start: float, end: float) -> List[float]:
    return [c for c in cuts if start < c < end]


def cut_density(cuts: Sequence[float], start: float, end: float) -> float:
    """Cuts per minute inside [start, end]."""
    d = max(end - start, 1e-9)
    return len(cuts_in(cuts, start, end)) * 60.0 / d