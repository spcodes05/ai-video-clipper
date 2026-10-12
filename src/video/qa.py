import logging
from pathlib import Path

from src.utils.errors import VideoInputError
from src.utils.ffmpeg import probe_video

log = logging.getLogger(__name__)


def validate_clip(
    path: Path, expected_duration: float, width: int, height: int, tolerance: float = 1.0
) -> list[str]:
    """Return a list of problems; an empty list means the clip passed."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return ["output file is missing or empty"]
    try:
        meta = probe_video(p)
    except VideoInputError as exc:
        return [f"output is unreadable: {exc}"]

    issues: list[str] = []
    if (meta.width, meta.height) != (width, height):
        issues.append(f"wrong dimensions {meta.width}x{meta.height}, expected {width}x{height}")
    if not meta.has_audio:
        issues.append("no audio stream")
    if abs(meta.duration - expected_duration) > tolerance:
        issues.append(f"duration {meta.duration:.1f}s differs from expected {expected_duration:.1f}s")
    return issues