import logging
from pathlib import Path

from src.models.schemas import Scene

log = logging.getLogger(__name__)


def detect_scenes(video_path: Path, duration: float, threshold: float = 27.0) -> list[Scene]:
    """Scene boundaries are one weak signal only; failure is non-fatal."""
    scenes: list[Scene] = []
    try:
        from scenedetect import ContentDetector, detect

        scene_list = detect(str(video_path), ContentDetector(threshold=threshold))
        scenes = [
            Scene(id=i, start=start.get_seconds(), end=end.get_seconds())
            for i, (start, end) in enumerate(scene_list)
        ]
    except Exception as exc:  # noqa: BLE001
        log.warning("Scene detection failed (%s); treating the video as a single scene.", exc)

    if not scenes:
        scenes = [Scene(id=0, start=0.0, end=duration)]
    log.info("Detected %d scenes", len(scenes))
    return scenes