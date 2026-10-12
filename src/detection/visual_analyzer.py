import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_FPS = 1.0
THUMB = (64, 36)


class VisualAnalyzer:
    """Cheap motion signal: mean frame difference on tiny grayscale thumbnails, ~1 sample/second."""

    def __init__(self, motion: np.ndarray, step: float = 1.0):
        self.motion = np.asarray(motion, dtype=np.float32)
        self.step = step if step > 0 else 1.0
        self.ref = max(float(np.percentile(self.motion, 95)), 1e-6) if self.motion.size else 1.0

    @classmethod
    def from_video(cls, path: Path) -> "VisualAnalyzer":
        try:
            import cv2
        except ImportError:
            log.warning("OpenCV not available; visual motion disabled.")
            return cls(np.array([], dtype=np.float32))

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            log.warning("OpenCV could not open %s; visual motion disabled.", path)
            return cls(np.array([], dtype=np.float32))

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(int(round(fps / SAMPLE_FPS)), 1)
        diffs: list[float] = []
        prev = None
        idx = 0
        try:
            while cap.grab():
                if idx % stride == 0:
                    ok, frame = cap.retrieve()
                    if ok:
                        gray = cv2.cvtColor(cv2.resize(frame, THUMB), cv2.COLOR_BGR2GRAY).astype(np.float32)
                        diffs.append(0.0 if prev is None else float(np.abs(gray - prev).mean()))
                        prev = gray
                idx += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("Visual analysis stopped early: %s", exc)
        finally:
            cap.release()
        log.info("Visual analysis: %d samples", len(diffs))
        return cls(np.array(diffs, dtype=np.float32), stride / fps)

    def score(self, start: float, end: float) -> float | None:
        """Relative motion 0..1, or None when unavailable."""
        if self.motion.size == 0:
            return None
        a = max(0, int(start / self.step))
        b = max(a + 1, int(end / self.step))
        seg = self.motion[a:b]
        if seg.size == 0:
            return None
        return float(min(seg.mean() / self.ref, 1.0))