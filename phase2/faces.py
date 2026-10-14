from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


class DetectorUnavailable(RuntimeError):
    """Face detection cannot run (missing dependency or unreadable video)."""


@dataclass(frozen=True)
class FaceBox:
    t: float
    x: float
    y: float
    w: float
    h: float
    conf: float = 1.0
    mouth: float = 0.0  # mouth-region motion proxy, NOT verified speaking


class HaarFaceDetector:
    """OpenCV Haar cascade (ships with opencv-python, CPU only).
    `mouth` is a lower-face motion proxy, not true active-speaker recognition."""

    def __init__(self, min_confidence: float = 2.0, min_neighbors: int = 5):
        try:
            import cv2
            import numpy as np
        except ImportError as e:
            raise DetectorUnavailable(
                "opencv-python/numpy not installed: pip install opencv-python numpy") from e
        self.cv2 = cv2
        self.np = np
        self.min_confidence = min_confidence
        self.min_neighbors = min_neighbors
        self._cc = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self._cc.empty():
            raise DetectorUnavailable("could not load Haar cascade from OpenCV data")

    def detect(self, frame, t: float, analysis_width: int = 640, next_frame=None) -> List[FaceBox]:
        cv2, np = self.cv2, self.np
        h, w = frame.shape[:2]
        scale = analysis_width / w if w > analysis_width else 1.0

        def prep(f):
            s = cv2.resize(f, (int(w * scale), int(h * scale))) if scale < 1.0 else f
            return cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)

        g1 = prep(frame)
        g2 = prep(next_frame) if next_frame is not None else None
        rects, _, weights = self._cc.detectMultiScale3(
            g1, scaleFactor=1.1, minNeighbors=self.min_neighbors,
            minSize=(30, 30), outputRejectLevels=True)
        out: List[FaceBox] = []
        for (x, y, bw, bh), wt in zip(rects, weights):
            conf = float(np.asarray(wt).ravel()[0])
            if conf < self.min_confidence:
                continue
            mouth = 0.0
            if g2 is not None and g2.shape == g1.shape:
                my0, my1 = int(y + 0.6 * bh), int(y + bh)
                mx0, mx1 = int(x + 0.2 * bw), int(x + 0.8 * bw)
                ry0, ry1 = int(y), int(y + 0.4 * bh)
                m = cv2.absdiff(g1[my0:my1, mx0:mx1], g2[my0:my1, mx0:mx1])
                r = cv2.absdiff(g1[ry0:ry1, mx0:mx1], g2[ry0:ry1, mx0:mx1])
                if m.size and r.size:
                    mouth = float(min(5.0, m.mean() / (r.mean() + 2.0)))
            out.append(FaceBox(t, x / scale, y / scale, bw / scale, bh / scale, conf, mouth))
        return out


def sample_faces(video, start: float, end: float, interval: float,
                 detector: HaarFaceDetector, analysis_width: int = 640
                 ) -> List[Tuple[float, List[FaceBox]]]:
    """Returns [(t_relative_to_start, boxes)] sampled every `interval` seconds."""
    cv2 = detector.cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise DetectorUnavailable(
            f"OpenCV cannot open {video} (non-ASCII path or unsupported codec?)")
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass
    out: List[Tuple[float, List[FaceBox]]] = []
    try:
        n = int((end - start) / interval) + 1
        for i in range(n):
            t = i * interval
            if start + t > end:
                break
            cap.set(cv2.CAP_PROP_POS_MSEC, (start + t) * 1000.0)
            ok, f1 = cap.read()
            if not ok:
                out.append((t, []))
                continue
            ok2, f2 = cap.read()
            out.append((t, detector.detect(f1, t, analysis_width, f2 if ok2 else None)))
    finally:
        cap.release()
    return out


def face_presence_ratio(samples: Sequence[Tuple[float, Sequence[FaceBox]]]) -> float:
    if not samples:
        return 0.0
    return sum(1 for _, b in samples if b) / len(samples)