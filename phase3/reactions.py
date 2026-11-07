"""Group reaction shots.

Evidence used (all measured, none guessed):
  * audio: a detected laugh/reaction window (existing laughter detector);
  * picture: several faces SMILING together (OpenCV smile cascade) or several faces visibly
    animated (frame-difference inside each face box) for a sustained moment.
These are smile/movement proxies, NOT emotion recognition: surprise, shock, gaze direction and
gestures are not recognised, and the report says which evidence triggered every window.
When a window is found while the crop shows one person, the edit cuts to the whole group."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from phase2.faces import FaceBox

from .config import ReactionCfg
from .timeline import Seg

log = logging.getLogger("phase3.reactions")
LaughWindow = Tuple[float, float, float]      # source seconds + score 0..1


@dataclass(frozen=True)
class ExprSample:
    t: float            # clip-relative seconds
    faces: int          # faces analysed (large enough to judge)
    smiling: int
    animated: int


def merge_windows(wins: Sequence[Seg], gap: float = 0.3) -> List[Seg]:
    out: List[Seg] = []
    for a, b in sorted(wins):
        if out and a <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def plan_reaction_windows(laughter: Sequence[LaughWindow], samples: Sequence[Tuple[float, Sequence[FaceBox]]],
                          clip_start: float, clip_end: float, cfg: ReactionCfg) -> List[Seg]:
    """Audio-triggered windows (clip-relative). Requires >= min_faces in >= min_face_ratio of the
    samples inside the window, so we never cut to a "group" that is not in the picture."""
    if not cfg.enabled or not samples:
        return []
    out: List[Seg] = []
    for a, b, sc in laughter:
        if sc < cfg.min_laughter_score or b <= clip_start or a >= clip_end:
            continue
        s = max(a, clip_start) - clip_start
        e = min(min(b, clip_end) - clip_start, s + cfg.max_window)
        if e - s < cfg.min_window:
            continue
        inside = [boxes for t, boxes in samples if s - 0.25 <= t <= e + 0.25]
        if inside and sum(1 for bx in inside if len(bx) >= cfg.min_faces) / len(inside) >= cfg.min_face_ratio:
            out.append((round(s, 3), round(e, 3)))
    return merge_windows(out)


def plan_visual_windows(expr: Sequence[ExprSample], interval: float, cfg: ReactionCfg) -> List[Seg]:
    """Windows where >= min_smile_faces smile together, or most faces move together, for at
    least min_window seconds (one missed sample is tolerated)."""
    if not cfg.enabled or not cfg.visual or not expr:
        return []
    flags = []
    for e in expr:
        smile = e.smiling >= cfg.min_smile_faces
        group_motion = e.faces >= 2 and e.animated >= max(2, int(0.6 * e.faces + 0.5))
        flags.append(smile or group_motion)
    wins: List[Seg] = []
    i = 0
    while i < len(expr):
        if not flags[i]:
            i += 1
            continue
        j, last = i, i
        while j + 1 < len(expr) and (flags[j + 1] or (j + 2 < len(expr) and flags[j + 2])):
            j += 1
            if flags[j]:
                last = j
        hit = sum(flags[i:last + 1]) / (last - i + 1)
        start, end = expr[i].t, expr[last].t + interval
        if end - start >= cfg.min_window and hit >= cfg.smile_ratio:
            wins.append((round(start, 3), round(min(end, start + cfg.max_window), 3)))
        i = last + 1
    return merge_windows(wins)


class ExpressionUnavailable(RuntimeError):
    pass


def analyze_expressions(video, start: float, samples: Sequence[Tuple[float, Sequence[FaceBox]]],
                        cfg: ReactionCfg, min_face_px: int = 56, max_samples: int = 160) -> List[ExprSample]:
    """Smile + motion proxies for samples that show >= 2 faces (the only ones a group shot needs)."""
    try:
        import cv2
    except ImportError as exc:
        raise ExpressionUnavailable("opencv is not installed") from exc
    if not hasattr(cv2, "CascadeClassifier"):
        raise ExpressionUnavailable("this OpenCV build has no CascadeClassifier")
    smile_cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    if smile_cc.empty():
        raise ExpressionUnavailable("smile cascade missing from OpenCV data")
    cand = [(t, b) for t, b in samples if len(b) >= 2][:max_samples]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ExpressionUnavailable("cannot open video")
    out: List[ExprSample] = []
    try:
        for t, boxes in cand:
            frames = []
            for dt in (0.0, 0.2):
                cap.set(cv2.CAP_PROP_POS_MSEC, (start + t + dt) * 1000)
                ok, f = cap.read()
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if ok else None)
            if frames[0] is None:
                continue
            g0, g1 = frames
            faces = smiling = animated = 0
            for b in boxes:
                if b.w < min_face_px:
                    continue
                x0, y0 = max(0, int(b.x)), max(0, int(b.y))
                x1, y1 = int(b.x + b.w), int(b.y + b.h)
                roi = g0[y0:y1, x0:x1]
                if roi.size == 0:
                    continue
                faces += 1
                low = roi[int(0.5 * roi.shape[0]):, :]
                sm = smile_cc.detectMultiScale(low, scaleFactor=1.6, minNeighbors=22,
                                               minSize=(max(20, b.w // 6), max(10, b.w // 12)))
                smiling += 1 if len(sm) else 0
                if g1 is not None:
                    r1 = g1[y0:y1, x0:x1]
                    if r1.shape == roi.shape and float(cv2.absdiff(roi, r1).mean()) > cfg.motion_threshold:
                        animated += 1
            out.append(ExprSample(t, faces, smiling, animated))
    finally:
        cap.release()
    return out
