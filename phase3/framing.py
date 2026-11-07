"""AUTO / CROP / CONTAIN decision for the square foreground panel + group-aware crop plan."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import median
from typing import List, Optional, Sequence, Tuple

from phase2.crop import (CropPlan, crop_size, reduce_keyframes, smooth_path, static_y,
                         thin_keyframes, x_expression)
from phase2.faces import FaceBox
from phase2.tracking import track_subject

from .config import ForegroundCfg, GroupCfg, FramingCfg

Samples = Sequence[Tuple[float, Sequence[FaceBox]]]
log = logging.getLogger("phase3.framing")


@dataclass
class FaceStats:
    samples: int = 0
    presence: float = 0.0        # fraction of samples with >= 1 face
    p90_faces: int = 0           # 90th percentile simultaneous faces
    max_faces: int = 0
    union_w_p90: float = 0.0     # 90th percentile width of the union of all faces (source px)
    union_cx: Optional[float] = None


@dataclass
class FramingDecision:
    mode: str                    # crop | contain
    reason: str
    stats: FaceStats = field(default_factory=FaceStats)
    group_crop: bool = False     # crop centred on the group instead of tracking one face


def _pct(vals: Sequence[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[k])


def face_stats(samples: Samples) -> FaceStats:
    if not samples:
        return FaceStats()
    counts = [len(b) for _, b in samples]
    unions, centers = [], []
    for _, boxes in samples:
        if boxes:
            x0 = min(b.x for b in boxes)
            x1 = max(b.x + b.w for b in boxes)
            unions.append(x1 - x0)
            centers.append((x0 + x1) / 2)
    return FaceStats(
        samples=len(samples),
        presence=sum(1 for c in counts if c) / len(counts),
        p90_faces=int(_pct(counts, 0.9)),
        max_faces=max(counts),
        union_w_p90=_pct(unions, 0.9),
        union_cx=median(centers) if centers else None,
    )


def decide_mode(fg: ForegroundCfg, group: GroupCfg, samples: Samples, sw: int, sh: int,
                cw: int, panel_ratio: float, faces_available: bool) -> FramingDecision:
    """Pick crop vs contain for one clip.

    AUTO is deliberately conservative: a crop is only chosen when the detected faces
    provably fit in it. Anything uncertain (no face data, weak detections, 3+ people)
    falls back to CONTAIN so nobody is cut out of the frame."""
    st = face_stats(samples)
    if fg.fit_mode == "contain":
        return FramingDecision("contain", "fit_mode=contain", st)
    if fg.fit_mode == "crop":
        return FramingDecision("crop", "fit_mode=crop", st,
                               group_crop=st.p90_faces >= 2 and st.union_cx is not None)
    # ---- auto
    if abs(sw / sh - panel_ratio) < 0.02:
        return FramingDecision("crop", "source already matches the panel ratio", st)
    if sw / sh < panel_ratio:
        # taller than the panel: a crop only trims top/bottom of a portrait source
        return FramingDecision("crop", "portrait source; vertical trim only", st)
    if not faces_available or st.samples == 0:
        if group.contain_when_unsure:
            return FramingDecision("contain", "no face data available", st)
        return FramingDecision("crop", "no face data; centre crop (contain_when_unsure=false)", st)
    if st.presence < group.min_face_ratio:
        return FramingDecision("contain", f"faces found in only {st.presence:.0%} of samples", st)
    if st.p90_faces >= group.group_min_faces:
        return FramingDecision("contain", f"group shot ({st.p90_faces} faces)", st)
    if st.p90_faces == 2:
        if st.union_w_p90 * group.union_margin <= cw:
            return FramingDecision("crop", "two faces fit in the crop", st, group_crop=True)
        return FramingDecision("contain", "two faces too far apart for a square crop", st)
    return FramingDecision("crop", "single speaker", st)


@dataclass
class CropSolution:
    plan: CropPlan
    keyframes: List[Tuple[float, float]]   # (clip-time t, crop left x)
    used_faces: bool


def plan_crop(samples: Samples, sw: int, sh: int, panel_w: int, panel_h: int,
              fcfg: FramingCfg, speech: Sequence[Tuple[float, float]],
              cuts: Sequence[float], group_cx: Optional[float] = None) -> CropSolution:
    """Smooth crop path. Reuses phase2 tracking/smoothing; for a group the crop is
    centred on the group (static, no hunting between speakers)."""
    cw, ch = crop_size(sw, sh, panel_w, panel_h)
    raw: List[Tuple[float, Optional[float]]] = []
    cys: List[float] = []
    used = False
    if samples:
        if group_cx is not None:
            raw = [(t, group_cx if boxes else None) for t, boxes in samples]
            cys = [b.y + b.h / 2 for _, boxes in samples for b in boxes]
            used = True
        else:
            subj = track_subject(list(samples), sw, fcfg, speech, cuts)
            raw = [(s.t, s.cx) for s in subj]
            cys = [s.cy for s in subj if s.cy is not None]
            used = any(s.cx is not None for s in subj)
    path = smooth_path(raw, sw, cw, fcfg, cuts)
    kf = reduce_keyframes(thin_keyframes(path, fcfg.keyframe_interval), 0.01 * cw)
    y = static_y(cys, ch, sh, fcfg.headroom)
    return CropSolution(CropPlan(cw, ch, x_expression(kf), y), kf, used)


def crop_x_at(kf: Sequence[Tuple[float, float]], t: float) -> float:
    """Piecewise-linear evaluation matching phase2.crop.x_expression."""
    if not kf:
        return 0.0
    if t <= kf[0][0]:
        return kf[0][1]
    for (t0, x0), (t1, x1) in zip(kf, kf[1:]):
        if t < t1:
            return x0 + (x1 - x0) * ((t - t0) / (t1 - t0) if t1 > t0 else 0.0)
    return kf[-1][1]
