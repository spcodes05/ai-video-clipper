from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import List, Optional, Sequence, Tuple

from .config import FramingCfg


@dataclass(frozen=True)
class CropPlan:
    cw: int
    ch: int
    x_expr: str
    y: int


def effective_size(w: int, h: int, rotation: int) -> Tuple[int, int]:
    return (h, w) if rotation % 360 in (90, 270) else (w, h)


def _even(v: float) -> int:
    v = int(v)
    return v - (v % 2)


def crop_size(src_w: int, src_h: int, out_w: int, out_h: int) -> Tuple[int, int]:
    """Largest crop with the output aspect ratio; never stretches."""
    r = out_w / out_h
    if src_w / src_h > r:
        ch, cw = src_h, src_h * r
    else:
        cw, ch = src_w, src_w / r
    return _even(cw), _even(ch)


def output_size(ch: int, out_w: int, out_h: int, max_upscale: float) -> Tuple[int, int]:
    """Cap upscaling so heavily upscaled sources are not blown up needlessly."""
    f = out_h / ch
    if f > max_upscale:
        oh = _even(ch * max_upscale)
        ow = _even(oh * out_w / out_h)
        return ow, oh
    return out_w, out_h


def smooth_path(raw: Sequence[Tuple[float, Optional[float]]], src_w: int, cw: int,
                cfg: FramingCfg, cuts: Sequence[float] = ()
                ) -> List[Tuple[float, float, bool]]:
    """raw: [(t, subject_cx or None)] -> [(t, crop_left_x, jumped)].
    Deadzone + exponential smoothing + max speed; holds last target when the
    subject is lost; hard-jumps only across scene cuts."""
    lo, hi = cw / 2, src_w - cw / 2
    if not raw:
        return [(0.0, (src_w - cw) / 2, False)]
    first = next((cx for _, cx in raw if cx is not None), src_w / 2)
    pos = min(max(first, lo), hi)
    target = pos
    prev_t: Optional[float] = None
    out: List[Tuple[float, float, bool]] = []
    for t, cx in raw:
        jumped = False
        if cx is not None:
            target = min(max(cx, lo), hi)
        if prev_t is None:
            new = pos
        else:
            dt = max(t - prev_t, 1e-3)
            if any(prev_t < c <= t for c in cuts):
                new = target
                jumped = abs(target - pos) > 1
            elif abs(target - pos) < cfg.deadzone * cw:
                new = pos
            else:
                step = cfg.max_speed * cw * dt
                new = pos + max(-step, min(step, cfg.smoothing * (target - pos)))
        pos, prev_t = new, t
        out.append((t, pos - cw / 2, jumped))
    return out


def thin_keyframes(path: Sequence[Tuple[float, float, bool]], interval: float
                   ) -> List[Tuple[float, float]]:
    kf: List[Tuple[float, float]] = []
    last_t = -1e9

    def add(t: float, x: float) -> None:
        if kf and t <= kf[-1][0]:
            t = kf[-1][0] + 1e-3
        kf.append((t, x))

    for i, (t, x, jumped) in enumerate(path):
        if jumped and i > 0:
            add(t - 1e-3, path[i - 1][1])
            add(t, x)
            last_t = t
        elif i == 0 or i == len(path) - 1 or t - last_t >= interval:
            add(t, x)
            last_t = t
    return kf or [(0.0, 0.0)]


def x_expression(kf: Sequence[Tuple[float, float]]) -> str:
    """Piecewise-linear x(t) for FFmpeg's crop filter (no spaces; wrap in quotes)."""
    if len(kf) == 1 or all(abs(x - kf[0][1]) < 0.01 for _, x in kf):
        return f"{kf[0][1]:.2f}"
    expr = f"{kf[-1][1]:.2f}"
    for (t0, x0), (t1, x1) in reversed(list(zip(kf, kf[1:]))):
        if abs(x1 - x0) < 0.01:
            seg = f"{x0:.2f}"
        else:
            seg = f"{x0:.2f}+({x1 - x0:.2f})*(t-{t0:.3f})/{t1 - t0:.3f}"
        expr = f"if(lt(t,{t1:.3f}),{seg},{expr})"
    return expr


def static_y(cys: Sequence[float], ch: int, src_h: int, headroom: float) -> int:
    if ch >= src_h:
        return 0
    cy = median(cys) if cys else src_h / 2
    return int(min(max(cy - headroom * ch, 0), src_h - ch))