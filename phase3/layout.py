"""Canvas / foreground-panel geometry (pure functions, no FFmpeg)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .config import ForegroundCfg, OutputCfg, Phase3ConfigError, parse_ratio


@dataclass(frozen=True)
class Layout:
    out_w: int
    out_h: int
    fg_x: int
    fg_y: int
    fg_w: int
    fg_h: int

    @property
    def fg_bottom(self) -> int:
        return self.fg_y + self.fg_h

    @property
    def fg_ratio(self) -> float:
        return self.fg_w / self.fg_h

    @property
    def is_square(self) -> bool:
        return self.fg_w == self.fg_h


def _even(v: float) -> int:
    v = int(round(v))
    return v - (v % 2)


def compute_layout(output: OutputCfg, fg: ForegroundCfg) -> Layout:
    out_w, out_h = output.dims()
    a, b = parse_ratio(fg.aspect_ratio)
    want_w = fg.width if fg.width > 0 else out_w
    want_w = min(want_w, out_w)
    # largest even width <= want_w for which width*b/a is an even integer (exact ratio, no rounding)
    fg_w = fg_h = 0
    for w in range(want_w - want_w % 2, 15, -2):
        if (w * b) % a == 0:
            h = w * b // a
            if h % 2 == 0:
                fg_w, fg_h = w, h
                break
    if not fg_w:
        raise Phase3ConfigError(f"cannot build an exact {fg.aspect_ratio} foreground panel")
    if fg_h > out_h:                                  # too tall for the canvas: shrink to fit
        for w in range(fg_w, 15, -2):
            if (w * b) % a == 0 and (w * b // a) % 2 == 0 and w * b // a <= out_h:
                fg_w, fg_h = w, w * b // a
                break
        else:
            raise Phase3ConfigError("foreground panel does not fit the output canvas")
    free = out_h - fg_h
    frac = {"center": 0.5, "top": 0.0, "bottom": 1.0, "upper": 0.25, "lower": 0.75}[fg.position]
    y = _even(free * frac) + fg.offset_y
    y = max(0, min(free, y))
    y -= y % 2
    x = _even((out_w - fg_w) / 2)
    return Layout(out_w, out_h, x, y, fg_w, fg_h)


def contain_size(src_w: int, src_h: int, box_w: int, box_h: int) -> Tuple[int, int]:
    """Largest even size with the SOURCE aspect ratio that fits in the box (never stretches)."""
    s = min(box_w / src_w, box_h / src_h)
    w, h = _even(src_w * s), _even(src_h * s)
    return max(2, min(w, box_w)), max(2, min(h, box_h))
