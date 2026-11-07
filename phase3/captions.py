"""Animated short-form captions (ASS) + hook overlay, positioned relative to the layout."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from phase2.boundaries import Word
from phase2.captions import _clean, _inline_color, _style_color, _ts, group_words, line_breaks

from .config import CaptionsCfg, HookCfg
from .hooks import Hook
from .layout import Layout
from .zoom import INTENSIFIERS

# style name -> behaviour
STYLE_SPECS = {
    "classic":     dict(bold=0, upper=False, mode="static",    size=0.94, outline=0.75, emph=False, words=4),
    "bold":        dict(bold=1, upper=True,  mode="word",      size=1.00, outline=1.00, emph=True,  words=4),
    "karaoke":     dict(bold=1, upper=True,  mode="karaoke",   size=1.00, outline=1.00, emph=True,  words=4),
    "highlighted": dict(bold=1, upper=True,  mode="pill",      size=1.00, outline=1.00, emph=True,  words=3),
    "minimal":     dict(bold=0, upper=False, mode="static",    size=0.80, outline=0.55, emph=False, words=6),
}
STOP = {"the", "and", "that", "this", "with", "have", "just", "like", "about", "there", "their",
        "would", "could", "should", "because", "really", "going", "thing", "what", "when", "where"}
LINE_H = 1.2


@dataclass(frozen=True)
class Placement:
    font_size: int
    y_top: int
    y_bottom: int
    margin_side: int
    position: str
    fallback: bool
    max_width: int


@dataclass
class CaptionStats:
    groups: int = 0
    events: int = 0
    placement: Optional[Placement] = None
    hook_event: bool = False


def px_scale(layout: Layout) -> float:
    """Pixel-valued settings are authored for a 1080-wide canvas and scaled to the real one."""
    return layout.out_w / 1080.0


def block_height(size: float, lines: int, outline: float) -> int:
    return int(round(lines * size * LINE_H + 2 * outline))


def plan_placement(layout: Layout, cfg: CaptionsCfg) -> Placement:
    """Caption block position: below the square panel by default, inside platform-safe areas."""
    spec = STYLE_SPECS[cfg.style]
    k = px_scale(layout)
    min_size = max(8, int(cfg.min_font_size * k))
    size = max(min_size, int(cfg.font_size * spec["size"] * k))
    outline = cfg.outline * spec["outline"] * k
    gap, safe_top, safe_bottom, margin_side = (int(v * k) for v in (cfg.gap, cfg.safe_top,
                                                                      cfg.safe_bottom, cfg.margin_side))
    lines = cfg.max_lines
    max_w = layout.out_w - 2 * margin_side
    fallback = False
    pos = cfg.position
    bottom_limit = layout.out_h - safe_bottom
    if pos == "below":
        y_top = layout.fg_bottom + gap
        s = size
        while block_height(s, lines, outline) > bottom_limit - y_top and s > min_size:
            s -= 2
        if block_height(s, lines, outline) > bottom_limit - y_top:
            pos, fallback = "over_panel", True
        else:
            size = s
    if pos == "over_panel":
        h = block_height(size, lines, outline)
        y_bottom = layout.fg_bottom - gap
        y_top = max(0, y_bottom - h)
    elif pos == "above":
        h = block_height(size, lines, outline)
        y_top = max(safe_top, layout.fg_y - gap - h)
    elif pos == "custom":
        y_top = int((cfg.y_ratio or 0.8) * layout.out_h)
    h = block_height(size, lines, outline)
    if pos == "below":
        y_top = layout.fg_bottom + gap
    y_top = max(0, min(int(y_top), layout.out_h - h))
    return Placement(size, int(y_top), int(y_top + h), margin_side, pos, fallback, max_w)


def fit_size(size: int, longest_chars: int, upper: bool, max_width: int, min_size: int) -> int:
    per = 0.62 if upper else 0.54
    need = longest_chars * size * per
    if need <= max_width:
        return size
    return max(min_size, int(size * max_width / need))


def pick_emphasis(texts: Sequence[str]) -> Optional[int]:
    best, best_score = None, 0.0
    for i, t in enumerate(texts):
        core = re.sub(r"[^\w']", "", t)
        low = core.lower()
        if len(core) < 3 or low in STOP:
            continue
        sc = 0.0
        sc += 2.0 if any(c.isdigit() for c in core) else 0.0
        sc += 1.5 if (core.isupper() and len(core) >= 3) else 0.0
        sc += 1.5 if t.endswith("!") else 0.0
        sc += 1.5 if low in INTENSIFIERS else 0.0
        sc += 0.1 * len(core) if len(core) >= 6 else 0.0
        if sc > best_score:
            best, best_score = i, sc
    return best if best_score >= 1.5 else None


def _style_line(name: str, cfg: CaptionsCfg, size: int, bold: int, outline: float,
                align: int, margin_v: int, margin_side: int, primary: str, secondary: str,
                outline_col: str, border_style: int = 1, shadow: Optional[int] = None) -> str:
    font = cfg.font.replace(",", " ")
    sh = cfg.shadow if shadow is None else shadow
    return (f"Style: {name},{font},{size},{_style_color(primary)},{_style_color(secondary)},"
            f"{_style_color(outline_col)},&H64000000,{-1 if bold else 0},0,0,0,100,100,0,0,"
            f"{border_style},{outline:.0f},{sh},{align},{margin_side},{margin_side},{margin_v},1")


def _hook_layout(layout: Layout, cfg: CaptionsCfg, hcfg: HookCfg, text: str) -> Tuple[List[str], int, int]:
    words = text.split()
    lines_idx = line_breaks(words, 16, 3)
    lines = [" ".join(words[i] for i in ln) for ln in lines_idx if ln]
    longest = max(len(l) for l in lines)
    k = px_scale(layout)
    size = fit_size(int(hcfg.font_size * k), longest + 2, True,
                    layout.out_w - 2 * int((cfg.margin_side + 20) * k), max(8, int(30 * k)))
    pad = 14 * k
    h = block_height(size, len(lines), pad)
    if hcfg.y_ratio is not None:
        y = int(hcfg.y_ratio * layout.out_h)
    else:
        top, bot = int(cfg.safe_top * k), layout.fg_y - int(24 * k)
        y = top + max(0, (bot - top - h) // 2) if bot - top >= h else top
    return lines, size, max(0, min(y, layout.out_h - h))


def build_ass(words: Sequence[Word], cfg: CaptionsCfg, layout: Layout, placement: Placement,
              hook: Optional[Hook] = None, hcfg: Optional[HookCfg] = None,
              hook_end: Optional[float] = None, plain: bool = False) -> Tuple[str, CaptionStats]:
    """`words` are OUTPUT-timeline words. `plain=True` is the safe fallback (static, no effects)."""
    st = CaptionStats(placement=placement)
    name = "classic" if plain else cfg.style
    spec = STYLE_SPECS[name]
    mode = spec["mode"]
    upper = spec["upper"]
    k = px_scale(layout)
    outline = cfg.outline * spec["outline"] * k
    min_size = max(8, int(cfg.min_font_size * k))
    pri, hl, em = _inline_color(cfg.primary), _inline_color(cfg.highlight), _inline_color(cfg.emphasis_color)
    gcfg = type("G", (), dict(max_words=min(cfg.max_words, spec["words"]) if cfg.max_words else spec["words"],
                              max_chars=cfg.max_chars, max_lines=cfg.max_lines, max_gap=cfg.max_gap))()
    if name == "minimal":
        gcfg.max_words = max(cfg.max_words, spec["words"])
    sty = [_style_line("Default", cfg, placement.font_size, spec["bold"], outline, 8, placement.y_top,
                       placement.margin_side,
                       cfg.highlight if mode == "karaoke" else cfg.primary,
                       cfg.primary, cfg.outline_color, shadow=int(round(cfg.shadow * k)))]
    events: List[str] = []

    groups = group_words(list(words), gcfg) if cfg.enabled else []
    st.groups = len(groups)
    for gi, g in enumerate(groups):
        texts = [_clean(w.text).upper() if upper else _clean(w.text) for w in g]
        lines = line_breaks(texts, cfg.max_chars, cfg.max_lines)
        longest = max(len(" ".join(texts[i] for i in ln)) for ln in lines)
        size = fit_size(placement.font_size, longest, upper, placement.max_width,
                        min_size) if cfg.dynamic_size else placement.font_size
        size_tag = f"\\fs{size}" if size != placement.font_size else ""
        emph = pick_emphasis(texts) if (cfg.emphasis and spec["emph"] and not plain) else None
        next_start = groups[gi + 1][0].start if gi + 1 < len(groups) else None

        def word_markup(i: int, active: int, kara_cs: Optional[int] = None) -> str:
            t = texts[i]
            tags = ""
            if mode == "karaoke":
                tags = f"\\kf{kara_cs}"
            elif mode == "word":
                tags = f"\\c{hl if i == active else (em if i == emph else pri)}"
            elif mode == "pill":
                if i == active:
                    tags = (f"\\1c&H000000&\\3c{hl}\\bord{outline + 5:.0f}\\c&H000000&")
                else:
                    tags = f"\\c{em if i == emph else pri}"
            if i == emph and mode != "static":
                tags += "\\fscx112\\fscy112"
            out = f"{{{tags}}}{t}" if tags else t
            if i == emph and mode != "static":
                out += "{\\fscx100\\fscy100}"
            if mode == "pill" and i == active:
                out += f"{{\\bord{outline:.0f}\\3c{_inline_color(cfg.outline_color)}}}"
            return out

        def render(active: int, kara: Optional[List[int]] = None) -> str:
            rows = []
            for ln in lines:
                rows.append(" ".join(word_markup(i, active, kara[i] if kara else None) for i in ln))
            return (f"{{{size_tag}}}" if size_tag else "") + "\\N".join(rows)

        if mode in ("static",):
            s0 = g[0].start
            e0 = g[-1].end + 0.12
            if next_start is not None:
                e0 = min(e0, next_start)
            if e0 > s0:
                events.append((s0, e0, render(-1)))
        elif mode == "karaoke":
            s0 = g[0].start
            e0 = g[-1].end + 0.12
            if next_start is not None:
                e0 = min(e0, next_start)
            if e0 > s0:
                edges = [w.start for w in g] + [e0]
                cs = []
                acc = 0
                for k in range(len(g)):
                    tgt = int(round((edges[k + 1] - s0) * 100))
                    cs.append(max(1, tgt - acc))
                    acc += cs[-1]
                events.append((s0, e0, render(-1, cs)))
        else:
            for i, w in enumerate(g):
                s0 = w.start
                e0 = g[i + 1].start if i + 1 < len(g) else w.end + 0.12
                e0 = max(e0, s0 + cfg.min_word_seconds)
                if next_start is not None:
                    e0 = min(e0, next_start)
                if e0 > s0:
                    events.append((s0, e0, render(i)))
    cap_events = [f"Dialogue: 0,{_ts(a)},{_ts(b)},Default,,0,0,0,,{t}" for a, b, t in events]
    st.events = len(cap_events)

    hook_events: List[str] = []
    if hook is not None and hcfg is not None and hcfg.enabled and not plain:
        lines, hsize, hy = _hook_layout(layout, cfg, hcfg, hook.text)
        hpad = 14 * k
        end = min(hcfg.duration, hook_end) if hook_end else hcfg.duration
        sty.append(_style_line("Hook", cfg, hsize, 1, hpad, 8, hy, placement.margin_side + int(20 * k),
                               hcfg.text_color, hcfg.text_color, hcfg.box_color, border_style=3, shadow=0))
        fi, fo = int(hcfg.fade_in * 1000), int(hcfg.fade_out * 1000)
        body = "\\N".join(_clean(l) for l in lines)
        hook_events.append(f"Dialogue: 1,{_ts(0.05)},{_ts(end)},Hook,,0,0,0,,"
                           f"{{\\fad({fi},{fo})\\fscx92\\fscy92\\t(0,{fi},\\fscx100\\fscy100)}}{body}")
        st.hook_event = True

    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {layout.out_w}\nPlayResY: {layout.out_h}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        + "\n".join(sty) + "\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    return header + "\n".join(hook_events + cap_events) + "\n", st


def caption_box_inside(p: Placement, layout: Layout, cfg: CaptionsCfg) -> List[str]:
    """Static geometry problems (empty list = OK)."""
    issues = []
    if p.y_top < 0 or p.y_bottom > layout.out_h:
        issues.append(f"caption block y {p.y_top}-{p.y_bottom} outside frame height {layout.out_h}")
    if p.margin_side * 2 >= layout.out_w:
        issues.append("caption side margins leave no width")
    if p.position == "below" and p.y_bottom > layout.out_h - int(cfg.safe_bottom * px_scale(layout)) + 1:
        issues.append("caption block enters the bottom platform-safe area")
    return issues
