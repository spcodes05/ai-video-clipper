"""Short-form captions (ASS): phrase-aware grouping, smooth active-word highlight, clean white hook.

Everything is positioned from the actual Layout, so moving the foreground panel moves the caption
block and the hook with it. Pixel-valued settings are authored for 1080 px wide and scale."""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from phase2.boundaries import Word, is_sentence_end
from phase2.captions import _ts

from .config import CaptionsCfg, HookCfg
from .hooks import Hook, apply_case
from .layout import Layout
from .zoom import INTENSIFIERS

# mode: static (whole phrase), tween (smooth colour sweep per word), pill (tween + highlight pill),
#       karaoke (\kf fill)
STYLE_SPECS = {
    "classic":     dict(bold=0, mode="static",  size=0.94, outline=0.75, emph=False, words=7, shadow=1.0),
    "bold":        dict(bold=1, mode="tween",   size=1.00, outline=1.00, emph=True,  words=6, shadow=1.0),
    "karaoke":     dict(bold=1, mode="karaoke", size=1.00, outline=1.00, emph=True,  words=6, shadow=1.0),
    "highlighted": dict(bold=1, mode="pill",    size=1.00, outline=1.00, emph=True,  words=5, shadow=0.0),
    "minimal":     dict(bold=0, mode="static",  size=0.80, outline=0.55, emph=False, words=8, shadow=1.0),
}
STOP = {"the", "and", "that", "this", "with", "have", "just", "like", "about", "there", "their",
        "would", "could", "should", "because", "really", "going", "thing", "what", "when", "where"}
# a phrase should not END on these (it reads as cut mid-thought)
WEAK_END = {"a", "an", "the", "to", "of", "in", "on", "at", "for", "with", "and", "but", "or", "so", "if",
            "my", "your", "his", "her", "their", "our", "is", "was", "are", "were", "be", "been", "i",
            "he", "she", "we", "they", "going", "gonna", "can", "could", "would", "should", "will", "do",
            "did", "does", "have", "has", "had", "just", "very", "really", "as", "than", "because",
            "from", "by", "about", "into", "not", "what", "who", "when", "where", "how", "why", "i'm",
            "he's", "she's", "it's", "that's", "don't", "didn't", "can't", "won't"}
LINE_H = 1.2
_PUNCT_END = re.compile(r"[.,;:]+$")
_NORM = re.compile(r"[^\w']")


@dataclass(frozen=True)
class Placement:
    font_size: int
    y_top: int
    y_bottom: int
    margin_side: int
    position: str
    fallback: bool
    max_width: int


@dataclass(frozen=True)
class HookPlacement:
    lines: Tuple[str, ...]
    font_size: int
    y_top: int
    y_bottom: int
    pad: float
    over_panel: bool
    text: str


@dataclass
class CaptionStats:
    groups: int = 0
    events: int = 0
    placement: Optional[Placement] = None
    hook_event: bool = False
    group_texts: Tuple[str, ...] = ()


def px_scale(layout: Layout) -> float:
    return layout.out_w / 1080.0


def block_height(size: float, lines: int, outline: float) -> int:
    return int(round(lines * size * LINE_H + 2 * outline))


def ass_color(hexrgb: str, alpha: int = 0) -> str:
    r, g, b = hexrgb[0:2], hexrgb[2:4], hexrgb[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def _inline(hexrgb: str) -> str:
    return f"&H{hexrgb[4:6]}{hexrgb[2:4]}{hexrgb[0:2]}&".upper()


# ------------------------------------------------------------------ text helpers
def display_text(raw: str, cfg: CaptionsCfg) -> str:
    t = _PUNCT_END.sub("", raw.strip().replace("{", "").replace("}", "").replace("\\", ""))
    if cfg.text_case == "upper":
        return t.upper()
    return t


def balanced_lines(texts: Sequence[str], max_chars: int, max_lines: int,
                   strict: bool = False) -> Optional[List[List[int]]]:
    """Split word indices into <= max_lines lines of similar length, preferring breaks after
    punctuation and avoiding breaks after weak words ("going | to", "he | was")."""
    n = len(texts)
    lens = [len(t) for t in texts]

    def line_len(a: int, b: int) -> int:
        return sum(lens[a:b]) + (b - a - 1)

    if n == 0:
        return []
    if line_len(0, n) <= max_chars or max_lines == 1:
        ok = line_len(0, n) <= max_chars
        return [list(range(n))] if (ok or not strict) else None
    best, best_cost = None, 1e9
    for k in range(2, min(max_lines, n) + 1):
        for cuts in itertools.combinations(range(1, n), k - 1):
            bounds = (0,) + cuts + (n,)
            ls = [line_len(bounds[i], bounds[i + 1]) for i in range(k)]
            over = sum(max(0, l - max_chars) for l in ls)
            cost = over * 10 + (max(ls) - min(ls)) * 0.5 + (k - 2) * 4
            for c in cuts:
                prev = _NORM.sub("", texts[c - 1].lower())
                if prev in WEAK_END:
                    cost += 6
                if texts[c - 1].endswith((",", ";", ":")) or texts[c - 1].endswith(("?", "!")):
                    cost -= 3
            if cost < best_cost:
                best, best_cost = bounds, cost
        if best is not None and best_cost < 6 and k == 2:
            break
    if best is None:
        return None
    lines = [list(range(best[i], best[i + 1])) for i in range(len(best) - 1)]
    if strict and any(line_len(l[0], l[-1] + 1) > max_chars for l in lines):
        return None
    return lines


def group_phrases(words: Sequence[Word], cfg: CaptionsCfg, max_words: Optional[int] = None) -> List[List[Word]]:
    """Group words into on-screen phrases without breaking thoughts apart.

    Hard boundaries: sentence ends and long pauses. Inside a unit a small DP picks the breaks with
    the lowest cost (never ending on a weak word like "to"/"was" unless forced, preferring commas)."""
    mw = max_words or cfg.max_words
    max_total = cfg.max_chars * cfg.max_lines
    units: List[List[Word]] = []
    cur: List[Word] = []
    for w in sorted(words, key=lambda w: w.start):
        if cur and w.start - cur[-1].end > cfg.max_gap:
            units.append(cur)
            cur = []
        cur.append(w)
        if is_sentence_end(w):
            units.append(cur)
            cur = []
    if cur:
        units.append(cur)

    groups: List[List[Word]] = []
    for u in units:
        n = len(u)
        txt = [display_text(w.text, cfg) for w in u]

        def feasible(a: int, b: int) -> bool:
            if b - a == 1:
                return True
            if b - a > mw or sum(len(t) for t in txt[a:b]) + (b - a - 1) > max_total:
                return False
            return balanced_lines(txt[a:b], cfg.max_chars, cfg.max_lines, strict=True) is not None

        INF = 1e9
        dp = [INF] * (n + 1)
        prev = [0] * (n + 1)
        dp[0] = 0.0
        for i in range(1, n + 1):
            for j in range(max(0, i - mw), i):
                if dp[j] >= INF or not feasible(j, i):
                    continue
                size = i - j
                last = _NORM.sub("", txt[i - 1].lower())
                cost = 1.0                                         # fewer groups is calmer
                if i < n and last in WEAK_END:
                    cost += 3.0
                if i < n and txt[i - 1] != u[i - 1].text.strip() and u[i - 1].text.strip().endswith(","):
                    cost -= 1.0
                if size == 1 and n > 1:
                    cost += 1.5
                cost += 0.15 * abs(size - min(mw, (n + 1) // 2 if n <= mw * 2 else mw))
                if dp[j] + cost < dp[i]:
                    dp[i], prev[i] = dp[j] + cost, j
        if dp[n] >= INF:                                           # should not happen: fall back
            groups.extend([[w] for w in u])
            continue
        cuts, i = [], n
        while i > 0:
            cuts.append((prev[i], i))
            i = prev[i]
        for a, b in reversed(cuts):
            groups.append(u[a:b])
    return groups


def pick_emphasis(texts: Sequence[str]) -> Optional[int]:
    best, best_score = None, 0.0
    for i, t in enumerate(texts):
        core = re.sub(r"[^\w']", "", t)
        low = core.lower()
        if len(core) < 4 or low in STOP or low in WEAK_END:
            continue
        sc = 0.0
        sc += 2.0 if any(c.isdigit() for c in core) else 0.0
        sc += 1.5 if (core.isupper() and len(core) >= 3) else 0.0
        sc += 1.5 if t.endswith("!") else 0.0
        sc += 1.5 if low in INTENSIFIERS else 0.0
        sc += 0.1 * len(core) if len(core) >= 7 else 0.0
        if sc > best_score:
            best, best_score = i, sc
    return best if best_score >= 1.5 else None


def fit_size(size: int, longest_chars: int, bold: bool, max_width: int, min_size: int) -> int:
    per = 0.56 if bold else 0.5
    need = longest_chars * size * per
    return size if need <= max_width else max(min_size, int(size * max_width / need))


# ------------------------------------------------------------------ placement
def plan_placement(layout: Layout, cfg: CaptionsCfg) -> Placement:
    """Caption block: below the panel by default, inside the platform-safe area."""
    spec = STYLE_SPECS[cfg.style]
    k = px_scale(layout)
    min_size = max(8, int(cfg.min_font_size * k))
    size = max(min_size, int(cfg.font_size * spec["size"] * k))
    outline = cfg.outline * spec["outline"] * k
    gap, safe_top, safe_bottom, margin_side = (int(v * k) for v in (cfg.gap, cfg.safe_top,
                                                                      cfg.safe_bottom, cfg.margin_side))
    lines = cfg.max_lines
    max_w = layout.out_w - 2 * margin_side
    fallback, pos = False, cfg.position
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
    h = block_height(size, lines, outline)
    if pos == "over_panel":
        y_top = max(0, layout.fg_bottom - gap - h)
    elif pos == "above":
        y_top = max(safe_top, layout.fg_y - gap - h)
    elif pos == "custom":
        y_top = int((cfg.y_ratio or 0.8) * layout.out_h)
    else:
        y_top = layout.fg_bottom + gap
    y_top = max(0, min(int(y_top), layout.out_h - h))
    return Placement(size, int(y_top), int(y_top + h), margin_side, pos, fallback, max_w)


# ------------------------------------------------------------------ hook
def hook_style(h: HookCfg) -> dict:
    """Resolved look of a hook style. No style uses a solid colour block; the pill is a translucent
    dark rounded-feel backing only for subtle_pill / creator_custom(background: pill)."""
    if h.style == "minimal_shadow":
        return dict(outline=0.0, shadow=3.5, pill=False, opacity=0.0, blur=1.0, color=h.text_color,
                    outline_color=h.outline_color, bg=h.background_color)
    if h.style == "subtle_pill":
        return dict(outline=0.0, shadow=0.0, pill=True, opacity=min(h.background_opacity, 0.6) or 0.45,
                    blur=0.0, color=h.text_color, outline_color=h.outline_color, bg=h.background_color)
    if h.style == "creator_custom":
        return dict(outline=h.outline, shadow=h.shadow, pill=h.background == "pill",
                    opacity=h.background_opacity, blur=0.0, color=h.text_color,
                    outline_color=h.outline_color, bg=h.background_color)
    return dict(outline=3.0, shadow=2.0, pill=False, opacity=0.0, blur=0.0, color=h.text_color,   # clean_white
                outline_color=h.outline_color, bg=h.background_color)


def plan_hook(layout: Layout, cfg: CaptionsCfg, hcfg: HookCfg, hook: Hook,
              caption: Optional[Placement] = None) -> Tuple[Optional[HookPlacement], Optional[str]]:
    """Fit the hook into <= max_lines lines and place it above the panel (relative to the layout).
    Returns (placement, None) or (None, reason) when it cannot be shown cleanly."""
    k = px_scale(layout)
    st = hook_style(hcfg)
    text = apply_case(hook.text, hcfg.text_case) if hcfg.text_case != "auto" else hook.text
    toks = text.split()
    if not 2 <= len(toks) <= max(hcfg.max_words, 2):
        return None, f"hook has {len(toks)} words (limit {hcfg.max_words})"
    margin = int(cfg.margin_side * k) + int(10 * k)
    width = layout.out_w - 2 * margin
    size, lines = 0, None
    for s in range(max(8, int(hcfg.font_size * k)), max(8, int(hcfg.min_font_size * k)) - 1, -2):
        per_line = int(width / (0.56 * s))
        lines = balanced_lines(toks, per_line, hcfg.max_lines, strict=True)
        if lines is not None:
            size = s
            break
    if lines is None:
        return None, "hook text does not fit in the allowed lines at the minimum font size"
    pad = (12 if st["pill"] else st["outline"]) * k
    h = block_height(size, len(lines), pad)
    safe_top = int(cfg.safe_top * k)
    top, bot = safe_top, layout.fg_y - int(24 * k)
    over_panel = False
    if hcfg.y_ratio is not None:
        y = max(safe_top, int(hcfg.y_ratio * layout.out_h))
    elif bot - top >= h:
        y = top + (bot - top - h) // 2
    else:                                   # no room above the panel: sit just inside its top edge
        y, over_panel = max(safe_top, layout.fg_y + int(24 * k)), True
    y = max(0, min(y, layout.out_h - h))
    if caption is not None and y < caption.y_bottom and y + h > caption.y_top:     # never overlap captions
        y2 = caption.y_top - int(20 * k) - h
        if y2 >= safe_top:
            y, over_panel = y2, y2 + h > layout.fg_y
        else:
            return None, "no free space for the hook without overlapping captions"
    over_panel = over_panel or (y + h > layout.fg_y and y < layout.fg_bottom)
    return HookPlacement(tuple(" ".join(toks[i] for i in ln) for ln in lines), size, int(y), int(y + h),
                         pad, over_panel, text), None


# ------------------------------------------------------------------ ASS
def _style_line(name: str, font: str, size: int, bold: int, outline: float, shadow: float, align: int,
                margin_v: int, margin_side: int, primary: str, secondary: str, outline_col: str,
                back: str = "&H78000000", border_style: int = 1) -> str:
    font = font.replace(",", " ")
    return (f"Style: {name},{font},{size},{primary},{secondary},{outline_col},{back},{-1 if bold else 0},"
            f"0,0,0,100,100,0,0,{border_style},{outline:.1f},{shadow:.1f},{align},{margin_side},"
            f"{margin_side},{margin_v},1")


def build_ass(words: Sequence[Word], cfg: CaptionsCfg, layout: Layout, placement: Placement,
              hook: Optional[Hook] = None, hcfg: Optional[HookCfg] = None,
              hook_end: Optional[float] = None, plain: bool = False,
              hook_plan: Optional[HookPlacement] = None) -> Tuple[str, CaptionStats]:
    """`words` are OUTPUT-timeline words. `plain=True` is the safe fallback (static captions)."""
    st = CaptionStats(placement=placement)
    name = "classic" if plain else cfg.style
    spec = STYLE_SPECS[name]
    mode = spec["mode"]
    k = px_scale(layout)
    outline = cfg.outline * spec["outline"] * k
    shadow = cfg.shadow * spec["shadow"] * k
    min_size = max(8, int(cfg.min_font_size * k))
    pri, hl, outc = _inline(cfg.primary), _inline(cfg.highlight), _inline(cfg.outline_color)
    sty = [_style_line("Default", cfg.font, placement.font_size, spec["bold"], outline, shadow, 8,
                       placement.y_top, placement.margin_side,
                       ass_color(cfg.highlight if mode == "karaoke" else cfg.primary),
                       ass_color(cfg.primary), ass_color(cfg.outline_color))]
    events: List[Tuple[float, float, str]] = []
    groups = group_phrases(list(words), cfg, spec["words"]) if cfg.enabled else []
    st.groups = len(groups)
    texts_all = []
    for gi, g in enumerate(groups):
        texts = [display_text(w.text, cfg) for w in g]
        texts_all.append(" ".join(texts))
        lines = balanced_lines(texts, cfg.max_chars, cfg.max_lines) or [list(range(len(g)))]
        longest = max(sum(len(texts[i]) for i in ln) + len(ln) - 1 for ln in lines)
        size = fit_size(placement.font_size, longest, bool(spec["bold"]), placement.max_width,
                        min_size) if cfg.dynamic_size else placement.font_size
        size_tag = f"\\fs{size}" if size != placement.font_size else ""
        emph = pick_emphasis(texts) if (cfg.emphasis and spec["emph"] and not plain) else None
        s0 = g[0].start
        e0 = g[-1].end + 0.12
        if gi + 1 < len(groups):
            e0 = min(e0, groups[gi + 1][0].start)
        if e0 - s0 < 0.05:
            continue
        edges = [w.start for w in g] + [e0]
        rel = [max(0, int(round((t - s0) * 1000))) for t in edges]

        def word_markup(i: int) -> str:
            t = texts[i]
            pre = ""
            if mode == "karaoke":
                cs = max(1, int(round((edges[i + 1] - edges[i]) * 100)))
                pre = f"\\kf{cs}"
            elif mode == "tween":
                a, b = rel[i], rel[i + 1]
                pre = f"\\1c{pri}\\t({a},{a + 70},\\1c{hl})" + (f"\\t({b},{b + 70},\\1c{pri})" if i < len(g) - 1 else "")
            elif mode == "pill":
                a, b = rel[i], rel[i + 1]
                on = f"\\1c&H000000&\\3c{hl}\\bord{outline + 4 * k:.1f}"
                off = f"\\1c{pri}\\3c{outc}\\bord{outline:.1f}"
                pre = f"\\1c{pri}\\t({a},{a + 70},{on})" + (f"\\t({b},{b + 70},{off})" if i < len(g) - 1 else "")
            out = f"{{{pre}}}" if pre else ""
            if i == emph:
                out += "{\\fscx108\\fscy108}" + t + "{\\fscx100\\fscy100}"
            else:
                out += t
            return out

        rows = [" ".join(word_markup(i) for i in ln) for ln in lines]
        events.append((s0, e0, (f"{{{size_tag}}}" if size_tag else "") + "\\N".join(rows)))
    cap_events = [f"Dialogue: 0,{_ts(a)},{_ts(b)},Default,,0,0,0,,{t}" for a, b, t in events]
    st.events = len(cap_events)
    st.group_texts = tuple(texts_all)

    hook_events: List[str] = []
    if hook is not None and hcfg is not None and hcfg.enabled and hook_plan is not None:
        hs = hook_style(hcfg)
        back = ass_color("000000", 0x70)
        if hs["pill"]:
            alpha = int(round((1 - hs["opacity"]) * 255))
            sty.append(_style_line("Hook", cfg.font, hook_plan.font_size, 1, hook_plan.pad, 0, 8, hook_plan.y_top,
                                   placement.margin_side, ass_color(hs["color"]), ass_color(hs["color"]),
                                   ass_color(hs["bg"], alpha), ass_color(hs["bg"], alpha), border_style=3))
        else:
            sty.append(_style_line("Hook", cfg.font, hook_plan.font_size, 1, hs["outline"] * k, hs["shadow"] * k,
                                   8, hook_plan.y_top, placement.margin_side, ass_color(hs["color"]),
                                   ass_color(hs["color"]), ass_color(hs["outline_color"]), back))
        end = min(hcfg.duration, hook_end) if hook_end else hcfg.duration
        fi, fo = int(hcfg.fade_in * 1000), int(hcfg.fade_out * 1000)
        tags = f"\\fad({fi},{fo})"
        if hs["blur"]:
            tags += f"\\blur{hs['blur']}"
        if hcfg.scale_in and fi > 0:
            tags += f"\\fscx94\\fscy94\\t(0,{fi},\\fscx100\\fscy100)"
        body = "\\N".join(l.replace("{", "").replace("}", "") for l in hook_plan.lines)
        hook_events.append(f"Dialogue: 1,{_ts(0.05)},{_ts(end)},Hook,,0,0,0,,{{{tags}}}{body}")
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
    issues = []
    if p.y_top < 0 or p.y_bottom > layout.out_h:
        issues.append(f"caption block y {p.y_top}-{p.y_bottom} outside frame height {layout.out_h}")
    if p.margin_side * 2 >= layout.out_w:
        issues.append("caption side margins leave no width")
    if p.position == "below" and p.y_bottom > layout.out_h - int(cfg.safe_bottom * px_scale(layout)) + 1:
        issues.append("caption block enters the bottom platform-safe area")
    return issues


def hook_box_issues(h: HookPlacement, layout: Layout, cfg: CaptionsCfg, cap: Placement) -> List[str]:
    k = px_scale(layout)
    issues = []
    if h.y_top < 0 or h.y_bottom > layout.out_h:
        issues.append("hook is outside the frame")
    if h.y_top < int(cfg.safe_top * k) - 1:
        issues.append("hook enters the top platform-safe area")
    if h.y_top < cap.y_bottom and h.y_bottom > cap.y_top:
        issues.append("hook overlaps the caption block")
    return issues
