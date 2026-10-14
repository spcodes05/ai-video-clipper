from __future__ import annotations

from typing import List, Optional, Sequence

from .boundaries import Word
from .config import CaptionCfg


def _style_color(rrggbb: str) -> str:
    return f"&H00{rrggbb[4:6]}{rrggbb[2:4]}{rrggbb[0:2]}"


def _inline_color(rrggbb: str) -> str:
    return f"&H{rrggbb[4:6]}{rrggbb[2:4]}{rrggbb[0:2]}&"


def _ts(sec: float) -> str:
    cs = int(round(max(0.0, sec) * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _clean(text: str) -> str:
    return (text.replace("\\", "").replace("{", "(").replace("}", ")")
            .replace("\n", " ").strip())


def group_words(words: Sequence[Word], cfg: CaptionCfg) -> List[List[Word]]:
    groups: List[List[Word]] = []
    cur: List[Word] = []
    for w in words:
        if not _clean(w.text):
            continue
        if cur:
            chars = len(" ".join(_clean(x.text) for x in cur + [w]))
            last = _clean(cur[-1].text)
            strong = last.endswith((".", "?", "!", "…"))
            soft = last.endswith((",", ";", ":")) and len(cur) >= 2
            if (len(cur) >= cfg.max_words or chars > cfg.max_chars * cfg.max_lines
                    or (w.start - cur[-1].end) > cfg.max_gap or strong or soft):
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def line_breaks(texts: Sequence[str], max_chars: int, max_lines: int) -> List[List[int]]:
    lines: List[List[int]] = [[]]
    n = 0
    for i, t in enumerate(texts):
        add = len(t) + (1 if lines[-1] else 0)
        if lines[-1] and n + add > max_chars and len(lines) < max_lines:
            lines.append([])
            n = 0
            add = len(t)
        lines[-1].append(i)
        n += add
    return lines


def build_ass(words: Sequence[Word], cfg: CaptionCfg, width: int, height: int,
              margin_bottom: Optional[int] = None) -> str:
    mv = cfg.margin_bottom if margin_bottom is None else margin_bottom
    pri, hl = _inline_color(cfg.primary), _inline_color(cfg.highlight)
    font = cfg.font.replace(",", " ")
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},{cfg.font_size},{_style_color(cfg.primary)},"
        f"{_style_color(cfg.primary)},{_style_color(cfg.outline_color)},&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,{cfg.outline},{cfg.shadow},2,"
        f"{cfg.margin_side},{cfg.margin_side},{mv},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    groups = group_words(words, cfg)
    lines_out: List[str] = []
    for gi, g in enumerate(groups):
        texts = [_clean(w.text) for w in g]
        lines = line_breaks(texts, cfg.max_chars, cfg.max_lines)
        next_start = groups[gi + 1][0].start if gi + 1 < len(groups) else None

        def render(active: int) -> str:
            rows = []
            for line in lines:
                parts = []
                for i in line:
                    if i == active:
                        parts.append(f"{{\\c{hl}}}{texts[i]}{{\\c{pri}}}")
                    else:
                        parts.append(texts[i])
                rows.append(" ".join(parts))
            return "\\N".join(rows)

        events = []
        if cfg.word_highlight:
            for i, w in enumerate(g):
                st = w.start
                en = g[i + 1].start if i + 1 < len(g) else w.end + 0.12
                en = max(en, st + cfg.min_word_seconds)
                if next_start is not None:
                    en = min(en, next_start)
                if en > st:
                    events.append((st, en, render(i)))
        else:
            st, en = g[0].start, g[-1].end + 0.12
            if next_start is not None:
                en = min(en, next_start)
            if en > st:
                events.append((st, en, render(-1)))
        for st, en, text in events:
            lines_out.append(f"Dialogue: 0,{_ts(st)},{_ts(en)},Default,,0,0,0,,{text}")
    return header + "\n".join(lines_out) + "\n"


def caption_margin_for_faces(cfg: CaptionCfg, out_h: int, max_face_bottom: Optional[float]) -> int:
    """Lower the caption block toward min_margin_bottom if a face would overlap it."""
    if max_face_bottom is None:
        return cfg.margin_bottom
    block = cfg.font_size * 1.25 * cfg.max_lines
    top = out_h - cfg.margin_bottom - block
    if max_face_bottom > top:
        return cfg.min_margin_bottom
    return cfg.margin_bottom