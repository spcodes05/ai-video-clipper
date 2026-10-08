from dataclasses import dataclass
from pathlib import Path

from src.captions.subtitle_generator import CaptionLine
from src.utils.errors import ConfigError


@dataclass(frozen=True)
class CaptionStyle:
    name: str
    font: str = "Arial"
    font_size: int = 88
    primary: str = "&H00FFFFFF"   # ASS colour order is &HAABBGGRR
    outline: str = "&H00000000"
    back: str = "&H64000000"
    bold: int = -1
    outline_width: int = 7
    shadow: int = 2
    margin_h: int = 80
    margin_v: int = 450           # distance from bottom: lower-middle of a 1920px frame
    uppercase: bool = True


# Add "highlight", "karaoke", creator-specific styles here later.
STYLES: dict[str, CaptionStyle] = {
    "bold": CaptionStyle(name="bold"),
    "clean": CaptionStyle(name="clean", font_size=72, bold=0, outline_width=4, uppercase=False),
}


def get_style(name: str) -> CaptionStyle:
    if name not in STYLES:
        raise ConfigError(f"Unknown caption style '{name}'. Available: {', '.join(sorted(STYLES))}")
    return STYLES[name]


def format_ass_time(seconds: float) -> str:
    cs_total = int(round(max(seconds, 0.0) * 100))
    h, rem = divmod(cs_total, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\n", " ")


def render_ass(lines: list[CaptionLine], style: CaptionStyle, width: int, height: int) -> str:
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{style.font},{style.font_size},{style.primary},&H000000FF,{style.outline},"
        f"{style.back},{style.bold},0,0,0,100,100,0,0,1,{style.outline_width},{style.shadow},2,"
        f"{style.margin_h},{style.margin_h},{style.margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = []
    for line in lines:
        text = line.text.upper() if style.uppercase else line.text
        events.append(
            f"Dialogue: 0,{format_ass_time(line.start)},{format_ass_time(line.end)},Default,,0,0,0,,{_escape(text)}"
        )
    return header + "\n".join(events) + "\n"


def write_ass(lines: list[CaptionLine], path: Path, style: CaptionStyle, width: int, height: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(render_ass(lines, style, width, height), encoding="utf-8")