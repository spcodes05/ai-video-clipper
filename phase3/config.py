"""Phase 3 configuration: typed dataclasses, strict validation, editing presets.

Precedence (low -> high):  defaults  <  editing preset  <  config.yaml `phase3:`  <  CLI flags.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field, fields, is_dataclass
from fractions import Fraction
from typing import Any, Dict, Optional, Tuple

from phase2.config import FramingCfg


class Phase3ConfigError(ValueError):
    """Invalid phase3 configuration."""


_HEX = re.compile(r"^[0-9A-Fa-f]{6}$")
FIT_MODES = ("auto", "crop", "contain")
CAPTION_STYLES = ("classic", "bold", "karaoke", "highlighted", "minimal")
CAPTION_ALIASES = {"clean": "classic"}          # legacy phase-1 style names
CAPTION_POSITIONS = ("below", "over_panel", "above", "custom")
FG_POSITIONS = ("center", "top", "bottom", "upper", "lower")
ENCODERS = ("auto", "cpu", "nvenc", "qsv", "amf", "videotoolbox")
X264_PRESETS = {"ultrafast", "superfast", "veryfast", "faster", "fast",
                "medium", "slow", "slower", "veryslow"}


def parse_ratio(value: str) -> Tuple[int, int]:
    """'9:16' / '1:1' / '16/9' / '1.7778' -> reduced integer pair."""
    s = str(value).strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[:/x]\s*(\d+(?:\.\d+)?)", s)
    try:
        if m:
            fr = Fraction(m.group(1)) / Fraction(m.group(2))
        else:
            fr = Fraction(s).limit_denominator(100)
    except (ValueError, ZeroDivisionError):
        raise Phase3ConfigError(f"invalid aspect ratio {value!r} (use e.g. 9:16)") from None
    if fr <= 0:
        raise Phase3ConfigError(f"invalid aspect ratio {value!r}")
    fr = fr.limit_denominator(1000)
    return fr.numerator, fr.denominator


def parse_resolution(value: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*[xX*]\s*(\d+)\s*", str(value))
    if not m:
        raise Phase3ConfigError(f"invalid resolution {value!r} (use e.g. 1080x1920)")
    return int(m.group(1)), int(m.group(2))


# --------------------------------------------------------------------------- sections
@dataclass
class OutputCfg:
    aspect_ratio: str = "9:16"
    resolution: str = ""            # "" -> taken from the `video:` section
    fps: int = 30
    crf: int = 20
    preset: str = "veryfast"
    audio_bitrate: str = "160k"
    encoder: str = "auto"           # auto | cpu | nvenc | qsv | amf | videotoolbox
    loudnorm: bool = True
    target_lufs: float = -16.0

    def validate(self) -> None:
        parse_ratio(self.aspect_ratio)
        if self.resolution:
            parse_resolution(self.resolution)
        if not 1 <= self.fps <= 120:
            raise Phase3ConfigError("output.fps must be 1..120")
        if not 0 <= self.crf <= 51:
            raise Phase3ConfigError("output.crf must be 0..51")
        if self.preset not in X264_PRESETS:
            raise Phase3ConfigError(f"output.preset must be one of {sorted(X264_PRESETS)}")
        if self.encoder not in ENCODERS:
            raise Phase3ConfigError(f"output.encoder must be one of {ENCODERS}")
        if not -70 <= self.target_lufs <= -5:
            raise Phase3ConfigError("output.target_lufs must be -70..-5")

    def dims(self) -> Tuple[int, int]:
        """Exact even (width, height) honouring aspect_ratio."""
        a, b = parse_ratio(self.aspect_ratio)
        if self.resolution:
            w, h = parse_resolution(self.resolution)
            if abs(w * b - h * a) > 0.01 * h * a:      # ratio differs -> keep width, derive height
                h = round(w * b / a)
        else:
            w = 1080
            h = round(w * b / a)
        w -= w % 2
        h -= h % 2
        if w < 64 or h < 64:
            raise Phase3ConfigError("output resolution too small")
        return w, h


@dataclass
class ForegroundCfg:
    aspect_ratio: str = "1:1"
    width: int = 0                  # 0 -> full canvas width
    position: str = "center"        # center | top | bottom | upper | lower
    offset_y: int = 0               # px shift after positioning (negative = up)
    fit_mode: str = "auto"          # auto | crop | contain
    panel_blur: bool = True         # contain mode: blurred fill behind the fitted video
    panel_blur_strength: float = 20.0

    def validate(self) -> None:
        parse_ratio(self.aspect_ratio)
        if self.width < 0:
            raise Phase3ConfigError("foreground.width must be >= 0")
        if self.position not in FG_POSITIONS:
            raise Phase3ConfigError(f"foreground.position must be one of {FG_POSITIONS}")
        if self.fit_mode not in FIT_MODES:
            raise Phase3ConfigError(f"foreground.fit_mode must be one of {FIT_MODES}")
        if self.panel_blur_strength < 0:
            raise Phase3ConfigError("foreground.panel_blur_strength must be >= 0")


@dataclass
class BackgroundCfg:
    enabled: bool = True
    mode: str = "blur"              # blur | solid
    blur_strength: float = 24.0     # gaussian sigma at 1080-wide output
    darken: float = 0.12            # 0..0.6 brightness reduction
    saturation: float = 1.1
    color: str = "101010"           # used when mode == solid or enabled == false

    def validate(self) -> None:
        if self.mode not in ("blur", "solid"):
            raise Phase3ConfigError("background.mode must be blur or solid")
        if self.blur_strength < 0 or not 0 <= self.darken <= 0.6 or self.saturation < 0:
            raise Phase3ConfigError("background: blur_strength >= 0, 0 <= darken <= 0.6, saturation >= 0")
        if not _HEX.match(self.color):
            raise Phase3ConfigError("background.color must be RRGGBB")


@dataclass
class GroupCfg:
    group_min_faces: int = 3        # >= this many simultaneous faces -> never crop (contain)
    union_margin: float = 1.25      # two faces are cropped only if their union * margin fits
    contain_when_unsure: bool = True  # AUTO: no reliable face data -> contain instead of centre-crop
    profile_faces: bool = True      # also detect side-on faces (OpenCV profile cascade, mirrored)
    min_face_ratio: float = 0.6     # fraction of samples that must contain a face to trust tracking

    def validate(self) -> None:
        if self.group_min_faces < 2:
            raise Phase3ConfigError("group.group_min_faces must be >= 2")
        if self.union_margin < 1.0 or not 0 <= self.min_face_ratio <= 1:
            raise Phase3ConfigError("group: union_margin >= 1 and 0 <= min_face_ratio <= 1")


@dataclass
class SilenceCfg:
    enabled: bool = False
    threshold_db: float = -38.0     # dBFS below which audio counts as silent
    adaptive: bool = True           # raise the threshold above the clip's noise floor if needed
    minimum_silence_duration: float = 0.6
    padding: float = 0.12           # kept before/after speech inside every cut
    keep_pause: float = 0.35        # natural pause left after a cut
    preserve_comedic_pauses: bool = True
    comedic_keep: float = 1.0       # comedic/setup pauses are only shortened beyond this
    crossfade_ms: int = 20          # micro-fade at every join (no clicks)
    max_cut_ratio: float = 0.35     # never remove more than this fraction of the clip
    max_segments: int = 40
    dramatic_pause_min: float = 0.8   # a pause this long before a laugh/short payoff is never cut down
    dramatic_pause_keep: float = 2.5  # ...up to this length
    snap_ms: int = 40                 # move each cut to the quietest instant within +-snap_ms (no clicks)
    max_cuts_per_10s: float = 4.0     # pacing guard

    def validate(self) -> None:
        if not -80 <= self.threshold_db <= -10:
            raise Phase3ConfigError("silence_removal.threshold_db must be -80..-10")
        if self.minimum_silence_duration <= 0 or self.padding < 0 or self.keep_pause < 0:
            raise Phase3ConfigError("silence_removal: durations must be positive")
        if self.comedic_keep < self.keep_pause:
            raise Phase3ConfigError("silence_removal.comedic_keep must be >= keep_pause")
        if not 0 <= self.crossfade_ms <= 200:
            raise Phase3ConfigError("silence_removal.crossfade_ms must be 0..200")
        if not 0 < self.max_cut_ratio <= 0.9 or self.max_segments < 2:
            raise Phase3ConfigError("silence_removal: 0 < max_cut_ratio <= 0.9, max_segments >= 2")


@dataclass
class JumpCutCfg:
    enabled: bool = False
    remove_fillers: bool = True
    remove_repeats: bool = True
    max_filler_seconds: float = 0.9
    min_segment: float = 0.8        # no keep-segment shorter than this between two cuts
    max_cut_ratio: float = 0.15

    def validate(self) -> None:
        if self.max_filler_seconds <= 0 or self.min_segment <= 0 or not 0 < self.max_cut_ratio <= 0.5:
            raise Phase3ConfigError("jump_cuts: positive durations and 0 < max_cut_ratio <= 0.5")


@dataclass
class ZoomCfg:
    enabled: bool = True
    normal_scale: float = 1.0
    small_scale: float = 1.04       # one solid signal
    emphasis_scale: float = 1.08    # strong moment (punchline after a beat, LLM-marked, loud + sentence end)
    maximum_scale: float = 1.12     # reserved for the best moment of the clip (punchline + laugh / reaction)
    transition_duration: float = 0.4
    min_event_seconds: float = 1.2
    max_event_seconds: float = 2.6
    min_spacing: float = 3.5
    max_fraction: float = 0.3       # at most this share of the clip is zoomed
    min_score: float = 1.2          # a moment needs this much combined evidence to earn a zoom
    min_signals: int = 2            # ...from at least this many independent signals (or one very strong one)
    face_safe: bool = True          # never zoom so far that a tracked face is cut off

    def validate(self) -> None:
        if not 1.0 <= self.normal_scale <= self.small_scale <= self.emphasis_scale <= self.maximum_scale <= 1.5:
            raise Phase3ConfigError(
                "zoom: require 1.0 <= normal_scale <= small_scale <= emphasis_scale <= maximum_scale <= 1.5")
        if self.transition_duration <= 0.05:
            raise Phase3ConfigError("zoom.transition_duration must be > 0.05")
        if self.min_event_seconds < 2 * self.transition_duration:
            raise Phase3ConfigError("zoom.min_event_seconds must be >= 2 * transition_duration")
        if self.max_event_seconds < self.min_event_seconds or not 0 < self.max_fraction <= 1:
            raise Phase3ConfigError("zoom: max_event_seconds >= min_event_seconds, 0 < max_fraction <= 1")
        if self.min_signals < 1:
            raise Phase3ConfigError("zoom.min_signals must be >= 1")


@dataclass
class ReactionCfg:
    enabled: bool = True
    min_window: float = 0.8
    max_window: float = 4.0
    min_laughter_score: float = 0.35
    min_faces: int = 2
    min_face_ratio: float = 0.5     # fraction of samples inside the window that must show >= min_faces
    visual: bool = True             # also look for sustained smiles / head motion (OpenCV, CPU)
    min_smile_faces: int = 2        # faces that must smile together for a visual group reaction
    smile_ratio: float = 0.5        # share of samples in the window that must show it
    motion_threshold: float = 12.0  # mean abs frame difference inside a face that counts as "animated"

    def validate(self) -> None:
        if self.min_window <= 0 or self.max_window < self.min_window or self.min_faces < 2:
            raise Phase3ConfigError("reactions: min_window > 0, max_window >= min_window, min_faces >= 2")
        if self.min_smile_faces < 2 or not 0 < self.smile_ratio <= 1 or self.motion_threshold <= 0:
            raise Phase3ConfigError("reactions: min_smile_faces >= 2, 0 < smile_ratio <= 1, motion_threshold > 0")


@dataclass
class CaptionsCfg:
    enabled: bool = True
    style: str = "bold"             # classic | bold (clean bold) | karaoke | highlighted | minimal
    font: str = "Arial"             # any installed bold sans-serif (e.g. "Segoe UI", "Montserrat") or font_file
    font_file: Optional[str] = None
    font_size: int = 64
    min_font_size: int = 44
    primary: str = "FFFFFF"
    highlight: str = "FFD93B"       # active word only
    emphasis_color: str = "FFFFFF"  # important words are emphasised by size, not by a new colour
    outline_color: str = "000000"
    outline: int = 5
    shadow: int = 1
    text_case: str = "auto"         # auto (as spoken) | sentence | upper
    max_words: int = 6
    max_chars: int = 22             # per line
    max_lines: int = 2
    position: str = "below"         # below | over_panel | above | custom
    gap: int = 24                   # px between panel edge and caption block
    y_ratio: Optional[float] = None  # custom: top of block as fraction of canvas height
    margin_side: int = 70
    safe_top: int = 150
    safe_bottom: int = 200
    emphasis: bool = True
    dynamic_size: bool = True
    min_word_seconds: float = 0.08
    max_gap: float = 0.8

    def validate(self) -> None:
        if self.style not in CAPTION_STYLES:
            raise Phase3ConfigError(f"captions.style must be one of {CAPTION_STYLES}")
        for n in ("primary", "highlight", "emphasis_color", "outline_color"):
            if not _HEX.match(getattr(self, n)):
                raise Phase3ConfigError(f"captions.{n} must be RRGGBB")
        if self.text_case not in ("auto", "sentence", "upper"):
            raise Phase3ConfigError("captions.text_case must be auto, sentence or upper")
        if self.font_size < 16 or not 8 <= self.min_font_size <= self.font_size:
            raise Phase3ConfigError("captions: font_size >= 16 and 8 <= min_font_size <= font_size")
        if self.max_words < 1 or self.max_chars < 5 or not 1 <= self.max_lines <= 3:
            raise Phase3ConfigError("captions: max_words >= 1, max_chars >= 5, max_lines 1..3")
        if self.position not in CAPTION_POSITIONS:
            raise Phase3ConfigError(f"captions.position must be one of {CAPTION_POSITIONS}")
        if self.position == "custom" and self.y_ratio is None:
            raise Phase3ConfigError("captions.position 'custom' requires captions.y_ratio")
        if self.y_ratio is not None and not 0 <= self.y_ratio <= 1:
            raise Phase3ConfigError("captions.y_ratio must be in [0, 1]")
        if min(self.gap, self.margin_side, self.safe_top, self.safe_bottom, self.outline, self.shadow) < 0:
            raise Phase3ConfigError("captions: gap/margins/safe areas/outline/shadow must be >= 0")
        if self.min_word_seconds <= 0 or self.max_gap <= 0:
            raise Phase3ConfigError("captions.min_word_seconds and max_gap must be > 0")


HOOK_STYLES = ("clean_white", "minimal_shadow", "subtle_pill", "creator_custom")


@dataclass
class HookCfg:
    enabled: bool = False
    style: str = "clean_white"      # clean_white | minimal_shadow | subtle_pill | creator_custom
    text: Optional[str] = None      # manual override (rendered with the chosen style)
    mode: str = "auto"              # auto | quote | template   (rule-based fallback behaviour)
    duration: float = 2.5
    max_words: int = 8
    max_lines: int = 2
    font_size: int = 54             # authored for 1080 wide; scaled with the canvas
    min_font_size: int = 34
    text_color: str = "FFFFFF"
    outline_color: str = "000000"
    outline: float = 3.0            # creator_custom
    shadow: float = 2.0             # creator_custom
    background: str = "none"        # none | pill        (creator_custom; subtle_pill is always a pill)
    background_color: str = "000000"
    background_opacity: float = 0.35
    text_case: str = "auto"         # auto | sentence | title | upper | as_is
    y_ratio: Optional[float] = None  # None -> centred in the free space above the panel
    fade_in: float = 0.2
    fade_out: float = 0.3
    scale_in: bool = True           # 94% -> 100% during the fade-in
    avoid_faces: bool = True        # never place the hook over a detected face
    min_score: float = 1.0          # rule-based candidates below this -> no hook at all

    def validate(self) -> None:
        if self.style not in HOOK_STYLES:
            raise Phase3ConfigError(f"hook.style must be one of {HOOK_STYLES}")
        if self.mode not in ("auto", "quote", "template"):
            raise Phase3ConfigError("hook.mode must be auto, quote or template")
        if not 0.5 <= self.duration <= 6.0 or self.max_words < 2 or not 1 <= self.max_lines <= 3:
            raise Phase3ConfigError("hook: 0.5 <= duration <= 6, max_words >= 2, max_lines 1..3")
        if self.font_size < 24 or not 12 <= self.min_font_size <= self.font_size:
            raise Phase3ConfigError("hook: font_size >= 24 and 12 <= min_font_size <= font_size")
        if self.y_ratio is not None and not 0 <= self.y_ratio <= 1:
            raise Phase3ConfigError("hook.y_ratio must be in [0, 1]")
        for n in ("text_color", "outline_color", "background_color"):
            if not _HEX.match(getattr(self, n)):
                raise Phase3ConfigError(f"hook.{n} must be RRGGBB")
        if self.background not in ("none", "pill"):
            raise Phase3ConfigError("hook.background must be none or pill")
        if self.text_case not in ("auto", "sentence", "title", "upper", "as_is"):
            raise Phase3ConfigError("hook.text_case must be auto, sentence, title, upper or as_is")
        if not 0 <= self.background_opacity <= 1 or self.outline < 0 or self.shadow < 0:
            raise Phase3ConfigError("hook: 0 <= background_opacity <= 1, outline/shadow >= 0")
        if self.fade_in < 0 or self.fade_out < 0 or self.fade_in + self.fade_out > self.duration:
            raise Phase3ConfigError("hook: fades must fit inside duration")


@dataclass
class AdvisorCfg:
    """Optional LLM help for the hook and for zoom moments (one small call per clip, cached)."""
    mode: str = "auto"              # auto (only if an API key / local endpoint is configured) | on | off
    hook: bool = True
    zoom_moments: bool = True

    def validate(self) -> None:
        if self.mode not in ("auto", "on", "off"):
            raise Phase3ConfigError("advisor.mode must be auto, on or off")


@dataclass
class BrollCfg:
    enabled: bool = False
    provider: str = "none"          # none | folder
    folder: Optional[str] = None
    max_inserts: int = 3
    max_seconds: float = 2.0
    min_gap: float = 4.0            # minimum seconds between inserts
    min_word_chars: int = 5

    def validate(self) -> None:
        if self.max_seconds <= 0 or self.min_gap < 0 or self.max_inserts < 0:
            raise Phase3ConfigError("broll: max_seconds > 0, min_gap >= 0, max_inserts >= 0")
        if self.provider not in ("none", "folder"):
            raise Phase3ConfigError("broll.provider must be none or folder")
        if self.enabled and self.provider == "folder" and not self.folder:
            raise Phase3ConfigError("broll.provider 'folder' requires broll.folder")


@dataclass
class QACfg:
    enabled: bool = True
    min_seconds: float = 3.0
    max_seconds: float = 180.0
    sync_tolerance: float = 0.25
    max_face_clip_ratio: float = 0.25
    contact_sheet: bool = True
    contact_frames: int = 6
    blank_luma: float = 6.0
    blank_std: float = 2.0
    min_bg_luma: float = 4.0
    check_audio: bool = True
    min_mean_volume_db: float = -40.0
    max_peak_db: float = -0.1

    def validate(self) -> None:
        if self.min_seconds <= 0 or self.max_seconds < self.min_seconds:
            raise Phase3ConfigError("qa: 0 < min_seconds <= max_seconds")
        if not 2 <= self.contact_frames <= 12:
            raise Phase3ConfigError("qa.contact_frames must be 2..12")
        if not 0 <= self.max_face_clip_ratio <= 1:
            raise Phase3ConfigError("qa.max_face_clip_ratio must be in [0, 1]")


# --------------------------------------------------------------------------- presets
PRESETS: Dict[str, Dict[str, Any]] = {
    "general": {
        "foreground": {"fit_mode": "auto"},
        "zoom": {"enabled": True, "small_scale": 1.04, "emphasis_scale": 1.07, "maximum_scale": 1.12},
        "captions": {"style": "bold"},
        "silence_removal": {"enabled": True},
        "jump_cuts": {"enabled": False},
        "reactions": {"enabled": True},
        "hook": {"enabled": False, "style": "clean_white"},
    },
    "podcast": {
        "foreground": {"fit_mode": "auto"},
        "group": {"group_min_faces": 3},
        "zoom": {"enabled": True, "small_scale": 1.03, "emphasis_scale": 1.06, "maximum_scale": 1.10,
                 "min_spacing": 4.5, "max_fraction": 0.22},
        "captions": {"style": "bold"},
        "silence_removal": {"enabled": True, "minimum_silence_duration": 0.7, "keep_pause": 0.4},
        "jump_cuts": {"enabled": True},
        "reactions": {"enabled": True},
        "hook": {"enabled": False, "style": "clean_white"},
    },
    "comedy": {
        "foreground": {"fit_mode": "auto"},
        "zoom": {"enabled": True, "small_scale": 1.04, "emphasis_scale": 1.08, "maximum_scale": 1.12,
                 "transition_duration": 0.3, "min_score": 1.1},
        "captions": {"style": "highlighted", "emphasis": True},
        "silence_removal": {"enabled": True, "preserve_comedic_pauses": True,
                            "minimum_silence_duration": 0.9, "comedic_keep": 1.2,
                            "dramatic_pause_keep": 3.0},
        "jump_cuts": {"enabled": False},
        "reactions": {"enabled": True, "min_laughter_score": 0.3},
        "hook": {"enabled": False, "style": "clean_white"},
    },
    "interview": {
        "foreground": {"fit_mode": "auto"},
        "group": {"group_min_faces": 4},
        "zoom": {"enabled": True, "small_scale": 1.02, "emphasis_scale": 1.04, "maximum_scale": 1.07,
                 "min_spacing": 5.5, "max_fraction": 0.18},
        "captions": {"style": "classic", "emphasis": False},
        "silence_removal": {"enabled": True, "minimum_silence_duration": 0.8, "keep_pause": 0.45},
        "jump_cuts": {"enabled": False},
        "reactions": {"enabled": False},
        "hook": {"enabled": False, "style": "clean_white"},
    },
    # Gameplay: no assumptions about faces/webcams -> never face-crop, whole frame stays visible.
    "gaming": {
        "foreground": {"fit_mode": "contain"},
        "zoom": {"enabled": False},
        "captions": {"style": "bold", "emphasis": True},
        "silence_removal": {"enabled": True, "minimum_silence_duration": 0.5, "keep_pause": 0.25},
        "jump_cuts": {"enabled": True},
        "reactions": {"enabled": False},
        "hook": {"enabled": False, "style": "clean_white"},
    },
    # Narrative: slower pacing, longer pauses survive, gentle emphasis only.
    "storytelling": {
        "foreground": {"fit_mode": "auto"},
        "zoom": {"enabled": True, "small_scale": 1.03, "emphasis_scale": 1.06, "maximum_scale": 1.09,
                 "transition_duration": 0.6, "min_event_seconds": 1.6, "min_spacing": 6.0,
                 "max_fraction": 0.2},
        "captions": {"style": "classic", "emphasis": True},
        "silence_removal": {"enabled": True, "minimum_silence_duration": 1.0, "keep_pause": 0.55,
                            "comedic_keep": 1.6, "dramatic_pause_keep": 3.0},
        "jump_cuts": {"enabled": False},
        "reactions": {"enabled": True},
        "hook": {"enabled": False, "style": "clean_white"},
    },
}
EDITING_STYLES = tuple(sorted(PRESETS))


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# --------------------------------------------------------------------------- builder
def _coerce(default: Any, value: Any, path: str) -> Any:
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, int):
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(default, float):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if ok:
            value = float(value)
    elif isinstance(default, str):
        ok = isinstance(value, str)
    else:  # Optional[...] defaulting to None
        ok = value is None or (isinstance(value, (str, int, float)) and not isinstance(value, bool))
    if not ok:
        raise Phase3ConfigError(f"'{path}' has invalid value {value!r}")
    return value


def _build(cls, data: Any, path: str):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise Phase3ConfigError(f"'{path}' must be a mapping")
    inst = cls()
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise Phase3ConfigError(f"Unknown key(s) in '{path}': {unknown}")
    for key, value in data.items():
        cur = getattr(inst, key)
        if is_dataclass(cur):
            setattr(inst, key, _build(type(cur), value, f"{path}.{key}"))
        else:
            setattr(inst, key, _coerce(cur, value, f"{path}.{key}"))
    try:
        if hasattr(inst, "validate"):
            inst.validate()
    except Phase3ConfigError:
        raise
    except ValueError as exc:             # phase2 FramingCfg raises its own ConfigError
        raise Phase3ConfigError(f"{path}: {exc}") from exc
    return inst


@dataclass
class Phase3Cfg:
    enabled: bool = False
    preset: str = "general"
    output: OutputCfg = field(default_factory=OutputCfg)
    foreground: ForegroundCfg = field(default_factory=ForegroundCfg)
    background: BackgroundCfg = field(default_factory=BackgroundCfg)
    framing: FramingCfg = field(default_factory=FramingCfg)
    group: GroupCfg = field(default_factory=GroupCfg)
    advisor: AdvisorCfg = field(default_factory=AdvisorCfg)
    silence_removal: SilenceCfg = field(default_factory=SilenceCfg)
    jump_cuts: JumpCutCfg = field(default_factory=JumpCutCfg)
    zoom: ZoomCfg = field(default_factory=ZoomCfg)
    reactions: ReactionCfg = field(default_factory=ReactionCfg)
    captions: CaptionsCfg = field(default_factory=CaptionsCfg)
    hook: HookCfg = field(default_factory=HookCfg)
    broll: BrollCfg = field(default_factory=BrollCfg)
    qa: QACfg = field(default_factory=QACfg)

    @classmethod
    def from_dict(cls, section: Optional[dict]) -> "Phase3Cfg":
        """`section` is the raw `phase3:` mapping (may carry `preset`)."""
        section = dict(section or {})
        preset = section.pop("preset", None) or "general"
        if not isinstance(preset, str) or preset not in PRESETS:
            raise Phase3ConfigError(f"Unknown editing style {preset!r}. Available: {', '.join(EDITING_STYLES)}")
        merged = deep_merge(PRESETS[preset], section)
        cap = merged.get("captions")
        if isinstance(cap, dict) and cap.get("style") in CAPTION_ALIASES:
            cap["style"] = CAPTION_ALIASES[cap["style"]]
        cfg = _build(cls, merged, "phase3")
        cfg.preset = preset
        cfg.captions.style = CAPTION_ALIASES.get(cfg.captions.style, cfg.captions.style)
        if cfg.captions.style not in CAPTION_STYLES:
            raise Phase3ConfigError(f"captions.style must be one of {CAPTION_STYLES}")
        if cfg.captions.font_file is not None and not isinstance(cfg.captions.font_file, str):
            raise Phase3ConfigError("captions.font_file must be a path string")
        cfg.output.dims()  # raises on impossible output geometry
        return cfg
