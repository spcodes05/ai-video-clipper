from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional


class ConfigError(ValueError):
    """Invalid phase2 configuration."""


DEFAULT_WEIGHTS: Dict[str, float] = {
    "opening": 0.20,
    "completeness": 0.20,
    "information": 0.20,
    "audio_quality": 0.10,
    "delivery": 0.10,
    "laughter": 0.05,
    "visual": 0.15,
}
_PRESETS = {"ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow"}
_HEX = re.compile(r"^[0-9A-Fa-f]{6}$")


@dataclass
class ClipCfg:
    min_seconds: float = 45.0
    target_seconds: float = 60.0
    max_seconds: float = 90.0
    start_lookback: float = 6.0
    end_lookahead: float = 8.0
    lead_in: float = 0.15
    tail_out: float = 0.35
    max_overlap: float = 0.4
    sentence_gap: float = 0.8

    def validate(self) -> None:
        if not (0 < self.min_seconds <= self.target_seconds <= self.max_seconds):
            raise ConfigError("clip: require 0 < min_seconds <= target_seconds <= max_seconds")
        if self.max_seconds > 180:
            raise ConfigError("clip.max_seconds must be <= 180 (platform limit)")
        if not 0 <= self.max_overlap <= 1:
            raise ConfigError("clip.max_overlap must be between 0 and 1")
        for n in ("start_lookback", "end_lookahead", "lead_in", "tail_out", "sentence_gap"):
            if getattr(self, n) < 0:
                raise ConfigError(f"clip.{n} must be >= 0")


@dataclass
class RenderCfg:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    crf: int = 20
    preset: str = "medium"
    audio_bitrate: str = "192k"
    loudnorm: bool = False
    target_lufs: float = -16.0
    limiter: bool = True
    max_upscale: float = 2.0

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ConfigError("render.width/height must be positive even integers")
        if not 1 <= self.fps <= 120:
            raise ConfigError("render.fps must be 1..120")
        if not 0 <= self.crf <= 51:
            raise ConfigError("render.crf must be 0..51")
        if self.preset not in _PRESETS:
            raise ConfigError(f"render.preset must be one of {sorted(_PRESETS)}")
        if not -70 <= self.target_lufs <= -5:
            raise ConfigError("render.target_lufs must be -70..-5")
        if self.max_upscale < 1:
            raise ConfigError("render.max_upscale must be >= 1")


@dataclass
class FramingCfg:
    enabled: bool = True
    sample_interval: float = 0.5
    analysis_width: int = 640
    min_confidence: float = 2.0
    min_neighbors: int = 5
    deadzone: float = 0.06
    smoothing: float = 0.15
    max_speed: float = 0.25
    keyframe_interval: float = 1.0
    headroom: float = 0.35
    track_ttl: float = 1.5
    match_distance: float = 0.12
    switch_ratio: float = 1.4
    hold_seconds: float = 2.0

    def validate(self) -> None:
        if self.sample_interval <= 0:
            raise ConfigError("framing.sample_interval must be > 0")
        if self.analysis_width < 160:
            raise ConfigError("framing.analysis_width must be >= 160")
        if not 0 <= self.deadzone < 1:
            raise ConfigError("framing.deadzone must be in [0, 1)")
        if not 0 < self.smoothing <= 1:
            raise ConfigError("framing.smoothing must be in (0, 1]")
        if self.max_speed <= 0:
            raise ConfigError("framing.max_speed must be > 0")
        if self.keyframe_interval < 0.25:
            raise ConfigError("framing.keyframe_interval must be >= 0.25")
        if self.min_neighbors < 1 or self.min_confidence < 0:
            raise ConfigError("framing.min_neighbors >= 1 and min_confidence >= 0 required")
        if not 0 <= self.headroom <= 1:
            raise ConfigError("framing.headroom must be in [0, 1]")
        if self.switch_ratio < 1 or self.track_ttl <= 0 or self.match_distance <= 0:
            raise ConfigError("framing: switch_ratio >= 1, track_ttl > 0, match_distance > 0 required")


@dataclass
class CaptionCfg:
    enabled: bool = True
    font: str = "Arial"
    font_file: Optional[str] = None
    font_size: int = 78
    primary: str = "FFFFFF"
    highlight: str = "00E5FF"
    outline_color: str = "000000"
    outline: int = 5
    shadow: int = 2
    max_words: int = 4
    max_chars: int = 22
    max_lines: int = 2
    margin_bottom: int = 420
    min_margin_bottom: int = 220
    margin_side: int = 90
    word_highlight: bool = True
    min_word_seconds: float = 0.08
    max_gap: float = 0.8

    def validate(self) -> None:
        for n in ("primary", "highlight", "outline_color"):
            if not _HEX.match(getattr(self, n)):
                raise ConfigError(f"captions.{n} must be a 6-digit RRGGBB hex string")
        if self.font_size < 8:
            raise ConfigError("captions.font_size must be >= 8")
        if self.max_words < 1 or self.max_chars < 5 or not 1 <= self.max_lines <= 3:
            raise ConfigError("captions: max_words >= 1, max_chars >= 5, max_lines 1..3")
        if min(self.margin_bottom, self.min_margin_bottom, self.margin_side) < 0:
            raise ConfigError("captions margins must be >= 0")
        if self.min_margin_bottom > self.margin_bottom:
            raise ConfigError("captions.min_margin_bottom must be <= margin_bottom")
        if self.min_word_seconds <= 0 or self.max_gap <= 0:
            raise ConfigError("captions.min_word_seconds and max_gap must be > 0")


@dataclass
class EditCfg:
    remove_silence: bool = False
    max_pause: float = 1.2
    keep_pause: float = 0.4

    def validate(self) -> None:
        if self.max_pause <= 0 or not 0 <= self.keep_pause < self.max_pause:
            raise ConfigError("edit: max_pause > 0 and 0 <= keep_pause < max_pause required")


@dataclass
class AnalysisCfg:
    scene_detection: bool = True
    scene_threshold: float = 0.35
    scene_fps: float = 2.0
    laughter: bool = False
    delivery: bool = False

    def validate(self) -> None:
        if not 0 < self.scene_threshold < 1:
            raise ConfigError("analysis.scene_threshold must be in (0, 1)")
        if self.scene_fps <= 0:
            raise ConfigError("analysis.scene_fps must be > 0")


def _build(cls, data: Any, path: str):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"'{path}' must be a mapping")
    defaults = cls()
    unknown = sorted(set(data) - {f.name for f in fields(cls)})
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{path}': {unknown}")
    kwargs = {}
    for key, value in data.items():
        d = getattr(defaults, key)
        if isinstance(d, bool):
            ok = isinstance(value, bool)
        elif isinstance(d, int):
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(d, float):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            if ok:
                value = float(value)
        elif isinstance(d, str):
            ok = isinstance(value, str)
        else:
            ok = value is None or isinstance(value, str)
        if not ok:
            raise ConfigError(f"'{path}.{key}' has invalid value {value!r}")
        kwargs[key] = value
    obj = cls(**kwargs)
    obj.validate()
    return obj


@dataclass
class Phase2Cfg:
    clip: ClipCfg = field(default_factory=ClipCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    framing: FramingCfg = field(default_factory=FramingCfg)
    captions: CaptionCfg = field(default_factory=CaptionCfg)
    edit: EditCfg = field(default_factory=EditCfg)
    analysis: AnalysisCfg = field(default_factory=AnalysisCfg)
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    @classmethod
    def from_config(cls, raw: Optional[dict]) -> "Phase2Cfg":
        """raw = full parsed config.yaml dict; reads the optional 'phase2' section."""
        sec = (raw or {}).get("phase2") or {}
        if not isinstance(sec, dict):
            raise ConfigError("'phase2' must be a mapping")
        allowed = {"clip", "render", "framing", "captions", "edit", "analysis", "weights"}
        unknown = sorted(set(sec) - allowed)
        if unknown:
            raise ConfigError(f"Unknown key(s) in 'phase2': {unknown}")
        weights = dict(DEFAULT_WEIGHTS)
        w = sec.get("weights") or {}
        if not isinstance(w, dict):
            raise ConfigError("'phase2.weights' must be a mapping")
        for k, v in w.items():
            if k not in DEFAULT_WEIGHTS:
                raise ConfigError(f"Unknown weight '{k}'; valid: {sorted(DEFAULT_WEIGHTS)}")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                raise ConfigError(f"phase2.weights.{k} must be a number >= 0")
            weights[k] = float(v)
        if sum(weights.values()) <= 0:
            raise ConfigError("phase2.weights must not all be zero")
        return cls(
            clip=_build(ClipCfg, sec.get("clip"), "phase2.clip"),
            render=_build(RenderCfg, sec.get("render"), "phase2.render"),
            framing=_build(FramingCfg, sec.get("framing"), "phase2.framing"),
            captions=_build(CaptionCfg, sec.get("captions"), "phase2.captions"),
            edit=_build(EditCfg, sec.get("edit"), "phase2.edit"),
            analysis=_build(AnalysisCfg, sec.get("analysis"), "phase2.analysis"),
            weights=weights,
        )