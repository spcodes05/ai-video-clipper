import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.utils.errors import ConfigError


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WhisperConfig(_Cfg):
    model: Literal["tiny", "base", "small", "medium", "large-v3"] = "small"
    device: str = "auto"
    compute_type: str = "auto"


class CandidateConfig(_Cfg):
    min_duration: float = 15.0
    max_duration: float = 60.0
    max_candidates: int = 40
    scene_threshold: float = 27.0
    overlap_suppression: float = 0.5
    signal_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "density": 0.15,
            "emotion": 0.15,
            "qa": 0.10,
            "punchline": 0.15,
            "hook": 0.15,
            "audio": 0.15,
            "scene": 0.05,
            "speech": 0.10,
        }
    )

    @model_validator(mode="after")
    def _check(self):
        if self.min_duration <= 0 or self.max_duration <= self.min_duration:
            raise ValueError("candidates: need 0 < min_duration < max_duration")
        if self.max_candidates < 1:
            raise ValueError("candidates.max_candidates must be >= 1")
        return self


class BoundaryConfig(_Cfg):
    lead_in: float = 0.25
    tail: float = 0.45
    snap_tolerance: float = 3.0


class LLMConfig(_Cfg):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    batch_size: int = 5
    max_retries: int = 3
    context_padding: float = 8.0


class ScoringWeights(_Cfg):
    hook: float = 0.20
    humor: float = 0.20
    surprise: float = 0.15
    emotion: float = 0.15
    shareability: float = 0.10
    visual_interest: float = 0.10
    context_completeness: float = 0.10


class ScoringConfig(_Cfg):
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    context_penalty: float = 0.85

    @model_validator(mode="after")
    def _check(self):
        values = self.weights.model_dump().values()
        if any(v < 0 for v in values) or sum(values) <= 0:
            raise ValueError("scoring.weights must be non-negative and sum to > 0")
        return self


class SelectionConfig(_Cfg):
    num_clips: int = 5
    allow_overlap: bool = False
    max_overlap_ratio: float = 0.0

    @model_validator(mode="after")
    def _check(self):
        if self.num_clips < 1:
            raise ValueError("selection.num_clips must be >= 1")
        return self


class VideoConfig(_Cfg):
    width: int = 1080
    height: int = 1920
    crf: int = 20
    preset: str = "veryfast"
    audio_bitrate: str = "160k"
    reframe_mode: str = "center"


class CaptionConfig(_Cfg):
    style: str = "bold"
    words_per_line: int = 3
    max_chars: int = 20


class AppConfig(_Cfg):
    output_dir: str = "output"
    logs_dir: str = "logs"
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    candidates: CandidateConfig = Field(default_factory=CandidateConfig)
    boundaries: BoundaryConfig = Field(default_factory=BoundaryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    captions: CaptionConfig = Field(default_factory=CaptionConfig)


def _set(data: dict, keys: tuple[str, ...], value) -> None:
    if value is None:
        return
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def load_config(path: str = "config.yaml", overrides: dict | None = None) -> AppConfig:
    """Precedence: defaults < config.yaml < environment < CLI overrides."""
    p = Path(path)
    data: dict = {}
    if p.exists():
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse {p}: {exc}") from exc
    elif path != "config.yaml":
        raise ConfigError(f"Config file not found: {p}")

    _set(data, ("whisper", "model"), os.getenv("WHISPER_MODEL"))
    _set(data, ("output_dir",), os.getenv("OUTPUT_DIR"))

    o = overrides or {}
    _set(data, ("output_dir",), o.get("output"))
    _set(data, ("selection", "num_clips"), o.get("num_clips"))
    _set(data, ("candidates", "min_duration"), o.get("min_duration"))
    _set(data, ("candidates", "max_duration"), o.get("max_duration"))
    _set(data, ("whisper", "model"), o.get("whisper_model"))
    _set(data, ("captions", "style"), o.get("caption_style"))

    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration:\n{exc}") from exc