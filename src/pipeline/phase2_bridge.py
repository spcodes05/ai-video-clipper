"""Glue between the production pipeline and the optional phase2 package."""
import dataclasses
import logging
from typing import Sequence

from src.models.schemas import Scene
from src.pipeline.config import AppConfig, SelectionConfig
from src.utils.errors import ConfigError

log = logging.getLogger(__name__)


def build_phase2_cfg(cfg: AppConfig):
    """Return a validated Phase2Cfg, or None when phase2.enabled is false/absent.

    Raises ConfigError (after logging) if Phase 2 is enabled but unusable, so an
    enabled-but-broken Phase 2 never silently falls back to the legacy renderer.
    Output size and encoder settings come from the `video:` section (the single
    source of truth, also used by clip QA); phase2.render keeps fps/loudness settings.
    """
    section = cfg.phase2
    if not section or not section.get("enabled", False):
        return None
    try:
        from phase2.config import ConfigError as P2ConfigError
        from phase2.config import Phase2Cfg
        import phase2.pipeline  # noqa: F401  (fail fast if any phase2 module is broken)
    except ImportError as exc:
        log.error("Phase 2 is enabled but cannot be imported: %s", exc)
        raise ConfigError(f"phase2.enabled is true but the phase2 package cannot be imported: {exc}") from exc
    try:
        p2 = Phase2Cfg.from_config({"phase2": section})
    except P2ConfigError as exc:
        log.error("Invalid phase2 configuration: %s", exc)
        raise ConfigError(f"Invalid phase2 configuration: {exc}") from exc

    v = cfg.video
    render = dataclasses.replace(
        p2.render,
        width=v.width,
        height=v.height,
        crf=v.crf,
        preset=v.preset,
        audio_bitrate=v.audio_bitrate,
        max_upscale=1e9,  # always emit video.width x video.height, as the legacy renderer does
    )
    log.info(
        "Phase 2 enabled: face tracking=%s, captions=%s, loudnorm=%s, silence removal=%s "
        "(size/encoder settings taken from video.*)",
        p2.framing.enabled, p2.captions.enabled, render.loudnorm, p2.edit.remove_silence,
    )
    return dataclasses.replace(p2, render=render)


def scene_cuts_from(scenes: Sequence[Scene]) -> list[float]:
    """Source-time scene cut positions (start of every scene except the first)."""
    return [float(s.start) for s in scenes if s.start > 0.01]


def selection_for(selection: SelectionConfig, phase2_cfg) -> SelectionConfig:
    """Greedy score-ordered overlap suppression already lives in select_clips (same
    algorithm as phase2.boundaries.dedupe). With Phase 2 on, allow up to
    phase2.clip.max_overlap overlap (fraction of the shorter clip) instead of none."""
    if phase2_cfg is None or selection.allow_overlap:
        return selection
    ratio = max(selection.max_overlap_ratio, phase2_cfg.clip.max_overlap)
    return selection.model_copy(update={"max_overlap_ratio": ratio})
