"""Glue between the production pipeline and the optional phase3 package."""
import copy
import logging

from src.pipeline.config import AppConfig
from src.utils.errors import ConfigError

log = logging.getLogger(__name__)


def build_phase3_cfg(cfg: AppConfig):
    """Validated Phase3Cfg, or None when phase3.enabled is false/absent.

    Output size/encoder settings default to the `video:` section (single source of truth)
    unless phase3.output sets them explicitly. Raises ConfigError if Phase 3 is enabled but
    unusable, so a broken Phase 3 never silently falls back to an older renderer."""
    section = cfg.phase3
    if not section or not section.get("enabled", False):
        return None
    try:
        from phase3.config import Phase3Cfg, Phase3ConfigError
        import phase3.render  # noqa: F401  (fail fast if any phase3 module is broken)
    except ImportError as exc:
        raise ConfigError(f"phase3.enabled is true but the phase3 package cannot be imported: {exc}") from exc
    sec = copy.deepcopy(section)
    out = sec.setdefault("output", {})
    v = cfg.video
    if "resolution" not in out:
        out["resolution"] = f"{v.width}x{v.height}"
    out.setdefault("crf", v.crf)
    out.setdefault("preset", v.preset)
    out.setdefault("audio_bitrate", v.audio_bitrate)
    try:
        p3 = Phase3Cfg.from_dict(sec)
    except Phase3ConfigError as exc:
        log.error("Invalid phase3 configuration: %s", exc)
        raise ConfigError(f"Invalid phase3 configuration: {exc}") from exc
    p3.enabled = True
    w, h = p3.output.dims()
    log.info("Phase 3 enabled: preset=%s canvas=%dx%d panel=%s fit=%s captions=%s zoom=%s silence=%s hook=%s",
             p3.preset, w, h, p3.foreground.aspect_ratio, p3.foreground.fit_mode, p3.captions.style,
             p3.zoom.enabled, p3.silence_removal.enabled, p3.hook.enabled)
    return p3
