from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .boundaries import Word, compute_keep_segments, retime_words
from .captions import build_ass, caption_margin_for_faces
from .config import Phase2Cfg
from .crop import (CropPlan, crop_size, effective_size, output_size, reduce_keyframes,
                   smooth_path, static_y, thin_keyframes, x_expression)
from .faces import DetectorUnavailable, HaarFaceDetector, sample_faces
from .ffmpeg_utils import (FFmpegError, build_audio_filter, build_render_command,
                           build_video_filter, run, validate_output, video_info)
from .tracking import track_subject


@dataclass
class RenderResult:
    output: Path
    duration: float
    captions_ok: bool
    used_face_tracking: bool


def render_clip(src, out, start: float, end: float, words: Sequence[Word],
                cfg: Phase2Cfg, *, speech_segments: Sequence[Tuple[float, float]] = (),
                scene_cuts: Sequence[float] = (), detector=None,
                debug_dir: Optional[Path] = None,
                log: Optional[logging.Logger] = None) -> RenderResult:
    """Render one clip. start/end/words/speech/scene_cuts are in SOURCE seconds."""
    log = log or logging.getLogger("phase2")
    src, out = Path(src).resolve(), Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if end <= start:
        raise ValueError("end must be greater than start")
    if cfg.captions.font_file and not Path(cfg.captions.font_file).is_file():
        raise FileNotFoundError(f"captions.font_file not found: {cfg.captions.font_file}")

    info = video_info(src)
    sw, sh = effective_size(info.width, info.height, info.rotation)
    cw, ch = crop_size(sw, sh, cfg.render.width, cfg.render.height)
    out_w, out_h = output_size(ch, cfg.render.width, cfg.render.height, cfg.render.max_upscale)
    clip_dur = end - start
    cuts = [c - start for c in scene_cuts if start < c < end]
    speech = [(a - start, b - start) for a, b in speech_segments]
    clip_words = [Word(w.start - start, w.end - start, w.text) for w in words
                  if w.start >= start - 1e-6 and w.end <= end + 1e-6]

    raw: List[Tuple[float, Optional[float]]] = []
    cys: List[float] = []
    samples = []
    used = False
    if cfg.framing.enabled and (cw < sw or ch < sh):
        try:
            det = detector or HaarFaceDetector(cfg.framing.min_confidence,
                                               cfg.framing.min_neighbors)
            samples = sample_faces(src, start, end, cfg.framing.sample_interval, det,
                                   cfg.framing.analysis_width)
            subj = track_subject(samples, sw, cfg.framing, speech, cuts)
            raw = [(s.t, s.cx) for s in subj]
            cys = [s.cy for s in subj if s.cy is not None]
            used = any(s.cx is not None for s in subj)
            if not used:
                log.warning("No faces detected in %s; using centered crop", src.name)
        except DetectorUnavailable as e:
            log.warning("Face tracking unavailable (%s); using centered crop", e)

    path = smooth_path(raw, sw, cw, cfg.framing, cuts)
    kf = reduce_keyframes(thin_keyframes(path, cfg.framing.keyframe_interval), 0.01 * cw)
    plan = CropPlan(cw, ch, x_expression(kf), static_y(cys, ch, sh, cfg.framing.headroom))

    keep = None
    if cfg.edit.remove_silence:
        k = compute_keep_segments(clip_words, clip_dur, cfg.edit.max_pause, cfg.edit.keep_pause)
        keep = None if (len(k) == 1 and k[0][0] <= 0.01 and abs(k[0][1] - clip_dur) < 0.01) else k
    expected = sum(b - a for a, b in keep) if keep else clip_dur
    cap_words = retime_words(clip_words, keep) if keep else clip_words

    face_bottom = None
    if samples:
        bottoms = [(b.y + b.h - plan.y) * out_h / ch for _, bs in samples for b in bs]
        face_bottom = max(bottoms) if bottoms else None
    margin = caption_margin_for_faces(cfg.captions, out_h, face_bottom)

    af = build_audio_filter(cfg.render.loudnorm, cfg.render.target_lufs,
                            cfg.render.limiter, keep) if info.has_audio else None

    attempts = [None]
    if cfg.captions.enabled and cap_words:
        fallback = dataclasses.replace(cfg.captions, word_highlight=False, font="Arial",
                                       font_file=None, shadow=0)
        attempts = [cfg.captions, fallback]

    captions_ok = True
    with tempfile.TemporaryDirectory(prefix="clipper_") as tmp:
        tmpd = Path(tmp)
        for n, ccfg in enumerate(attempts):
            ass_name = fonts_dir = None
            if ccfg is not None:
                (tmpd / "captions.ass").write_text(
                    build_ass(cap_words, ccfg, out_w, out_h, margin), encoding="utf-8")
                ass_name = "captions.ass"
                if ccfg.font_file:
                    (tmpd / "fonts").mkdir(exist_ok=True)
                    shutil.copy(ccfg.font_file, tmpd / "fonts" / Path(ccfg.font_file).name)
                    fonts_dir = "fonts"
            vf = build_video_filter(plan, out_w, out_h, cfg.render.fps, keep, ass_name, fonts_dir)
            cmd = build_render_command(src, out, start, clip_dur, vf, af, cfg.render.crf,
                                       cfg.render.preset, cfg.render.audio_bitrate,
                                       info.has_audio)
            try:
                run(cmd, "render", cwd=tmp)
                captions_ok = n == 0
                break
            except FFmpegError as e:
                # Only a caption problem justifies the plain-caption retry; anything else
                # (e.g. a crop/filter error) would fail again, so surface it immediately.
                caption_related = any(k in str(e).lower() for k in ("subtitles", ".ass", "font", "libass"))
                if n == len(attempts) - 1 or not caption_related:
                    raise
                log.warning("Render with styled captions failed (%s); retrying with plain captions", e)

    validate_output(out, out_w, out_h, expected, info.has_audio)

    if debug_dir:
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{out.stem}.analysis.json").write_text(json.dumps({
            "crop": {"cw": cw, "ch": ch, "y": plan.y, "x_expr": plan.x_expr},
            "keyframes": kf, "keep": keep, "scene_cuts": cuts,
            "samples": [(t, [dataclasses.asdict(b) for b in bs]) for t, bs in samples],
        }, indent=2), encoding="utf-8")
    return RenderResult(out, expected, captions_ok, used)