"""Phase 3 clip renderer: analysis -> edit plan -> ONE FFmpeg pass -> QA + report."""
from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from phase2.boundaries import Word
from phase2.crop import CropPlan, crop_size, effective_size, reduce_keyframes, x_expression
from phase2.faces import DetectorUnavailable, HaarFaceDetector, sample_faces
from phase2.ffmpeg_utils import FFmpegError, run, video_info

from . import qa as qa_mod
from .broll import get_provider, plan_broll
from .captions import STYLE_SPECS, build_ass, caption_box_inside, plan_placement
from .config import Phase3Cfg
from .filters import (MAX_INLINE_GRAPH, GraphSpec, build_command, build_filter_complex,
                      resolve_encoder, video_codec_args)
from .framing import FramingDecision, crop_x_at, decide_mode, face_stats, plan_crop
from .hooks import generate_hook
from .jumpcuts import plan_jump_cuts
from .layout import compute_layout
from .reactions import plan_reaction_windows
from .silence import (SilencePlan, detect_silences, effective_threshold, frame_db, plan_silence,
                      read_audio, validate_keep)
from .timeline import complement, is_identity, keep_duration, map_interval, retime, retime_keyframes
from .zoom import ZoomEvent, max_safe_scale, plan_zoom, zoom_at

LaughWindow = Tuple[float, float, float]


@dataclass
class RenderResult:
    output: Path
    duration: float
    captions_ok: bool
    mode: str
    qa: qa_mod.QAReport
    report: dict = field(default_factory=dict)


def _face_clip_ratio(samples, kf, crop_y: int, cw: int, ch: int, events: Sequence[ZoomEvent],
                     trans: float, group: bool) -> Optional[float]:
    """Share of face-bearing samples in which a face is cut by the (zoomed) crop window."""
    n = bad = 0
    for t, boxes in samples:
        if not boxes:
            continue
        n += 1
        x = crop_x_at(kf, t)
        z = zoom_at(events, t, trans)
        mx, my = cw * (1 - 1 / z) / 2, ch * (1 - 1 / z) / 2
        vx0, vx1, vy0, vy1 = x + mx, x + cw - mx, crop_y + my, crop_y + ch - my
        clipped = False
        for b in boxes:
            fx, fy = b.x + b.w / 2, b.y + b.h / 2
            inside_base = x <= fx <= x + cw and crop_y <= fy <= crop_y + ch
            if not inside_base and not group:
                continue                      # a bystander deliberately left out of a single-speaker crop
            ix = max(0.0, min(b.x + b.w, vx1) - max(b.x, vx0))
            iy = max(0.0, min(b.y + b.h, vy1) - max(b.y, vy0))
            if (ix * iy) / max(b.w * b.h, 1e-9) < 0.65:
                clipped = True
        bad += clipped
    return (bad / n) if n else None


def render_clip(src, out, start: float, end: float, words: Sequence[Word], cfg: Phase3Cfg, *,
                speech_segments: Sequence[Tuple[float, float]] = (), scene_cuts: Sequence[float] = (),
                laughter: Sequence[LaughWindow] = (), detector=None,
                report_dir: Optional[Path] = None, log: Optional[logging.Logger] = None) -> RenderResult:
    """Render one clip. start/end/words/speech/scene_cuts/laughter are in SOURCE seconds."""
    log = log or logging.getLogger("phase3")
    src, out = Path(src).resolve(), Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if end <= start:
        raise ValueError("end must be greater than start")
    if cfg.captions.font_file and not Path(cfg.captions.font_file).is_file():
        raise FileNotFoundError(f"captions.font_file not found: {cfg.captions.font_file}")

    info = video_info(src)
    sw, sh = effective_size(info.width, info.height, info.rotation)
    layout = compute_layout(cfg.output, cfg.foreground)
    dur = end - start
    cw, ch = crop_size(sw, sh, layout.fg_w, layout.fg_h)

    cw_words = sorted((Word(w.start - start, w.end - start, w.text) for w in words
                       if w.start >= start - 1e-6 and w.end <= end + 1e-6), key=lambda w: w.start)
    laugh = [(max(a, start) - start, min(b, end) - start, s) for a, b, s in laughter if b > start and a < end]
    laugh_spans = [(a, b) for a, b, _ in laugh]
    speech = [(a - start, b - start) for a, b in speech_segments]
    cuts = [c - start for c in scene_cuts if start < c < end]
    rep: dict = {"clip": out.name, "source_range": [start, end], "layout": dataclasses.asdict(layout),
                 "preset": cfg.preset, "warnings": []}

    # ---------------------------------------------------------------- faces + framing mode
    samples: list = []
    faces_ok = False
    if cfg.foreground.fit_mode != "contain":
        try:
            det = detector or HaarFaceDetector(cfg.framing.min_confidence, cfg.framing.min_neighbors)
            samples = sample_faces(src, start, end, cfg.framing.sample_interval, det,
                                   cfg.framing.analysis_width)
            faces_ok = True
        except DetectorUnavailable as exc:
            log.warning("Face analysis unavailable (%s); framing falls back conservatively", exc)
            rep["warnings"].append(f"face analysis unavailable: {exc}")
    decision: FramingDecision = decide_mode(cfg.foreground, cfg.group, samples, sw, sh, cw,
                                            layout.fg_ratio, faces_ok)
    stats = decision.stats
    log.info("%s: framing=%s (%s) faces p90=%d presence=%.0f%%", out.name, decision.mode,
             decision.reason, stats.p90_faces, stats.presence * 100)
    rep["framing"] = {"mode": decision.mode, "reason": decision.reason, "faces_p90": stats.p90_faces,
                      "faces_max": stats.max_faces, "face_presence": round(stats.presence, 3)}

    sol = None
    if decision.mode == "crop":
        sol = plan_crop(samples, sw, sh, layout.fg_w, layout.fg_h, cfg.framing, speech, cuts,
                        stats.union_cx if decision.group_crop else None)

    # ---------------------------------------------------------------- audio analysis
    need_audio = (cfg.silence_removal.enabled or (cfg.zoom.enabled and decision.mode == "crop")) \
        and info.has_audio
    audio = read_audio(src, start, dur) if need_audio else None
    db = frame_db(audio) if audio is not None and len(audio) else None

    # ---------------------------------------------------------------- jump cuts + silence
    jc = plan_jump_cuts(cw_words, dur, cfg.jump_cuts, laugh_spans)
    keep: Optional[List[Tuple[float, float]]] = None
    sil_plan: Optional[SilencePlan] = None
    if cfg.silence_removal.enabled and cw_words:
        thr = effective_threshold(db, cfg.silence_removal) if db is not None else cfg.silence_removal.threshold_db
        silences = detect_silences(db, thr, cfg.silence_removal.minimum_silence_duration) if db is not None else None
        if db is None:
            log.warning("%s: no audio analysis; silence removal uses transcript gaps only", out.name)
        sil_plan = plan_silence(cw_words, dur, silences, cfg.silence_removal, laugh_spans, jc.cuts, thr,
                                cut_ok=[cw_words[i] for i in jc.removed_idx])
        keep = sil_plan.keep
    elif jc.cuts:
        keep = complement(jc.cuts, dur)
        if validate_keep([w for i, w in enumerate(cw_words) if i not in jc.removed_idx], keep):
            keep = None
    if keep and is_identity(keep, dur):
        keep = None
    if keep and len(keep) > cfg.silence_removal.max_segments:
        keep = None
        rep["warnings"].append("too many cut segments; editing cuts skipped")
    removed = jc.removed_idx if keep else set()
    out_dur = keep_duration(keep, dur)
    rep["cuts"] = {"enabled": bool(keep), "segments": keep, "removed_seconds": round(dur - out_dur, 3),
                   "jump_cuts": jc.reasons if keep else [],
                   "comedic_pauses_preserved": sil_plan.preserved_comedic if sil_plan else 0}

    # ---------------------------------------------------------------- reactions + zoom
    reaction_clip: List[Tuple[float, float]] = []
    zoom_events: List[ZoomEvent] = []
    if decision.mode == "crop" and sol is not None:
        reaction_clip = plan_reaction_windows(laughter, samples, start, end, cfg.reactions)
        zoom_events = plan_zoom([w for i, w in enumerate(cw_words) if i not in removed], dur, cfg.zoom,
                                db, laugh_spans, blocked=reaction_clip)
        if cfg.zoom.face_safe and samples:
            safe: List[ZoomEvent] = []
            for e in zoom_events:
                lim = 99.0
                for t, boxes in samples:
                    if e.start - 0.3 <= t <= e.end + 0.3 and boxes:
                        x = crop_x_at(sol.keyframes, t)
                        lim = min(lim, max_safe_scale([(b.x, b.y, b.w, b.h) for b in boxes],
                                                      x, sol.plan.y, cw, ch))
                sc = min(e.scale, lim)
                if sc >= 1.02:
                    safe.append(dataclasses.replace(e, scale=round(sc, 4)))
            zoom_events = safe
    out_reactions = [m for m in (map_interval(a, b, keep) for a, b in reaction_clip) if m]
    out_zoom: List[ZoomEvent] = []
    for e in zoom_events:
        m = map_interval(e.start, e.end, keep)
        if m and m[1] - m[0] >= max(2 * cfg.zoom.transition_duration, 0.6 * cfg.zoom.min_event_seconds):
            out_zoom.append(dataclasses.replace(e, start=round(m[0], 3), end=round(m[1], 3)))
    rep["zoom"] = [dataclasses.asdict(e) for e in out_zoom]
    rep["reaction_shots"] = [list(m) for m in out_reactions]

    # ---------------------------------------------------------------- face clipping metric
    face_clip = None
    if sol is not None and samples:
        face_clip = _face_clip_ratio(samples, sol.keyframes, sol.plan.y, cw, ch, zoom_events,
                                     cfg.zoom.transition_duration, decision.group_crop)

    # ---------------------------------------------------------------- hook + captions
    kept_words = [w for i, w in enumerate(cw_words) if i not in removed]
    hook = generate_hook([w for i, w in enumerate(cw_words) if i not in removed], laugh_spans, cfg.hook,
                         [s for _, _, s in laugh])
    cap_words = [Word(retime(w.start, keep), retime(w.end, keep), w.text) for w in kept_words]
    placement = plan_placement(layout, cfg.captions)
    geometry_issues = caption_box_inside(placement, layout, cfg.captions)
    rep["hook"] = dataclasses.asdict(hook) if hook else None
    rep["captions"] = {"style": cfg.captions.style, "position": placement.position,
                       "fallback_position": placement.fallback, "y": [placement.y_top, placement.y_bottom],
                       "font_size": placement.font_size}
    if cfg.broll.enabled:
        ins = plan_broll(cw_words, get_provider(cfg.broll), cfg.broll, reaction_clip)
        rep["broll"] = {"planned": [dataclasses.asdict(i) for i in ins], "rendered": False,
                        "note": "B-roll is planned and reported only; it is not composited into the video."}

    # ---------------------------------------------------------------- crop plan on cut timeline
    plan = None
    if sol is not None:
        kf_out = retime_keyframes(sol.keyframes, keep)
        if keep:
            kf_out = reduce_keyframes(kf_out, 0.01 * cw)
        plan = CropPlan(cw, ch, x_expression(kf_out), sol.plan.y)

    spec = GraphSpec(layout=layout, src_w=sw, src_h=sh, fps=cfg.output.fps, mode=decision.mode,
                     crop_w=cw, crop_h=ch, crop_x_expr=plan.x_expr if plan else "0",
                     crop_y=plan.y if plan else 0, keep=keep, zoom_events=out_zoom,
                     zoom_trans=cfg.zoom.transition_duration, zoom_normal=cfg.zoom.normal_scale,
                     reaction_windows=out_reactions, bg=cfg.background,
                     panel_blur=cfg.foreground.panel_blur,
                     panel_blur_strength=cfg.foreground.panel_blur_strength,
                     has_audio=info.has_audio, crossfade_ms=cfg.silence_removal.crossfade_ms,
                     loudnorm=cfg.output.loudnorm, target_lufs=cfg.output.target_lufs)

    attempts: List[Optional[bool]] = [False]            # plain? (False = styled)
    if cfg.captions.enabled and cap_words:
        attempts = [False, True]
    codec = resolve_encoder(cfg.output.encoder)
    captions_ok = True
    stats_c = None
    with tempfile.TemporaryDirectory(prefix="clipper3_") as tmp:
        tmpd = Path(tmp)
        done = False
        for n, plain in enumerate(attempts):
            if cfg.captions.enabled and cap_words:
                ccfg = cfg.captions
                if plain:
                    ccfg = dataclasses.replace(ccfg, font="Arial", font_file=None)
                ass, stats_c = build_ass(cap_words, ccfg, layout, placement, hook, cfg.hook,
                                         hook_end=out_dur * 0.5, plain=bool(plain))
                (tmpd / "captions.ass").write_text(ass, encoding="utf-8")
                spec.ass_name = "captions.ass"
                spec.fonts_dir = None
                if ccfg.font_file:
                    (tmpd / "fonts").mkdir(exist_ok=True)
                    shutil.copy(ccfg.font_file, tmpd / "fonts" / Path(ccfg.font_file).name)
                    spec.fonts_dir = "fonts"
            graph = build_filter_complex(spec)
            script = None
            if len(graph) > MAX_INLINE_GRAPH:
                script = str(tmpd / "graph.txt")
                Path(script).write_text(graph, encoding="utf-8")
            try:
                for enc in ([codec] if codec == "libx264" else [codec, "libx264"]):
                    cmd = build_command(src, out, start, dur, graph, info.has_audio,
                                        video_codec_args(enc, cfg.output), cfg.output.audio_bitrate, script)
                    try:
                        run(cmd, "render", cwd=tmp)
                        done = True
                        break
                    except FFmpegError as e:
                        if enc == "libx264" or any(k in str(e).lower() for k in ("subtitles", ".ass", "font", "libass")):
                            raise
                        log.warning("Hardware encoder %s failed (%s); retrying on CPU", enc, e)
                captions_ok = n == 0
                break
            except FFmpegError as e:
                caption_related = any(k in str(e).lower() for k in ("subtitles", ".ass", "font", "libass"))
                if n == len(attempts) - 1 or not caption_related:
                    raise
                log.warning("Render with styled captions failed (%s); retrying with plain captions", e)
        if not done:
            raise FFmpegError("render", "FFmpeg did not produce an output")

    # ---------------------------------------------------------------- QA + report
    caption_times = []
    if cap_words:
        step = max(1, len(cap_words) // 5)
        caption_times = [(cap_words[i].start + cap_words[i].end) / 2 for i in range(0, len(cap_words), step)][:5]
    qa_info = {"src_has_audio": info.has_audio, "words": cap_words, "caption_events": stats_c.events if stats_c else 0,
               "caption_times": caption_times, "placement": placement, "face_clip_ratio": face_clip,
               "mode": decision.mode, "caption_geometry_issues": geometry_issues}
    if cfg.qa.enabled:
        report = qa_mod.check_clip(out, layout, out_dur, cfg, qa_info)
    else:
        report = qa_mod.QAReport()
    rep["qa"] = report.as_dict()
    rep["output_duration"] = round(out_dur, 3)
    rep["captions"]["styled"] = captions_ok
    rep["encoder"] = codec
    if report_dir:
        d = Path(report_dir)
        d.mkdir(parents=True, exist_ok=True)
        if cfg.qa.enabled and cfg.qa.contact_sheet:
            ok = qa_mod.contact_sheet(out, layout, d / f"{out.stem}.contact.jpg", cfg.qa.contact_frames)
            rep["contact_sheet"] = f"{out.stem}.contact.jpg" if ok else None
        (d / f"{out.stem}.report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    for wmsg in report.warnings:
        log.warning("%s QA warning: %s", out.name, wmsg)
    if not report.ok:
        raise qa_mod.Phase3QAError(f"{out.name} failed quality checks: " + "; ".join(report.issues))
    return RenderResult(out, out_dur, captions_ok, decision.mode, report, rep)
