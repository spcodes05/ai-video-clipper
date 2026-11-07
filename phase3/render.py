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
from .advisor import AdvisorResult
from .broll import get_provider, plan_broll
from .captions import (STYLE_SPECS, build_ass, caption_box_inside, hook_box_issues, plan_hook,
                       plan_placement)
from .config import Phase3Cfg
from .filters import (MAX_INLINE_GRAPH, GraphSpec, build_command, build_filter_complex,
                      resolve_encoder, video_codec_args)
from .framing import (FramingDecision, crop_x_at, decide_mode, face_stats, hook_face_conflict,
                      make_face_detector, plan_crop)
from .hooks import generate_hook
from .jumpcuts import plan_jump_cuts
from .layout import compute_layout
from .reactions import (ExpressionUnavailable, analyze_expressions, merge_windows,
                        plan_reaction_windows, plan_visual_windows)
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
                report_dir: Optional[Path] = None, log: Optional[logging.Logger] = None,
                advisor=None, title: str = "") -> RenderResult:
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
    features: dict = {}
    rep: dict = {"clip": out.name, "source_range": [start, end], "layout": dataclasses.asdict(layout),
                 "preset": cfg.preset, "warnings": []}

    # ---------------------------------------------------------------- faces + framing mode
    samples: list = []
    faces_ok = False
    if cfg.foreground.fit_mode != "contain":
        try:
            det = detector or make_face_detector(cfg.framing.min_confidence, cfg.framing.min_neighbors,
                                                 cfg.group.profile_faces)
            samples = sample_faces(src, start, end, cfg.framing.sample_interval, det,
                                   cfg.framing.analysis_width)
            faces_ok = True
        except DetectorUnavailable as exc:
            log.warning("Face analysis unavailable (%s); framing falls back conservatively", exc)
            rep["warnings"].append(f"face analysis unavailable: {exc}")
            features["face_analysis"] = f"unavailable ({exc})"
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
                                cut_ok=[cw_words[i] for i in jc.removed_idx], db=db)
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

    # ---------------------------------------------------------------- advisor (optional LLM)
    adv = AdvisorResult()
    want_hook = cfg.hook.enabled and not cfg.hook.text and cfg.advisor.hook
    want_moments = cfg.zoom.enabled and decision.mode == "crop" and cfg.advisor.zoom_moments
    if advisor is not None and cw_words and (want_hook or want_moments):
        adv = advisor.advise([w for i, w in enumerate(cw_words) if i not in removed], dur,
                             cfg.hook.max_words, title, bool(laugh_spans))
        features["advisor"] = "used" if adv.used else f"failed: {adv.error}"
    else:
        features["advisor"] = "not configured (local heuristics only)" if advisor is None else "not needed"

    # ---------------------------------------------------------------- reactions + zoom
    reaction_clip: List[Tuple[float, float]] = []
    reaction_basis: List[str] = []
    zoom_events: List[ZoomEvent] = []
    if decision.mode == "crop" and sol is not None:
        audio_w = plan_reaction_windows(laughter, samples, start, end, cfg.reactions)
        vis_w: List[Tuple[float, float]] = []
        if cfg.reactions.enabled and cfg.reactions.visual and sum(1 for _, b in samples if len(b) >= 2) >= 3:
            try:
                expr = analyze_expressions(src, start, samples, cfg.reactions)
                vis_w = plan_visual_windows(expr, cfg.framing.sample_interval, cfg.reactions)
                features["visual_reactions"] = f"analysed {len(expr)} multi-face samples (smile/motion proxies)"
            except ExpressionUnavailable as exc:
                features["visual_reactions"] = f"unavailable ({exc})"
        elif cfg.reactions.enabled and cfg.reactions.visual:
            features["visual_reactions"] = "skipped (fewer than 3 samples with 2+ faces)"
        reaction_clip = merge_windows(list(audio_w) + vis_w)
        reaction_basis = (["audio"] if audio_w else []) + (["visual"] if vis_w else [])
        zoom_events = plan_zoom([w for i, w in enumerate(cw_words) if i not in removed], dur, cfg.zoom,
                                db, laugh_spans, blocked=reaction_clip,
                                moments=[(m.start, m.end, m.strength) for m in adv.moments])
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
    elif decision.mode == "contain":
        features["zoom"] = "off in contain mode (whole frame stays visible)"
        features["reactions"] = "not needed in contain mode (everyone already visible)"
    out_reactions = [m for m in (map_interval(a, b, keep) for a, b in reaction_clip) if m]
    out_zoom: List[ZoomEvent] = []
    for e in zoom_events:
        m = map_interval(e.start, e.end, keep)
        if m and m[1] - m[0] >= max(2 * cfg.zoom.transition_duration, 0.6 * cfg.zoom.min_event_seconds):
            out_zoom.append(dataclasses.replace(e, start=round(m[0], 3), end=round(m[1], 3)))
    rep["zoom"] = [dataclasses.asdict(e) for e in out_zoom]
    rep["reaction_shots"] = [list(m) for m in out_reactions]
    rep["reaction_basis"] = reaction_basis

    # ---------------------------------------------------------------- face clipping metric
    face_clip = None
    if sol is not None and samples:
        face_clip = _face_clip_ratio(samples, sol.keyframes, sol.plan.y, cw, ch, zoom_events,
                                     cfg.zoom.transition_duration, decision.group_crop)

    # ---------------------------------------------------------------- hook + captions
    kept_words = [w for i, w in enumerate(cw_words) if i not in removed]
    hook = generate_hook(kept_words, laugh_spans, cfg.hook, [s_ for _, _, s_ in laugh], adv.hook)
    cap_words = [Word(retime(w.start, keep), retime(w.end, keep), w.text) for w in kept_words]
    placement = plan_placement(layout, cfg.captions)
    geometry_issues = caption_box_inside(placement, layout, cfg.captions)
    hook_plan, hook_reason, hook_issues = None, None, []
    if hook is None:
        hook_reason = "disabled" if not cfg.hook.enabled else "no honest hook found for this clip"
    elif out_dur < 4.0:
        hook_reason, hook = "clip too short for a hook", None
    else:
        hook_plan, hook_reason = plan_hook(layout, cfg.captions, cfg.hook, hook, placement)
        if hook_plan and cfg.hook.avoid_faces and hook_plan.y_bottom > layout.fg_y and hook_plan.y_top < layout.fg_bottom:
            if samples and hook_face_conflict(samples, hook_plan.y_top, hook_plan.y_bottom, decision.mode,
                                              sw, sh, layout, (sol.plan.y if sol else 0), ch):
                hook_plan, hook_reason = None, "hook would cover a face"
            elif not samples:
                rep["warnings"].append("hook sits over the panel and no face data is available to check it")
        if hook_plan:
            hook_issues = hook_box_issues(hook_plan, layout, cfg.captions, placement)
        else:
            hook = None
    rep["hook"] = ({"text": hook_plan.text, "source": hook.source, "style": cfg.hook.style,
                    "lines": list(hook_plan.lines), "y": [hook_plan.y_top, hook_plan.y_bottom],
                    "font_size": hook_plan.font_size, "over_panel": hook_plan.over_panel}
                   if hook_plan and hook else {"text": None, "reason": hook_reason})
    features["hook"] = f"rendered: {hook_plan.text!r}" if hook_plan and hook else f"none ({hook_reason})"
    rep["captions"] = {"style": cfg.captions.style, "position": placement.position,
                       "fallback_position": placement.fallback, "y": [placement.y_top, placement.y_bottom],
                       "font_size": placement.font_size}

    # ---------------------------------------------------------------- b-roll (real compositing)
    broll_out: List[Tuple[str, float, float]] = []
    if cfg.broll.enabled:
        prov = get_provider(cfg.broll)
        ins = plan_broll(kept_words, prov, cfg.broll, dur, list(reaction_clip) + list(laugh_spans))
        for i_ in ins:
            m = map_interval(i_.start, i_.end, keep)
            if m and m[1] - m[0] >= 1.0 and Path(i_.path).is_file():       # kept inside ONE segment
                broll_out.append((i_.path, round(m[0], 3), round(m[1], 3)))
        rep["broll"] = {"status": (f"rendered {len(broll_out)} insert(s)" if broll_out else
                                   "enabled but no matching footage / safe slot"),
                        "inserts": [list(b_) for b_ in broll_out]}
        features["broll"] = rep["broll"]["status"]
    else:
        rep["broll"] = {"status": "disabled"}
        features["broll"] = "disabled"

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
                     reaction_windows=out_reactions, broll=broll_out, bg=cfg.background,
                     panel_blur=cfg.foreground.panel_blur,
                     panel_blur_strength=cfg.foreground.panel_blur_strength,
                     has_audio=info.has_audio, crossfade_ms=cfg.silence_removal.crossfade_ms,
                     loudnorm=cfg.output.loudnorm, target_lufs=cfg.output.target_lufs)

    attempts: List[Optional[bool]] = [False]            # plain? (False = styled)
    need_ass = bool(cfg.captions.enabled and cap_words) or hook_plan is not None
    if need_ass:
        attempts = [False, True]
    codec = resolve_encoder(cfg.output.encoder)
    captions_ok = True
    stats_c = None
    with tempfile.TemporaryDirectory(prefix="clipper3_") as tmp:
        tmpd = Path(tmp)
        done = False
        for n, plain in enumerate(attempts):
            if need_ass:
                ccfg = cfg.captions
                if plain:
                    ccfg = dataclasses.replace(ccfg, font="Arial", font_file=None)
                ass, stats_c = build_ass(cap_words, ccfg, layout, placement, hook, cfg.hook,
                                         hook_end=out_dur * 0.5, plain=bool(plain), hook_plan=hook_plan)
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
                                        video_codec_args(enc, cfg.output), cfg.output.audio_bitrate, script,
                                        [b_[0] for b_ in broll_out])
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
               "mode": decision.mode, "caption_geometry_issues": geometry_issues, "hook_issues": hook_issues}
    if cfg.qa.enabled:
        report = qa_mod.check_clip(out, layout, out_dur, cfg, qa_info)
    else:
        report = qa_mod.QAReport()
    rep["captions"]["groups"] = stats_c.groups if stats_c else 0
    features["captions"] = f"{stats_c.groups} phrases, style {cfg.captions.style}" if stats_c else "disabled"
    features["silence_removal"] = (f"removed {dur - out_dur:.1f}s" if keep else
                                   ("enabled, nothing to remove" if cfg.silence_removal.enabled else "disabled"))
    features["jump_cuts"] = (f"{len(jc.reasons)} cut(s)" if (keep and jc.reasons) else
                             ("enabled, none needed" if cfg.jump_cuts.enabled else "disabled"))
    features.setdefault("zoom", f"{len(out_zoom)} event(s)" if cfg.zoom.enabled else "disabled")
    features.setdefault("reactions", (f"{len(out_reactions)} window(s) via {'+'.join(reaction_basis)}"
                                      if out_reactions else ("none found" if cfg.reactions.enabled else "disabled")))
    rep["features"] = features
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
        qa_mod.write_preview(rep, d / f"{out.stem}.preview.md")
    for wmsg in report.warnings:
        log.warning("%s QA warning: %s", out.name, wmsg)
    if not report.ok:
        raise qa_mod.Phase3QAError(f"{out.name} failed quality checks: " + "; ".join(report.issues))
    return RenderResult(out, out_dur, captions_ok, decision.mode, report, rep)
