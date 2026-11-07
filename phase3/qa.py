"""Post-render quality checks, per-clip reports and contact sheets."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from phase2.ffmpeg_utils import FFMPEG, FFPROBE

from .config import Phase3Cfg
from .layout import Layout


class Phase3QAError(RuntimeError):
    """A critical QA check failed; the message lists every problem."""


@dataclass
class QAReport:
    issues: List[str] = field(default_factory=list)       # critical
    warnings: List[str] = field(default_factory=list)
    checks: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict:
        return {"ok": self.ok, "issues": self.issues, "warnings": self.warnings, "checks": self.checks}


def probe_streams(path) -> dict:
    p = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams",
                        "-show_format", str(path)], capture_output=True, check=False)
    if p.returncode != 0:
        raise Phase3QAError(f"ffprobe cannot read {path}")
    d = json.loads(p.stdout.decode("utf-8", "replace"))
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), None)
    fmt = float(d.get("format", {}).get("duration") or 0.0)

    def dur(s):
        try:
            return float(s.get("duration")) if s and s.get("duration") not in (None, "N/A") else fmt
        except (TypeError, ValueError):
            return fmt

    return {"v": v, "a": a, "format_duration": fmt, "v_dur": dur(v), "a_dur": dur(a),
            "v_start": float((v or {}).get("start_time") or 0.0),
            "a_start": float((a or {}).get("start_time") or 0.0)}


def _laplacian_var(cv2, gray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_clip(path, layout: Layout, expected_duration: float, cfg: Phase3Cfg, info: dict) -> QAReport:
    """info keys: src_has_audio, words, caption_events, caption_times, placement, face_clip_ratio,
    mode, caption_geometry_issues."""
    rep = QAReport()
    p = Path(path)
    q = cfg.qa
    if not p.exists() or p.stat().st_size == 0:
        rep.issues.append("video missing or empty")
        return rep
    try:
        s = probe_streams(p)
    except Phase3QAError as exc:
        rep.issues.append(f"video not playable: {exc}")
        return rep
    v = s["v"]
    if v is None:
        rep.issues.append("no video stream")
        return rep
    w, h = int(v["width"]), int(v["height"])
    rep.checks.update(width=w, height=h, video_duration=round(s["v_dur"], 3))
    if (w, h) != (layout.out_w, layout.out_h):
        rep.issues.append(f"resolution {w}x{h}, expected {layout.out_w}x{layout.out_h}")
    from .config import parse_ratio
    a_, b_ = parse_ratio(cfg.output.aspect_ratio)
    if w * b_ != h * a_:
        rep.issues.append(f"canvas {w}x{h} is not exactly {cfg.output.aspect_ratio}")
    fa, fb = parse_ratio(cfg.foreground.aspect_ratio)
    if layout.fg_w * fb != layout.fg_h * fa:
        rep.issues.append(f"foreground panel {layout.fg_w}x{layout.fg_h} is not exactly {cfg.foreground.aspect_ratio}")
    rep.checks["foreground"] = [layout.fg_x, layout.fg_y, layout.fg_w, layout.fg_h]

    if info.get("src_has_audio", True):
        if s["a"] is None:
            rep.issues.append("audio stream missing")
        else:
            drift = abs(s["a_dur"] - s["v_dur"])
            rep.checks["av_duration_diff"] = round(drift, 3)
            if drift > q.sync_tolerance or abs(s["a_start"] - s["v_start"]) > q.sync_tolerance:
                rep.issues.append(f"audio/video out of sync (duration diff {drift:.2f}s, "
                                  f"start diff {abs(s['a_start'] - s['v_start']):.2f}s)")
    dur = s["v_dur"]
    if abs(dur - expected_duration) > 0.75:
        rep.issues.append(f"duration {dur:.2f}s, expected {expected_duration:.2f}s")
    if not q.min_seconds <= dur <= q.max_seconds:
        rep.issues.append(f"duration {dur:.1f}s outside allowed {q.min_seconds}-{q.max_seconds}s")

    # captions: analytic checks (render success is verified by FFmpeg itself)
    if cfg.captions.enabled and info.get("words"):
        if not info.get("caption_events"):
            rep.issues.append("captions enabled but none were generated")
        for m in info.get("caption_geometry_issues", []):
            rep.issues.append(m)
    ratio = info.get("face_clip_ratio")
    if ratio is not None:
        rep.checks["face_clip_ratio"] = round(ratio, 3)
        if ratio > q.max_face_clip_ratio:
            rep.issues.append(f"faces cut off in {ratio:.0%} of samples (limit {q.max_face_clip_ratio:.0%})")
        elif ratio > 0.05:
            rep.warnings.append(f"faces partly cut off in {ratio:.0%} of samples")

    _pixel_checks(p, layout, cfg, info, dur, rep)
    return rep


def _pixel_checks(p: Path, layout: Layout, cfg: Phase3Cfg, info: dict, dur: float, rep: QAReport) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        rep.warnings.append("opencv/numpy missing: pixel checks skipped")
        return
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        rep.issues.append("video cannot be opened for decoding")
        return
    q = cfg.qa
    n = 7
    times = [dur * (0.06 + 0.88 * i / (n - 1)) for i in range(n)]
    blanks, dark_bg, soft_fg, decoded = [], 0, 0, 0
    L = layout
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, f = cap.read()
            if not ok:
                continue
            decoded += 1
            if f.shape[0] != L.out_h or f.shape[1] != L.out_w:
                continue                       # wrong size is already reported as a critical issue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            panel = g[L.fg_y:L.fg_bottom, L.fg_x:L.fg_x + L.fg_w]
            if panel.size and (float(panel.mean()) < q.blank_luma or float(panel.std()) < q.blank_std):
                blanks.append(round(t, 2))
            bands = []
            if L.fg_y > 8:
                bands.append(g[:L.fg_y])
            if L.out_h - L.fg_bottom > 8:
                bands.append(g[L.fg_bottom:])
            if L.fg_x > 8:
                bands.append(g[L.fg_y:L.fg_bottom, :L.fg_x])
            if cfg.background.enabled and cfg.background.mode == "blur":
                if bands and all(float(np.median(b)) < q.min_bg_luma for b in bands):
                    dark_bg += 1
                if bands and panel.size:
                    bgv = max(_laplacian_var(cv2, b[: min(len(b), 400)]) for b in bands)
                    if bgv > 5.0 and _laplacian_var(cv2, panel) < bgv * 1.2:
                        soft_fg += 1
    finally:
        cap.release()
    rep.checks["frames_checked"] = decoded
    if decoded == 0:
        rep.issues.append("no frame could be decoded")
        return
    if blanks:
        rep.issues.append(f"blank/uniform foreground frame(s) at t={blanks}")
    if dark_bg == decoded:
        rep.issues.append("unexpected black borders (background is black in every sampled frame)")
    if soft_fg > decoded // 2:
        rep.warnings.append("foreground panel is not noticeably sharper than the background")
    times_c = info.get("caption_times") or []
    pl = info.get("placement")
    if cfg.captions.enabled and times_c and pl is not None:
        cap = cv2.VideoCapture(str(p))
        hit = 0
        try:
            for t in times_c[:5]:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, f = cap.read()
                if not ok:
                    continue
                band = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[pl.y_top:pl.y_bottom]
                if band.size and float((band > 190).mean()) > 0.0015:
                    hit += 1
        finally:
            cap.release()
        rep.checks["caption_pixel_hits"] = hit
        if hit == 0:
            rep.warnings.append("no caption pixels found in the caption band at sampled times")


def contact_sheet(path, layout: Layout, out_jpg, frames: int = 6) -> bool:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dur = total / fps if total else 0
    tiles = []
    try:
        for i in range(frames):
            cap.set(cv2.CAP_PROP_POS_MSEC, dur * (0.05 + 0.9 * i / max(frames - 1, 1)) * 1000)
            ok, f = cap.read()
            if not ok:
                continue
            s = 480 / f.shape[0]
            f = cv2.resize(f, (int(f.shape[1] * s), 480))
            x0, y0 = int(layout.fg_x * s), int(layout.fg_y * s)
            cv2.rectangle(f, (x0, y0), (x0 + int(layout.fg_w * s) - 1, y0 + int(layout.fg_h * s) - 1),
                          (0, 0, 255), 1)
            tiles.append(f)
    finally:
        cap.release()
    if not tiles:
        return False
    Path(out_jpg).parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_jpg), np.hstack(tiles)))
