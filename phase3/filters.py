"""FFmpeg filter_complex + encoder selection for the Phase 3 composition.

Layout:  [source] -> (optional keep-segment trim/concat) -> blurred 9:16 background
                                                          + sharp square panel (crop | contain)
                                                          -> overlay -> captions/hook -> yuv420p
"""
from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from phase2.crop import crop_size
from phase2.ffmpeg_utils import FFMPEG

from .config import BackgroundCfg, OutputCfg
from .layout import Layout, contain_size
from .timeline import Seg
from .zoom import ZoomEvent, zoom_expression


@dataclass
class GraphSpec:
    layout: Layout
    src_w: int
    src_h: int
    fps: int
    mode: str                                   # crop | contain
    crop_w: int = 0
    crop_h: int = 0
    crop_x_expr: str = "0"
    crop_y: int = 0
    keep: Optional[Sequence[Seg]] = None
    zoom_events: Sequence[ZoomEvent] = ()
    zoom_trans: float = 0.35
    zoom_normal: float = 1.0
    reaction_windows: Sequence[Seg] = ()        # output time; crop mode only -> wide group shot
    bg: BackgroundCfg = field(default_factory=BackgroundCfg)
    panel_blur: bool = True
    panel_blur_strength: float = 20.0
    ass_name: Optional[str] = None
    fonts_dir: Optional[str] = None
    has_audio: bool = True
    crossfade_ms: int = 20
    loudnorm: bool = True
    target_lufs: float = -16.0


def _even(v: float, lo: int = 2) -> int:
    v = int(v)
    return max(lo, v - v % 2)


def _between(windows: Sequence[Seg]) -> str:
    return "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in windows)


def _blur_chain(w: int, h: int, strength: float, darken: float, sat: float) -> str:
    sw, sh = _even(w / 8), _even(h / 8)
    sigma = max(strength / 8.0, 0.3)
    return (f"scale={sw}:{sh}:flags=bilinear,gblur=sigma={sigma:.2f},scale={w}:{h}:flags=bicubic,"
            f"eq=brightness={-darken:.3f}:saturation={sat:.3f}")


def build_video_graph(g: GraphSpec) -> str:
    L = g.layout
    keep = list(g.keep) if g.keep else None
    parts: List[str] = []
    if keep and len(keep) > 1:
        n = len(keep)
        parts.append(f"[0:v]fps={g.fps},split={n}" + "".join(f"[s{i}]" for i in range(n)))
        for i, (a, b) in enumerate(keep):
            parts.append(f"[s{i}]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[t{i}]")
        parts.append("".join(f"[t{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vc]")
    elif keep:
        a, b = keep[0]
        parts.append(f"[0:v]fps={g.fps},trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[vc]")
    else:
        parts.append(f"[0:v]fps={g.fps},setpts=PTS-STARTPTS[vc]")

    need_crop = g.mode == "crop"
    need_contain = g.mode == "contain" or (need_crop and bool(g.reaction_windows))
    users = ["bg"] + (["c"] if need_crop else []) + (["w"] if need_contain else [])
    if len(users) == 1:
        parts.append("[vc]null[bg_in]")
        names = {"bg": "bg_in"}
    else:
        parts.append("[vc]split=%d" % len(users) + "".join(f"[{u}_in]" for u in users))
        names = {u: f"{u}_in" for u in users}

    # ---- blurred canvas background (derived from the source, static centre region)
    if g.bg.enabled and g.bg.mode == "blur":
        bcw, bch = crop_size(g.src_w, g.src_h, L.out_w, L.out_h)
        parts.append(f"[{names['bg']}]crop={bcw}:{bch}:(iw-{bcw})/2:(ih-{bch})/2,"
                     + _blur_chain(L.out_w, L.out_h, g.bg.blur_strength, g.bg.darken, g.bg.saturation)
                     + ",setsar=1[bg]")
    else:
        parts.append(f"[{names['bg']}]scale={L.out_w}:{L.out_h},"
                     f"drawbox=x=0:y=0:w=iw:h=ih:color=0x{g.bg.color}:t=fill,setsar=1[bg]")

    # ---- sharp foreground panel
    if need_crop:
        chain = (f"[{names['c']}]crop={g.crop_w}:{g.crop_h}:x='{g.crop_x_expr}':y={g.crop_y},"
                 f"scale={L.fg_w}:{L.fg_h}:flags=lanczos,setsar=1")
        if g.zoom_events:
            z = zoom_expression(g.zoom_events, g.zoom_trans, g.zoom_normal)
            chain += (f",scale=w='trunc({L.fg_w}*({z})/2)*2':h='trunc({L.fg_h}*({z})/2)*2'"
                      f":eval=frame:flags=bicubic,crop={L.fg_w}:{L.fg_h}")
        parts.append(chain + "[fgc]")
    if need_contain:
        vw, vh = contain_size(g.src_w, g.src_h, L.fg_w, L.fg_h)
        src = names["w"]
        if g.panel_blur:
            pcw, pch = crop_size(g.src_w, g.src_h, L.fg_w, L.fg_h)
            parts.append(f"[{src}]split=2[wb][wv]")
            parts.append(f"[wb]crop={pcw}:{pch}:(iw-{pcw})/2:(ih-{pch})/2,"
                         + _blur_chain(L.fg_w, L.fg_h, g.panel_blur_strength, 0.18, 1.05) + ",setsar=1[pbg]")
            parts.append(f"[wv]scale={vw}:{vh}:flags=lanczos,setsar=1[pv]")
            parts.append("[pbg][pv]overlay=(W-w)/2:(H-h)/2[fgw]")
        else:
            parts.append(f"[{src}]scale={vw}:{vh}:flags=lanczos,setsar=1,"
                         f"pad={L.fg_w}:{L.fg_h}:(ow-iw)/2:(oh-ih)/2:color=black[fgw]")
    if need_crop and g.reaction_windows:
        parts.append(f"[fgc][fgw]overlay=0:0:enable='{_between(g.reaction_windows)}'[fgm]")
        fg = "fgm"
    else:
        fg = "fgc" if need_crop else "fgw"

    tail = f"[bg][{fg}]overlay={L.fg_x}:{L.fg_y},setsar=1"
    if g.ass_name:
        tail += f",subtitles=filename={g.ass_name}"
        if g.fonts_dir:
            tail += f":fontsdir={g.fonts_dir}"
    parts.append(tail + ",format=yuv420p[vout]")
    return ";".join(parts)


def build_audio_graph(g: GraphSpec) -> Optional[str]:
    if not g.has_audio:
        return None
    keep = list(g.keep) if g.keep else None
    parts: List[str] = []
    tail = []
    if g.loudnorm:
        tail.append(f"loudnorm=I={g.target_lufs}:TP=-1.5:LRA=11")
    tail += ["aresample=48000", "alimiter=limit=0.95"]
    if keep and len(keep) > 1:
        n = len(keep)
        cf = g.crossfade_ms / 1000.0
        parts.append(f"[0:a]asplit={n}" + "".join(f"[as{i}]" for i in range(n)))
        for i, (a, b) in enumerate(keep):
            d = min(cf, (b - a) / 3.0)
            f = f"[as{i}]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS"
            if d > 0.002:
                if i > 0:
                    f += f",afade=t=in:st=0:d={d:.3f}"
                if i < n - 1:
                    f += f",afade=t=out:st={b - a - d:.3f}:d={d:.3f}"
            parts.append(f + f"[ta{i}]")
        parts.append("".join(f"[ta{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1,"
                     + ",".join(tail) + "[aout]")
    elif keep:
        a, b = keep[0]
        parts.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,"
                     + ",".join(tail) + "[aout]")
    else:
        parts.append("[0:a]" + ",".join(tail) + "[aout]")
    return ";".join(parts)


def build_filter_complex(g: GraphSpec) -> str:
    v = build_video_graph(g)
    a = build_audio_graph(g)
    return v + (";" + a if a else "")


# --------------------------------------------------------------------------- encoders
_HW = [("nvenc", "h264_nvenc"), ("qsv", "h264_qsv"), ("amf", "h264_amf"),
       ("videotoolbox", "h264_videotoolbox")]


@functools.lru_cache(maxsize=None)
def _encoder_works(codec: str) -> bool:
    try:
        p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=256x256:d=0.2:r=30", "-c:v", codec, "-f", "null", "-"],
                           capture_output=True, timeout=25, check=False)
        return p.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def resolve_encoder(requested: str) -> str:
    """-> libx264 | h264_nvenc | h264_qsv | h264_amf | h264_videotoolbox (verified by a test encode)."""
    if requested == "cpu":
        return "libx264"
    names = dict(_HW)
    if requested in names:
        return names[requested] if _encoder_works(names[requested]) else "libx264"
    for _, codec in _HW:                                   # auto
        if _encoder_works(codec):
            return codec
    return "libx264"


def video_codec_args(codec: str, out: OutputCfg) -> List[str]:
    if codec == "h264_nvenc":
        a = ["-c:v", codec, "-preset", "p5", "-rc", "vbr", "-cq", str(out.crf + 2), "-b:v", "0"]
    elif codec == "h264_qsv":
        a = ["-c:v", codec, "-global_quality", str(out.crf), "-preset", "medium"]
    elif codec == "h264_amf":
        a = ["-c:v", codec, "-quality", "balanced", "-rc", "cqp",
             "-qp_i", str(out.crf), "-qp_p", str(out.crf)]
    elif codec == "h264_videotoolbox":
        a = ["-c:v", codec, "-q:v", "65"]
    else:
        a = ["-c:v", "libx264", "-preset", out.preset, "-crf", str(out.crf)]
    return a + ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]


MAX_INLINE_GRAPH = 24000      # Windows command lines are limited to 32,767 characters


def build_command(src, out_path, start: float, duration: float, graph: str, has_audio: bool,
                  codec_args: Sequence[str], audio_bitrate: str, script_path: Optional[str] = None
                  ) -> List[str]:
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}",
           "-i", str(src), "-t", f"{duration:.3f}"]
    if script_path:
        cmd += ["-filter_complex_script", script_path]
    else:
        cmd += ["-filter_complex", graph]
    cmd += ["-map", "[vout]"]
    if has_audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000"]
    else:
        cmd += ["-an"]
    return cmd + list(codec_args) + [str(out_path)]
