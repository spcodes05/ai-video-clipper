import logging
from pathlib import Path
from typing import Sequence

from src.captions.ass_renderer import get_style, write_ass
from src.captions.subtitle_generator import group_words, words_for_clip
from src.models.schemas import RankedClip, RenderedClip, Transcript, VideoMetadata
from src.pipeline.config import CaptionConfig, VideoConfig
from src.utils.errors import FFmpegError
from src.utils.ffmpeg import run_ffmpeg
from src.video.cutter import build_input_args
from src.video.reframer import Reframer

log = logging.getLogger(__name__)


class ClipRenderer:
    """Cut + 9:16 reframe + burn captions + encode in ONE FFmpeg pass (no intermediates).

    Boundaries come from clip.final_start/final_end, set once by
    src.analysis.boundaries.refine_boundaries before selection; the renderer never
    changes them. When `phase2_cfg` is given (see src.pipeline.phase2_bridge) the clip
    is rendered by phase2.pipeline.render_clip (face tracking + dynamic crop, phase2
    captions, audio filters, optional silence removal). Otherwise the legacy path below
    is used unchanged. A Phase 2 failure raises FFmpegError for that clip; there is no
    silent fallback to the legacy renderer.
    """

    def __init__(
        self,
        video_cfg: VideoConfig,
        caption_cfg: CaptionConfig,
        reframer: Reframer,
        captions_dir: Path,
        phase2_cfg=None,
    ):
        self.v = video_cfg
        self.c = caption_cfg
        self.reframer = reframer
        self.captions_dir = Path(captions_dir)
        self.reframer.target_w, self.reframer.target_h = video_cfg.width, video_cfg.height
        self.phase2_cfg = phase2_cfg

    def render(
        self,
        src_path: Path,
        meta: VideoMetadata,
        clip: RankedClip,
        transcript: Transcript,
        out_path: Path,
        scene_cuts: Sequence[float] = (),
    ) -> RenderedClip:
        if self.phase2_cfg is not None:
            return self._render_phase2(src_path, clip, transcript, out_path, scene_cuts)
        return self._render_legacy(src_path, meta, clip, transcript, out_path)

    # ------------------------------------------------------------------ phase 2
    def _render_phase2(self, src_path, clip, transcript, out_path, scene_cuts) -> RenderedClip:
        from phase2.boundaries import Word
        from phase2.pipeline import render_clip

        start, end = float(clip.final_start), float(clip.final_end)
        if end <= start:
            raise FFmpegError(f"Invalid clip range: start={start}, end={end}")
        out_path = Path(out_path)

        # words_for_clip returns clip-relative times (and synthesizes words when a segment
        # has none); convert back to SOURCE time, which is what render_clip expects.
        words = [
            Word(start + w.start, start + w.end, w.word)
            for w in words_for_clip(transcript.segments, start, end)
        ]
        speech = [
            (float(s.start), float(s.end))
            for s in transcript.segments
            if s.end > start and s.start < end
        ]
        if not words:
            log.warning("No words found for %s; Phase 2 will render without captions.", out_path.name)

        try:
            result = render_clip(
                src_path,
                out_path,
                start,
                end,
                words,
                self.phase2_cfg,
                speech_segments=speech,
                scene_cuts=list(scene_cuts),
                debug_dir=self.captions_dir.parent / "phase2",
                log=log,
            )
        except FFmpegError:
            raise
        except Exception as exc:  # noqa: BLE001 - report per clip, keep the pipeline alive
            log.exception("Phase 2 render failed for %s", out_path.name)
            raise FFmpegError(f"Phase 2 render failed for {out_path.name}: {exc}") from exc

        if not result.captions_ok:
            log.warning("%s was rendered with the plain caption fallback style.", out_path.name)
        log.info(
            "Phase 2 rendered %s: %.2fs-%.2fs, face tracking applied=%s, captions styled=%s",
            out_path.name, start, end, result.used_face_tracking, result.captions_ok,
        )
        return RenderedClip(
            clip_number=clip.clip_number or 0,
            path=str(out_path),
            start=start,
            end=end,
            duration=round(result.duration, 2),
            viral_score=clip.viral_score,
            title=clip.analysis.title,
            reasoning=clip.analysis.reasoning,
        )

    # ------------------------------------------------------------------ legacy
    def _render_legacy(self, src_path, meta, clip, transcript, out_path) -> RenderedClip:
        start, end = clip.final_start, clip.final_end
        duration = end - start

        words = words_for_clip(transcript.segments, start, end)
        lines = group_words(words, duration, self.c.words_per_line, self.c.max_chars)

        filters = [
            self.reframer.crop_filter(meta, start, end),
            f"scale={self.v.width}:{self.v.height}:flags=lanczos",
        ]
        if meta.fps > 30.5:
            filters.append("fps=30")

        cwd = None
        if lines:
            self.captions_dir.mkdir(parents=True, exist_ok=True)
            ass_path = self.captions_dir / f"{Path(out_path).stem}.ass"
            write_ass(lines, ass_path, get_style(self.c.style), self.v.width, self.v.height)
            # Relative filename + cwd avoids Windows drive-letter escaping problems in filters.
            filters.append(f"ass={ass_path.name}")
            cwd = str(ass_path.parent)
        else:
            log.warning("No words found for %s; rendering without captions.", Path(out_path).name)
        filters.append("format=yuv420p")

        args = [
            *build_input_args(src_path, start, end),
            "-map", "0:v:0",
            "-map", "0:a:0",
            "-vf", ",".join(filters),
            "-c:v", "libx264",
            "-preset", self.v.preset,
            "-crf", str(self.v.crf),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar", "48000",
            "-c:a", "aac",
            "-b:a", self.v.audio_bitrate,
            "-movflags", "+faststart",
            str(Path(out_path).resolve()),
        ]
        run_ffmpeg(args, cwd=cwd)

        return RenderedClip(
            clip_number=clip.clip_number or 0,
            path=str(out_path),
            start=start,
            end=end,
            duration=round(duration, 2),
            viral_score=clip.viral_score,
            title=clip.analysis.title,
            reasoning=clip.analysis.reasoning,
        )
