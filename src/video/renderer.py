import logging
from pathlib import Path

from src.captions.ass_renderer import get_style, write_ass
from src.captions.subtitle_generator import group_words, words_for_clip
from src.models.schemas import RankedClip, RenderedClip, Transcript, VideoMetadata
from src.pipeline.config import CaptionConfig, VideoConfig
from src.utils.ffmpeg import run_ffmpeg
from src.video.cutter import build_input_args
from src.video.reframer import Reframer

log = logging.getLogger(__name__)


class ClipRenderer:
    """Cut + 9:16 reframe + burn captions + encode in ONE FFmpeg pass (no intermediates)."""

    def __init__(self, video_cfg: VideoConfig, caption_cfg: CaptionConfig, reframer: Reframer, captions_dir: Path):
        self.v = video_cfg
        self.c = caption_cfg
        self.reframer = reframer
        self.captions_dir = Path(captions_dir)
        self.reframer.target_w, self.reframer.target_h = video_cfg.width, video_cfg.height

    def render(
        self,
        src_path: Path,
        meta: VideoMetadata,
        clip: RankedClip,
        transcript: Transcript,
        out_path: Path,
    ) -> RenderedClip:
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
            ass_path = self.captions_dir / f"{out_path.stem}.ass"
            write_ass(lines, ass_path, get_style(self.c.style), self.v.width, self.v.height)
            # Relative filename + cwd avoids Windows drive-letter escaping problems in filters.
            filters.append(f"ass={ass_path.name}")
            cwd = str(ass_path.parent)
        else:
            log.warning("No words found for %s; rendering without captions.", out_path.name)
        filters.append("format=yuv420p")

        args = [
            *build_input_args(src_path, start, end),
            "-map", "0:v:0",
            "-map", "0:a:0",
            "-vf", ",".join(filters),
            "-c:v", "libx264",
            "-preset", self.v.preset,
            "-crf", str(self.v.crf),
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