
import logging
from pathlib import Path

import yaml

from src.captions.ass_renderer import get_style, write_ass
from src.captions.subtitle_generator import group_words, words_for_clip
from src.models.schemas import (
    RankedClip,
    RenderedClip,
    Transcript,
    VideoMetadata,
)
from src.pipeline.config import CaptionConfig, VideoConfig
from src.utils.ffmpeg import run_ffmpeg
from src.video.cutter import build_input_args
from src.video.reframer import Reframer

# Phase 2 is optional so the original renderer can still work
# if the Phase 2 modules are unavailable.
try:
    from phase2.config import Phase2Cfg
    from phase2.boundaries import Word, refine_boundaries

    PHASE2_AVAILABLE = True
except ImportError:
    Phase2Cfg = None
    Word = None
    refine_boundaries = None
    PHASE2_AVAILABLE = False


log = logging.getLogger(__name__)


class ClipRenderer:
    """
    Cut, reframe to 9:16, burn captions, normalize audio,
    and encode in one FFmpeg pass.

    Phase 2 boundary refinement is optional. If it cannot run,
    the renderer retains the original clip boundaries.
    """

    def __init__(
        self,
        video_cfg: VideoConfig,
        caption_cfg: CaptionConfig,
        reframer: Reframer,
        captions_dir: Path,
    ):
        self.v = video_cfg
        self.c = caption_cfg
        self.reframer = reframer
        self.captions_dir = Path(captions_dir)

        self.reframer.target_w = video_cfg.width
        self.reframer.target_h = video_cfg.height

        # Resolve config relative to the project working directory.
        self.config_path = Path.cwd() / "config.yaml"

        self._phase2_cfg = None
        self._phase2_initialized = False

    def _load_phase2_config(self):
        """Load Phase 2 configuration once, when available."""
        if self._phase2_initialized:
            return self._phase2_cfg

        self._phase2_initialized = True

        if not PHASE2_AVAILABLE:
            log.info("Phase 2 modules unavailable; using original boundaries.")
            return None

        try:
            if not self.config_path.is_file():
                log.warning(
                    "config.yaml not found at %s; skipping Phase 2 refinement.",
                    self.config_path,
                )
                return None

            with self.config_path.open("r", encoding="utf-8") as file:
                raw_cfg = yaml.safe_load(file) or {}

            self._phase2_cfg = Phase2Cfg.from_config(raw_cfg)
            return self._phase2_cfg

        except Exception:
            log.exception(
                "Could not initialize Phase 2 configuration. "
                "Using original clip boundaries."
            )
            return None

    def _refine_boundaries(
        self,
        transcript: Transcript,
        start: float,
        end: float,
    ) -> tuple[float, float]:
        """
        Refine a clip's boundaries using available timestamped words.

        The words are expected to retain source-video timestamps.
        If refinement is unavailable or invalid, return the original
        boundaries.
        """
        original = (start, end)

        if end <= start:
            log.warning("Invalid original clip boundaries: %s - %s", start, end)
            return original

        phase2_cfg = self._load_phase2_config()

        if phase2_cfg is None:
            return original

        try:
            # This extracts the words already available to the renderer.
            # It does not require a second Whisper transcription.
            transcript_words = words_for_clip(
                transcript.segments,
                start,
                end,
            )

            phase2_words = [
                Word(
                    float(word.start),
                    float(word.end),
                    str(word.word),
                )
                for word in transcript_words
                if (
                    word.start is not None
                    and word.end is not None
                    and word.end > word.start
                    and getattr(word, "word", None)
                )
            ]

            if not phase2_words:
                log.warning(
                    "No timestamped words available for Phase 2 refinement."
                )
                return original

            candidate = refine_boundaries(
                phase2_words,
                start,
                end,
                phase2_cfg.clip,
            )

            if candidate is None:
                log.info("Phase 2 returned no candidate; keeping original bounds.")
                return original

            refined_start = float(candidate.start)
            refined_end = float(candidate.end)

            if (
                refined_start < 0
                or refined_end <= refined_start
                or refined_start < start - 5.0
                or refined_end > end + 5.0
            ):
                log.warning(
                    "Rejected invalid or unexpected Phase 2 boundaries: "
                    "%.2f - %.2f; original: %.2f - %.2f",
                    refined_start,
                    refined_end,
                    start,
                    end,
                )
                return original

            log.info(
                "Phase 2 boundaries: %.2f-%.2f -> %.2f-%.2f",
                start,
                end,
                refined_start,
                refined_end,
            )

            return refined_start, refined_end

        except Exception:
            log.exception(
                "Phase 2 boundary refinement failed. "
                "Keeping original boundaries for this clip."
            )
            return original

    def render(
        self,
        src_path: Path,
        meta: VideoMetadata,
        clip: RankedClip,
        transcript: Transcript,
        out_path: Path,
    ) -> RenderedClip:
        # Preserve the original ranked clip and refine only local
        # rendering boundaries.
        original_start = float(clip.final_start)
        original_end = float(clip.final_end)

        start, end = self._refine_boundaries(
            transcript,
            original_start,
            original_end,
        )

        duration = end - start

        if duration <= 0:
            raise ValueError(
                f"Invalid clip duration: start={start}, end={end}"
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract caption words using the final rendering boundaries.
        words = words_for_clip(
            transcript.segments,
            start,
            end,
        )

        lines = group_words(
            words,
            duration,
            self.c.words_per_line,
            self.c.max_chars,
        )

        filters = [
            self.reframer.crop_filter(meta, start, end),
            f"scale={self.v.width}:{self.v.height}:flags=lanczos",
        ]

        # Convert high-frame-rate sources to 30 FPS.
        if meta.fps > 30.5:
            filters.append("fps=30")

        cwd = None

        if lines:
            self.captions_dir.mkdir(parents=True, exist_ok=True)

            ass_path = self.captions_dir / f"{out_path.stem}.ass"

            write_ass(
                lines,
                ass_path,
                get_style(self.c.style),
                self.v.width,
                self.v.height,
            )

            # Use a relative ASS filename with the captions directory
            # as the working directory to avoid Windows path escaping.
            filters.append(f"ass={ass_path.name}")
            cwd = str(ass_path.parent)

        else:
            log.warning(
                "No caption words found for %s; rendering without captions.",
                out_path.name,
            )

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
            str(out_path.resolve()),
        ]

        log.info(
            "Rendering %s: %.2fs-%.2fs (%.2fs)",
            out_path.name,
            start,
            end,
            duration,
        )

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
