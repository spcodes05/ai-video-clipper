import json
import logging
import shutil
import subprocess
from pathlib import Path

from src.models.schemas import VideoMetadata
from src.utils.errors import FFmpegError, VideoInputError

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv"}


def check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise FFmpegError(
                f"'{tool}' was not found on PATH. Install FFmpeg (see README) "
                "and restart your terminal."
            )


def run_ffmpeg(args: list[str], cwd: str | None = None) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    log.debug("ffmpeg cmd: %s (cwd=%s)", " ".join(cmd), cwd)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if proc.returncode != 0:
        raise FFmpegError(f"FFmpeg failed (exit {proc.returncode}): {proc.stderr.strip()[-1500:]}")


def _parse_fps(value: str | None) -> float:
    try:
        if value and "/" in value:
            num, den = value.split("/")
            return float(num) / float(den) if float(den) else 0.0
        return float(value) if value else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(path: Path) -> VideoMetadata:
    path = Path(path)
    if not path.exists():
        raise VideoInputError(f"Input file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoInputError(
            f"Unsupported format '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise VideoInputError(f"Video appears corrupted or unreadable: {proc.stderr.strip()[-500:]}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise VideoInputError("ffprobe returned unreadable output.") from exc

    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise VideoInputError("No video stream found in the file.")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)

    return VideoMetadata(
        path=str(path),
        duration=duration,
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        fps=_parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        codec=video.get("codec_name", "unknown"),
        has_audio=has_audio,
    )


def validate_video(meta: VideoMetadata, min_duration: float) -> None:
    if not meta.has_audio:
        raise VideoInputError("The video has no audio track, so it cannot be transcribed.")
    if meta.duration <= 0 or meta.width <= 0 or meta.height <= 0:
        raise VideoInputError("Video metadata is invalid (zero duration or dimensions). File may be corrupted.")
    if meta.duration < min_duration:
        raise VideoInputError(
            f"Video is only {meta.duration:.1f}s long, shorter than the minimum clip duration ({min_duration}s)."
        )


def extract_audio(video_path: Path, wav_path: Path) -> None:
    """16 kHz mono PCM WAV, used for both Whisper and audio analysis."""
    run_ffmpeg(["-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)])