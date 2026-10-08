from pathlib import Path

from src.utils.errors import FFmpegError


def build_input_args(src: Path, start: float, end: float) -> list[str]:
    """
    Input-side seek (-ss before -i) is fast, and because the clip is re-encoded in the
    same pass it is frame-accurate. No intermediate cut file is written.
    """
    if end <= start:
        raise FFmpegError(f"Invalid clip range: start={start}, end={end}")
    return ["-ss", f"{start:.3f}", "-i", str(Path(src).resolve()), "-t", f"{end - start:.3f}"]