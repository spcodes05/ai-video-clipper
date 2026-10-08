from src.models.schemas import TranscriptSegment
from src.pipeline.config import BoundaryConfig


def _nearest(value: float, points: list[float]) -> float:
    return min(points, key=lambda p: abs(p - value)) if points else value


def refine_boundaries(
    start: float,
    end: float,
    segments: list[TranscriptSegment],
    video_duration: float,
    cfg: BoundaryConfig,
    max_duration: float,
) -> tuple[float, float]:
    """
    Snap LLM-recommended times to real sentence boundaries, add a small lead-in/tail,
    and never bleed into neighbouring speech or exceed max_duration.
    """
    seg_starts = [s.start for s in segments]
    seg_ends = [s.end for s in segments]

    new_start = _nearest(start, seg_starts)
    if abs(new_start - start) > cfg.snap_tolerance:
        new_start = start
    new_end = _nearest(end, seg_ends)
    if abs(new_end - end) > cfg.snap_tolerance:
        new_end = end
    if new_end <= new_start + 1.0:
        new_start, new_end = start, end

    if new_end - new_start > max_duration:
        limit = new_start + max_duration
        valid = [e for e in seg_ends if new_start < e <= limit]
        new_end = max(valid) if valid else limit

    prev_end = max((e for e in seg_ends if e <= new_start), default=0.0)
    next_start = min((s for s in seg_starts if s >= new_end), default=video_duration)

    out_start = max(0.0, prev_end, new_start - cfg.lead_in)
    out_end = min(video_duration, next_start, new_end + cfg.tail)
    if out_end <= out_start:
        out_start, out_end = max(0.0, new_start), min(video_duration, new_end)
    return round(out_start, 3), round(out_end, 3)