from src.models.schemas import TranscriptSegment, WordTimestamp
from src.pipeline.config import BoundaryConfig

_STRIP = "\"')]”’"
_END = (".", "?", "!", "…")


def _nearest(value: float, points: list[float]) -> float:
    return min(points, key=lambda p: abs(p - value)) if points else value


def _words(segments: list[TranscriptSegment]) -> list[WordTimestamp]:
    return sorted((w for s in segments for w in s.words), key=lambda w: w.start)


def _sentence_edges(words: list[WordTimestamp], gap: float):
    """Word-level edges as (punct_starts, pause_starts, punct_ends, pause_ends).

    Punctuation edges are real sentence boundaries; pause edges (a gap >= `gap`) are weaker
    and only used when no punctuation edge is in range.
    """
    p_starts: list[float] = []
    z_starts: list[float] = []
    p_ends: list[float] = []
    z_ends: list[float] = []
    for i, w in enumerate(words):
        if i == 0 or words[i - 1].word.rstrip(_STRIP).endswith(_END):
            p_starts.append(w.start)
        elif w.start - words[i - 1].end >= gap:
            z_starts.append(w.start)
        last = i == len(words) - 1
        if w.word.rstrip(_STRIP).endswith(_END) or last:
            p_ends.append(w.end)
        elif words[i + 1].start - w.end >= gap:
            z_ends.append(w.end)
    return p_starts, z_starts, p_ends, z_ends


def _pick(value: float, strong: list[float], weak: list[float], tol: float) -> float | None:
    for pts in (strong, weak):
        near = [t for t in pts if abs(t - value) <= tol]
        if near:
            return _nearest(value, near)
    return None


def refine_boundaries(
    start: float,
    end: float,
    segments: list[TranscriptSegment],
    video_duration: float,
    cfg: BoundaryConfig,
    max_duration: float,
    audio=None,
) -> tuple[float, float]:
    """
    Snap recommended times to real sentence boundaries (word-level when word timestamps
    exist, else segment-level), add a small lead-in/tail, never bleed into neighbouring
    speech or exceed max_duration, and (with `audio`) hold a little longer after the last
    word so a laugh or reaction can land.
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

    words = _words(segments) if cfg.word_snap else []
    ends = seg_ends
    if words:
        ps, zs, pe, ze = _sentence_edges(words, cfg.pause_gap)
        ends = sorted(pe + ze)
        ws = _pick(new_start, ps, zs, cfg.snap_tolerance)
        we = _pick(new_end, pe, ze, cfg.snap_tolerance)
        ws = new_start if ws is None else ws
        we = new_end if we is None else we
        if we > ws + 1.0:
            new_start, new_end = ws, we

    if new_end - new_start > max_duration:
        limit = new_start + max_duration
        valid = [e for e in ends if new_start < e <= limit]
        new_end = max(valid) if valid else limit

    if words:
        prev_end = max((w.end for w in words if w.end <= new_start + 1e-3), default=0.0)
        next_start = min((w.start for w in words if w.start >= new_end - 1e-3), default=video_duration)
    else:
        prev_end = max((e for e in seg_ends if e <= new_start), default=0.0)
        next_start = min((s for s in seg_starts if s >= new_end), default=video_duration)

    out_start = max(0.0, prev_end, new_start - cfg.lead_in)
    out_end = min(video_duration, next_start, new_end + cfg.tail)
    if out_end <= out_start:
        out_start, out_end = max(0.0, new_start), min(video_duration, new_end)
        return round(out_start, 3), round(out_end, 3)

    if audio is not None and cfg.laugh_tail > 0:
        limit = min(video_duration, next_start, new_end + cfg.laugh_tail)
        held = audio.sustained_until(new_end, limit, cfg.laugh_threshold)
        if held > out_end:
            out_end = min(held + 0.15, limit)
    return round(out_start, 3), round(out_end, 3)
