def format_timestamp(seconds: float) -> str:
    """HH:MM:SS.mmm"""
    total_ms = int(round(max(seconds, 0.0) * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def overlap_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    """Overlap as a fraction of the shorter interval (0..1)."""
    shortest = min(a1 - a0, b1 - b0)
    if shortest <= 0:
        return 0.0
    return overlap_seconds(a0, a1, b0, b1) / shortest