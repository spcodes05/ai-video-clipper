from dataclasses import dataclass, field

from src.models.schemas import TranscriptSegment, WordTimestamp

SENTENCE_END = (".", "?", "!")


@dataclass
class CaptionLine:
    start: float
    end: float
    words: list[WordTimestamp] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


def _synthesize_words(seg: TranscriptSegment) -> list[WordTimestamp]:
    """Fallback when a segment has no word timestamps: spread words by character length."""
    tokens = seg.text.split()
    if not tokens:
        return []
    weights = [max(len(t), 1) for t in tokens]
    scale = max(seg.end - seg.start, 0.01) / sum(weights)
    out, cursor = [], seg.start
    for tok, wt in zip(tokens, weights):
        out.append(WordTimestamp(word=tok, start=cursor, end=cursor + wt * scale))
        cursor += wt * scale
    return out


def words_for_clip(segments: list[TranscriptSegment], clip_start: float, clip_end: float) -> list[WordTimestamp]:
    """Words inside the clip, with times made relative to the clip start."""
    duration = clip_end - clip_start
    out: list[WordTimestamp] = []
    for seg in segments:
        if seg.end <= clip_start or seg.start >= clip_end:
            continue
        for w in seg.words or _synthesize_words(seg):
            mid = (w.start + w.end) / 2
            if mid < clip_start or mid > clip_end:
                continue
            token = w.word.strip()
            if not token:
                continue
            start = max(0.0, w.start - clip_start)
            end = min(duration, max(w.end - clip_start, start + 0.05))
            out.append(WordTimestamp(word=token, start=start, end=end, probability=w.probability))
    return out


def group_words(
    words: list[WordTimestamp],
    clip_duration: float,
    max_words: int = 3,
    max_chars: int = 20,
    max_gap: float = 0.7,
    hold: float = 0.35,
) -> list[CaptionLine]:
    lines: list[CaptionLine] = []
    cur: list[WordTimestamp] = []

    def flush() -> None:
        if cur:
            lines.append(CaptionLine(start=cur[0].start, end=cur[-1].end, words=list(cur)))
            cur.clear()

    for w in words:
        if cur:
            too_long = len(" ".join(x.word for x in cur + [w])) > max_chars
            if (
                len(cur) >= max_words
                or too_long
                or (w.start - cur[-1].end) > max_gap
                or cur[-1].word.endswith(SENTENCE_END)
            ):
                flush()
        cur.append(w)
    flush()

    # Hold each line briefly so captions don't flicker, without overlapping the next line.
    for k, line in enumerate(lines):
        nxt = lines[k + 1].start if k + 1 < len(lines) else clip_duration
        line.end = max(min(line.end + hold, nxt), line.start + 0.05)
    return lines