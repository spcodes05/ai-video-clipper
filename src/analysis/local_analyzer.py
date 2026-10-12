import logging
import re
from collections import Counter

from src.detection.candidate_generator import FRAGMENT_STARTERS, LAUGH_TOKENS, WORD_RE
from src.detection.visual_analyzer import VisualAnalyzer
from src.models.schemas import CandidateClip, ClipAnalysis, Transcript, TranscriptSegment
from src.pipeline.config import AppConfig

log = logging.getLogger(__name__)

DANGLING_OPENERS = FRAGMENT_STARTERS | frozenset(
    {"so", "it", "he", "she", "they", "this", "these", "those", "him", "her", "them"}
)
STOPWORDS = frozenset(
    """the a an and or but of to in on at for with is are was were be been it this that these those
    i you he she we they my your his her our their not no so as if then than there here have has had
    do does did will would can could just like""".split()
)


def _first_token(text: str) -> str:
    m = WORD_RE.search(text)
    return m.group(0).lower() if m else ""


def _opens_badly(seg: TranscriptSegment) -> bool:
    text = seg.text.strip()
    return bool(text) and (text[0].islower() or _first_token(text) in FRAGMENT_STARTERS)


def _closes(seg: TranscriptSegment) -> bool:
    return seg.text.strip().endswith((".", "?", "!"))


def _trim(segs: list[TranscriptSegment], min_duration: float) -> list[TranscriptSegment]:
    """Drop leading fragments / trailing unfinished sentences while staying >= min_duration."""
    lo, hi = 0, len(segs) - 1
    while hi - lo >= 1 and _opens_badly(segs[lo]) and segs[hi].end - segs[lo + 1].start >= min_duration:
        lo += 1
    while hi - lo >= 1 and not _closes(segs[hi]) and segs[hi - 1].end - segs[lo].start >= min_duration:
        hi -= 1
    return segs[lo : hi + 1]


def _title(text: str) -> str:
    sentences = [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s]
    if not sentences:
        return "Untitled clip"
    pick = next((s for s in sentences if s.endswith("?")), sentences[0])
    words = pick.split()
    title = " ".join(words[:12]).rstrip(",;:- ")
    return title + ("..." if len(words) > 12 else "")


class LocalAnalyzer:
    """
    Deterministic, LLM-free replacement for the hosted analyzer.
    Produces the same ClipAnalysis objects, from weak local signals only.
    """

    def __init__(self, cfg: AppConfig, visual: VisualAnalyzer | None = None):
        self.cfg = cfg
        self.visual = visual
        self.ideal = cfg.profiles[cfg.analysis.profile].ideal_duration

    def analyze(self, candidates: list[CandidateClip], transcript: Transcript) -> list[ClipAnalysis]:
        out: list[ClipAnalysis] = []
        for c in candidates:
            try:
                out.append(self._analyze_one(c, transcript))
            except Exception as exc:  # noqa: BLE001
                log.warning("Skipping %s: local analysis failed (%s)", c.id, exc)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _segments(c: CandidateClip, transcript: Transcript) -> list[TranscriptSegment]:
        inside = [s for s in transcript.segments if s.start >= c.start - 1e-3 and s.end <= c.end + 1e-3]
        return inside or [s for s in transcript.segments if s.end > c.start and s.start < c.end]

    def _analyze_one(self, c: CandidateClip, transcript: Transcript) -> ClipAnalysis:
        segs = self._segments(c, transcript)
        segs = _trim(segs, self.cfg.candidates.min_duration) if segs else segs
        if segs:
            start, end = segs[0].start, segs[-1].end
            text = " ".join(s.text.strip() for s in segs)
        else:
            start, end, text = c.start, c.end, c.transcript

        tokens = [t.lower() for t in WORD_RE.findall(text)]
        dur = max(end - start, 0.01)
        g = c.signals.get

        opening_clean = 0.0 if segs and _opens_badly(segs[0]) else 1.0
        ending_complete = 1.0 if segs and _closes(segs[-1]) else 0.3
        dangling = 1.0 if tokens and tokens[0] in DANGLING_OPENERS else 0.0
        context = 0.4 * opening_clean + 0.3 * ending_complete + 0.3 * (1.0 - dangling)

        content = [t for t in tokens if t not in STOPWORDS and len(t) > 3]
        repetition = 0.0
        if len(content) >= 5:
            counts = Counter(content)
            repetition = min(2.0 * sum(n for n in counts.values() if n >= 2) / len(content), 1.0)

        exclaim = min(text.count("!") / 3.0, 1.0)
        laugh = 1.0 if any(t in LAUGH_TOKENS for t in tokens) else 0.0
        fit = max(0.0, 1.0 - abs(dur - self.ideal) / self.ideal)

        punch, emotion, qa = g("punchline", 0.0), g("emotion", 0.0), g("qa", 0.0)
        audio, density, scene = g("audio", 0.0), g("density", 0.0), g("scene", 0.0)

        motion = self.visual.score(start, end) if self.visual else None
        visual_interest = 0.5 if motion is None else 0.6 * motion + 0.4 * scene

        s = {
            "hook": 0.7 * g("hook", 0.0) + 0.3 * opening_clean,
            "humor": 0.45 * punch + 0.25 * laugh + 0.15 * audio + 0.15 * qa,
            "surprise": 0.4 * audio + 0.3 * emotion + 0.3 * exclaim,
            "emotion": 0.7 * emotion + 0.3 * audio,
            "shareability": 0.30 * density + 0.25 * punch + 0.20 * emotion + 0.15 * repetition + 0.10 * fit,
            "visual_interest": visual_interest,
            "context_completeness": 0.8 * context + 0.2 * fit,
            "payoff": 0.6 * punch + 0.2 * ending_complete + 0.2 * qa,
        }
        s["overall"] = (
            0.30 * s["hook"] + 0.25 * s["payoff"] + 0.25 * s["context_completeness"] + 0.20 * s["shareability"]
        )
        needs_context = context < 0.65

        ranked = sorted(((k, v) for k, v in s.items() if k != "overall"), key=lambda kv: kv[1], reverse=True)
        top = ", ".join(f"{k} {v * 100:.0f}" for k, v in ranked[:3])
        low = f"{ranked[-1][0]} {ranked[-1][1] * 100:.0f}"
        notes = ""
        if motion is None:
            notes += " Visual motion unavailable (neutral 50 used)."
        if needs_context:
            notes += " Clip may need more context."
        reasoning = (
            f"Local heuristic (profile: {self.cfg.analysis.profile}). Strongest: {top}. Weakest: {low}.{notes} "
            "Weak signals only; not a virality prediction."
        )

        return ClipAnalysis(
            candidate_id=c.id,
            recommended_start=round(start, 3),
            recommended_end=round(end, 3),
            title=_title(text),
            reasoning=reasoning,
            requires_too_much_context=needs_context,
            **{k: round(v * 100.0, 1) for k, v in s.items()},
        )