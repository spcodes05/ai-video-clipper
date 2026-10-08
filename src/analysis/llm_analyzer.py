import json
import logging
import re

from pydantic import ValidationError

from src.llm.base import LLMProvider
from src.models.schemas import CandidateClip, ClipAnalysis, Transcript, TranscriptSegment
from src.pipeline.config import LLMConfig
from src.utils.errors import LLMError

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert short-form video editor (TikTok, Instagram Reels, YouTube Shorts).
You receive candidate moments cut from a longer video, each with a timestamped transcript.
Evaluate every candidate honestly. Most candidates are mediocre: use the full 0-100 range and give
80+ only to genuinely strong moments.

Score each candidate from 0 to 100 on:
- humor, hook (do the first 3 seconds grab attention), surprise, emotion, shareability,
- visual_interest (judge from the words alone; use 50 if unknown),
- context_completeness (100 = fully understandable by a cold viewer),
- payoff (is there a punchline / resolution), overall (overall short-form potential).

Also return:
- recommended_start / recommended_end: ABSOLUTE seconds in the source video. Start shortly before the
  key statement or at the beginning of the story, on a natural sentence boundary (no fragments, no dead
  air). End right after the punchline / reaction / payoff. Stay inside the allowed range given per candidate.
- title: a short punchy hook line (max 12 words)
- reasoning: 1-2 sentences on why this clip works or doesn't
- requires_too_much_context: true if a cold viewer would be confused

Respond with ONLY a JSON object of exactly this shape (no markdown):
{"analyses":[{"candidate_id":"...","humor":0,"hook":0,"surprise":0,"emotion":0,"shareability":0,
"visual_interest":0,"context_completeness":0,"payoff":0,"overall":0,"recommended_start":0.0,
"recommended_end":0.0,"title":"...","reasoning":"...","requires_too_much_context":false}]}
Include exactly one entry per candidate, using the given candidate_id."""


def _segments_between(segments: list[TranscriptSegment], start: float, end: float) -> list[TranscriptSegment]:
    return [s for s in segments if s.end > start and s.start < end]


def _fmt(segs: list[TranscriptSegment]) -> str:
    return "\n".join(f"[{s.start:.1f}s] {s.text.strip()}" for s in segs) or "(none)"


def _extract_json(raw: str):
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


class LLMAnalyzer:
    def __init__(self, provider: LLMProvider, cfg: LLMConfig):
        self.provider = provider
        self.cfg = cfg

    # ------------------------------------------------------------------ public
    def analyze(self, candidates: list[CandidateClip], transcript: Transcript) -> list[ClipAnalysis]:
        results: list[ClipAnalysis] = []
        size = max(1, self.cfg.batch_size)
        for i in range(0, len(candidates), size):
            batch = candidates[i : i + size]
            log.info("Analyzing batch %d-%d of %d", i + 1, i + len(batch), len(candidates))
            results.extend(self._analyze_batch(batch, transcript))
        missing = len(candidates) - len(results)
        if missing:
            log.warning("%d candidates could not be analyzed and were dropped.", missing)
        return results

    # ------------------------------------------------------------------ internals
    def _analyze_batch(self, batch: list[CandidateClip], transcript: Transcript) -> list[ClipAnalysis]:
        pending = {c.id: c for c in batch}
        done: list[ClipAnalysis] = []
        retry_note = ""

        for attempt in range(1, self.cfg.max_retries + 1):
            if not pending:
                break
            prompt = self._build_prompt(list(pending.values()), transcript) + retry_note
            try:
                raw = self.provider.complete_json(SYSTEM_PROMPT, prompt)
                data = _extract_json(raw)
            except json.JSONDecodeError as exc:
                log.warning("Attempt %d: invalid JSON from LLM (%s)", attempt, exc)
                retry_note = "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
                continue
            except LLMError as exc:
                log.warning("Attempt %d: provider error: %s", attempt, exc)
                if attempt == self.cfg.max_retries:
                    raise
                continue

            items = data.get("analyses", []) if isinstance(data, dict) else data
            if not isinstance(items, list):
                retry_note = "\n\nThe 'analyses' key must be a list. Reply with ONLY the JSON object."
                continue

            for item in items:
                try:
                    analysis = ClipAnalysis.model_validate(item)
                except (ValidationError, TypeError) as exc:
                    log.warning("Attempt %d: invalid analysis item skipped: %s", attempt, exc)
                    continue
                cand = pending.get(analysis.candidate_id)
                if cand is None:
                    continue
                done.append(self._sanitize(analysis, cand))
                del pending[analysis.candidate_id]

            if pending:
                ids = ", ".join(pending)
                retry_note = f"\n\nYour previous reply was missing or had invalid entries for: {ids}. Provide them."

        for cid in pending:
            log.warning("Giving up on %s after %d attempts", cid, self.cfg.max_retries)
        return done

    def _build_prompt(self, batch: list[CandidateClip], transcript: Transcript) -> str:
        pad = self.cfg.context_padding
        parts = [f"Analyze these {len(batch)} candidates.\n"]
        for c in batch:
            lo, hi = max(0.0, c.start - pad), c.end + pad
            before = [s for s in transcript.segments if lo <= s.end <= c.start][-2:]
            after = [s for s in transcript.segments if c.end <= s.start <= hi][:2]
            body = _segments_between(transcript.segments, c.start, c.end)
            parts.append(
                f"### {c.id}\n"
                f"candidate range: {c.start:.1f}s - {c.end:.1f}s\n"
                f"allowed range for recommended_start/end: {lo:.1f}s - {hi:.1f}s\n"
                f"CONTEXT BEFORE:\n{_fmt(before)}\n"
                f"TRANSCRIPT:\n{_fmt(body)}\n"
                f"CONTEXT AFTER:\n{_fmt(after)}\n"
            )
        return "\n".join(parts)

    def _sanitize(self, a: ClipAnalysis, cand: CandidateClip) -> ClipAnalysis:
        pad = self.cfg.context_padding
        lo, hi = max(0.0, cand.start - pad), cand.end + pad
        valid = a.recommended_end > a.recommended_start and lo <= a.recommended_start and a.recommended_end <= hi
        if not valid:
            log.debug("%s: recommended times out of range, using candidate bounds", cand.id)
            a = a.model_copy(update={"recommended_start": cand.start, "recommended_end": cand.end})
        return a