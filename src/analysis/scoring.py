import logging

from src.models.schemas import CandidateClip, ClipAnalysis, RankedClip
from src.pipeline.config import ScoringConfig, SelectionConfig
from src.utils.timeutils import overlap_ratio

log = logging.getLogger(__name__)


def compute_viral_score(a: ClipAnalysis, cfg: ScoringConfig) -> float:
    """Configurable heuristic; replaced by a trained model in a later phase."""
    weights = cfg.weights.model_dump()
    total = sum(weights.values())
    score = sum(w * getattr(a, name) for name, w in weights.items()) / total
    if a.requires_too_much_context:
        score *= cfg.context_penalty
    return round(score, 2)


def rank_clips(
    candidates: list[CandidateClip], analyses: list[ClipAnalysis], cfg: ScoringConfig
) -> list[RankedClip]:
    by_id = {c.id: c for c in candidates}
    ranked = [
        RankedClip(
            candidate=by_id[a.candidate_id],
            analysis=a,
            viral_score=compute_viral_score(a, cfg),
            final_start=a.recommended_start,
            final_end=a.recommended_end,
        )
        for a in analyses
        if a.candidate_id in by_id
    ]
    ranked.sort(key=lambda r: r.viral_score, reverse=True)
    for i, r in enumerate(ranked, start=1):
        r.rank = i
    return ranked


def select_clips(ranked: list[RankedClip], cfg: SelectionConfig) -> list[RankedClip]:
    """Pick the top N by score, skipping clips that overlap an already selected one."""
    selected: list[RankedClip] = []
    for r in sorted(ranked, key=lambda x: x.viral_score, reverse=True):
        if len(selected) >= cfg.num_clips:
            break
        if not cfg.allow_overlap and any(
            overlap_ratio(r.final_start, r.final_end, s.final_start, s.final_end) > cfg.max_overlap_ratio
            for s in selected
        ):
            log.debug("Skipping %s: overlaps a higher-scoring clip", r.candidate.id)
            continue
        selected.append(r)
    for n, r in enumerate(selected, start=1):
        r.selected = True
        r.clip_number = n
    return selected