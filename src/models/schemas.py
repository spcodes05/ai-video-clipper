from pydantic import BaseModel, ConfigDict, Field, field_validator


class VideoMetadata(BaseModel):
    path: str
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    has_audio: bool


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: float = 1.0


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)


class Transcript(BaseModel):
    language: str = "unknown"
    duration: float = 0.0
    segments: list[TranscriptSegment] = Field(default_factory=list)


class Scene(BaseModel):
    id: int = 0
    start: float
    end: float


class AudioFeatures(BaseModel):
    energy: float = 0.0
    peak_energy: float = 0.0
    speech_ratio: float = 0.0
    silence_ratio: float = 1.0


class CandidateClip(BaseModel):
    id: str
    start: float
    end: float
    transcript: str
    scene_ids: list[int] = Field(default_factory=list)
    audio_features: AudioFeatures
    heuristic_score: float = 0.0
    signals: dict[str, float] = Field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


SCORE_FIELDS = (
    "humor",
    "hook",
    "surprise",
    "emotion",
    "shareability",
    "visual_interest",
    "context_completeness",
    "payoff",
    "overall",
)


class ClipAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    humor: float
    hook: float
    surprise: float
    emotion: float
    shareability: float
    visual_interest: float
    context_completeness: float
    payoff: float
    overall: float
    recommended_start: float
    recommended_end: float
    title: str
    reasoning: str
    requires_too_much_context: bool = False

    @field_validator(*SCORE_FIELDS, mode="before")
    @classmethod
    def _clamp_score(cls, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("score must be numeric")
        return max(0.0, min(100.0, value))


class RankedClip(BaseModel):
    candidate: CandidateClip
    analysis: ClipAnalysis
    viral_score: float
    rank: int = 0
    final_start: float
    final_end: float
    selected: bool = False
    clip_number: int | None = None


class RenderedClip(BaseModel):
    clip_number: int
    path: str
    start: float
    end: float
    duration: float
    viral_score: float
    title: str
    reasoning: str