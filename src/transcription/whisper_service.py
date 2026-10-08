import logging
from abc import ABC, abstractmethod
from pathlib import Path

from src.models.schemas import Transcript, TranscriptSegment, WordTimestamp
from src.pipeline.config import WhisperConfig
from src.utils.errors import TranscriptionError
from src.utils.timeutils import format_timestamp

log = logging.getLogger(__name__)


class TranscriptionService(ABC):
    """Replaceable transcription backend."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> Transcript: ...


class WhisperService(TranscriptionService):
    def __init__(self, cfg: WhisperConfig):
        self.cfg = cfg
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel

                log.info("Loading Whisper model '%s' (device=%s)", self.cfg.model, self.cfg.device)
                self._model = WhisperModel(
                    self.cfg.model, device=self.cfg.device, compute_type=self.cfg.compute_type
                )
            except Exception as exc:  # noqa: BLE001
                raise TranscriptionError(f"Could not load Whisper model '{self.cfg.model}': {exc}") from exc
        return self._model

    def transcribe(self, audio_path: Path) -> Transcript:
        model = self._load()
        try:
            raw_segments, info = model.transcribe(
                str(audio_path), word_timestamps=True, vad_filter=True, beam_size=5
            )
            segments: list[TranscriptSegment] = []
            for seg in raw_segments:  # generator: actual decoding happens here
                words = [
                    WordTimestamp(
                        word=w.word.strip(), start=w.start, end=w.end, probability=w.probability
                    )
                    for w in (seg.words or [])
                    if w.word.strip()
                ]
                segments.append(
                    TranscriptSegment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words)
                )
                if len(segments) % 50 == 0:
                    log.info("Transcribed %d segments (up to %s)", len(segments), format_timestamp(seg.end))
        except TranscriptionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(f"Transcription failed: {exc}") from exc
        return Transcript(language=info.language, segments=segments)


def transcript_to_text(transcript: Transcript) -> str:
    return "\n".join(
        f"[{format_timestamp(s.start)} -> {format_timestamp(s.end)}] {s.text}" for s in transcript.segments
    )


def save_transcript(transcript: Transcript, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript.json").write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "transcript.txt").write_text(transcript_to_text(transcript), encoding="utf-8")