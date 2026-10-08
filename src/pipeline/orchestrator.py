import json
import logging
import tempfile
from pathlib import Path

from src.analysis.boundaries import refine_boundaries
from src.analysis.llm_analyzer import LLMAnalyzer
from src.analysis.scoring import rank_clips, select_clips
from src.detection.audio_analyzer import AudioAnalyzer
from src.detection.candidate_generator import CandidateGenerator
from src.detection.scene_detector import detect_scenes
from src.llm import get_provider
from src.models.schemas import RenderedClip, Transcript, VideoMetadata
from src.pipeline.config import AppConfig
from src.transcription.whisper_service import WhisperService, save_transcript
from src.utils.errors import FFmpegError, PipelineError
from src.utils.ffmpeg import check_ffmpeg, extract_audio, probe_video, validate_video
from src.video.reframer import get_reframer
from src.video.renderer import ClipRenderer

log = logging.getLogger(__name__)
TOTAL_STEPS = 8


def _dump(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class Pipeline:
    def __init__(self, cfg: AppConfig, input_path: Path):
        self.cfg = cfg
        self.input_path = Path(input_path)
        self.out = Path(cfg.output_dir)
        self.clips_dir = self.out / "clips"
        self.transcripts_dir = self.out / "transcripts"
        self.analysis_dir = self.out / "analysis"

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _step(n: int, msg: str) -> None:
        print(f"[{n}/{TOTAL_STEPS}] {msg}", flush=True)
        log.info("[%d/%d] %s", n, TOTAL_STEPS, msg)

    # ------------------------------------------------------------------ run
    def run(self) -> list[RenderedClip]:
        meta, provider = self._inspect()
        transcript, audio = self._transcribe(meta)
        scenes = self._detect_scenes(meta)
        candidates = self._generate_candidates(transcript, scenes, audio)
        analyses = self._analyze(candidates, transcript, provider)
        selected = self._rank_and_select(candidates, analyses, transcript, meta)
        rendered = self._render(meta, transcript, selected)
        self._step(8, "Complete.")
        for r in rendered:
            print(f"      clip_{r.clip_number:02d}  score={r.viral_score:5.1f}  {r.duration:4.1f}s  {r.title}")
        print(f"\nOutput written to: {self.out.resolve()}")
        return rendered

    # ------------------------------------------------------------------ steps
    def _inspect(self):
        self._step(1, "Inspecting video...")
        check_ffmpeg()
        meta = probe_video(self.input_path)
        validate_video(meta, self.cfg.candidates.min_duration)
        # Fail fast on a missing API key BEFORE the expensive transcription.
        provider = get_provider(self.cfg.llm)
        for d in (self.clips_dir, self.transcripts_dir, self.analysis_dir):
            d.mkdir(parents=True, exist_ok=True)
        _dump(self.analysis_dir / "video_metadata.json", meta.model_dump(mode="json"))
        log.info("Video: %s", meta.model_dump())
        return meta, provider

    def _transcribe(self, meta: VideoMetadata):
        self._step(2, "Transcribing (this can take a while on CPU)...")
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            extract_audio(self.input_path, wav)
            transcript = WhisperService(self.cfg.whisper).transcribe(wav)
            audio = AudioAnalyzer.from_wav(wav)
        transcript.duration = meta.duration
        save_transcript(transcript, self.transcripts_dir)
        if not transcript.segments:
            raise PipelineError("Transcription produced no speech segments; nothing to clip.")
        return transcript, audio

    def _detect_scenes(self, meta: VideoMetadata):
        self._step(3, "Detecting scenes...")
        scenes = detect_scenes(self.input_path, meta.duration, self.cfg.candidates.scene_threshold)
        _dump(self.analysis_dir / "scenes.json", [s.model_dump(mode="json") for s in scenes])
        return scenes

    def _generate_candidates(self, transcript, scenes, audio):
        self._step(4, "Generating candidates...")
        candidates = CandidateGenerator(self.cfg.candidates, audio, scenes).generate(transcript)
        if not candidates:
            raise PipelineError(
                "No candidate moments found. The video may be too short or contain too little speech."
            )
        _dump(self.analysis_dir / "candidates.json", [c.model_dump(mode="json") for c in candidates])
        print(f"      {len(candidates)} candidates")
        return candidates

    def _analyze(self, candidates, transcript, provider):
        self._step(5, f"Analyzing {len(candidates)} candidates with LLM...")
        analyses = LLMAnalyzer(provider, self.cfg.llm).analyze(candidates, transcript)
        if not analyses:
            raise PipelineError("The LLM returned no valid analyses. Check logs/app.log for details.")
        return analyses

    def _rank_and_select(self, candidates, analyses, transcript, meta):
        self._step(6, "Ranking clips...")
        ranked = rank_clips(candidates, analyses, self.cfg.scoring)
        for r in ranked:
            r.final_start, r.final_end = refine_boundaries(
                r.analysis.recommended_start,
                r.analysis.recommended_end,
                transcript.segments,
                meta.duration,
                self.cfg.boundaries,
                self.cfg.candidates.max_duration,
            )
        selected = select_clips(ranked, self.cfg.selection)
        if len(selected) < self.cfg.selection.num_clips:
            log.warning("Only %d/%d clips could be selected.", len(selected), self.cfg.selection.num_clips)
        _dump(self.analysis_dir / "rankings.json", [r.model_dump(mode="json") for r in ranked])
        if not selected:
            raise PipelineError("No clips could be selected.")
        return selected

    def _render(self, meta, transcript, selected):
        self._step(7, "Rendering...")
        for old in self.clips_dir.glob("clip_*.mp4"):
            old.unlink()
        renderer = ClipRenderer(
            self.cfg.video,
            self.cfg.captions,
            get_reframer(self.cfg.video.reframe_mode),
            self.analysis_dir / "captions",
        )
        rendered: list[RenderedClip] = []
        for clip in selected:
            out_path = self.clips_dir / f"clip_{clip.clip_number:02d}.mp4"
            print(f"      rendering {out_path.name} ({clip.final_end - clip.final_start:.1f}s)...", flush=True)
            try:
                rendered.append(renderer.render(self.input_path, meta, clip, transcript, out_path))
            except FFmpegError as exc:
                log.error("Rendering %s failed: %s", out_path.name, exc)
                print(f"      ! {out_path.name} failed: {exc}")
        if not rendered:
            raise PipelineError("All clip renders failed. See logs/app.log.")
        _dump(self.analysis_dir / "rendered.json", [r.model_dump(mode="json") for r in rendered])
        return rendered