import json
import logging
import tempfile
from pathlib import Path

from src.analysis.boundaries import refine_boundaries
from src.analysis.local_analyzer import LocalAnalyzer
from src.analysis.scoring import rank_clips, select_clips
from src.detection.audio_analyzer import AudioAnalyzer
from src.detection.candidate_generator import CandidateGenerator
from src.detection.scene_detector import detect_scenes
from src.detection.visual_analyzer import VisualAnalyzer
from src.models.schemas import RenderedClip, Transcript, VideoMetadata
from src.pipeline.config import AppConfig
from src.pipeline.phase2_bridge import build_phase2_cfg, scene_cuts_from, selection_for
from src.transcription.whisper_service import WhisperService, save_transcript
from src.utils.errors import FFmpegError, PipelineError
from src.utils.ffmpeg import check_ffmpeg, extract_audio, probe_video, validate_video
from src.video.qa import validate_clip
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
        self.phase2_cfg = build_phase2_cfg(cfg)  # None unless phase2.enabled; raises ConfigError if broken

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _step(n: int, msg: str) -> None:
        print(f"[{n}/{TOTAL_STEPS}] {msg}", flush=True)
        log.info("[%d/%d] %s", n, TOTAL_STEPS, msg)

    # ------------------------------------------------------------------ run
    def run(self) -> list[RenderedClip]:
        meta = self._inspect()
        transcript, audio = self._transcribe(meta)
        scenes = self._detect_scenes(meta)
        visual = VisualAnalyzer.from_video(self.input_path)
        candidates = self._generate_candidates(transcript, scenes, audio)
        analyses = self._analyze(candidates, transcript, visual)
        selected = self._rank_and_select(candidates, analyses, transcript, meta)
        rendered = self._render(meta, transcript, selected, scenes)
        self._step(8, "Complete.")
        for r in rendered:
            print(f"      clip_{r.clip_number:02d}  score={r.viral_score:5.1f}  {r.duration:4.1f}s  {r.title}")
        print(f"\nOutput written to: {self.out.resolve()}")
        return rendered

    # ------------------------------------------------------------------ steps
    def _inspect(self) -> VideoMetadata:
        self._step(1, "Inspecting video...")
        check_ffmpeg()
        meta = probe_video(self.input_path)
        validate_video(meta, self.cfg.candidates.min_duration)
        for d in (self.clips_dir, self.transcripts_dir, self.analysis_dir):
            d.mkdir(parents=True, exist_ok=True)
        _dump(self.analysis_dir / "video_metadata.json", meta.model_dump(mode="json"))
        log.info("Video: %s", meta.model_dump())
        return meta

    def _transcribe(self, meta: VideoMetadata):
        self._step(2, "Transcribing (this can take a while on CPU)...")
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            extract_audio(self.input_path, wav)
            transcript: Transcript = WhisperService(self.cfg.whisper).transcribe(wav)
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

    def _analyze(self, candidates, transcript, visual):
        self._step(5, f"Scoring {len(candidates)} candidates locally (profile: {self.cfg.analysis.profile})...")
        analyses = LocalAnalyzer(self.cfg, visual).analyze(candidates, transcript)
        if not analyses:
            raise PipelineError("Local analysis produced no results. Check logs/app.log for details.")
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
        selected = select_clips(ranked, selection_for(self.cfg.selection, self.phase2_cfg))
        if len(selected) < self.cfg.selection.num_clips:
            log.warning("Only %d/%d clips could be selected.", len(selected), self.cfg.selection.num_clips)
        _dump(self.analysis_dir / "rankings.json", [r.model_dump(mode="json") for r in ranked])
        if not selected:
            raise PipelineError("No clips could be selected.")
        return selected

    def _render(self, meta, transcript, selected, scenes=()):
        self._step(7, "Rendering (Phase 2 enabled)..." if self.phase2_cfg else "Rendering...")
        scene_cuts = scene_cuts_from(scenes)
        for old in self.clips_dir.glob("clip_*.mp4"):
            old.unlink()
        renderer = ClipRenderer(
            self.cfg.video,
            self.cfg.captions,
            get_reframer(self.cfg.video.reframe_mode),
            self.analysis_dir / "captions",
            phase2_cfg=self.phase2_cfg,
        )
        rendered: list[RenderedClip] = []
        report: list[dict] = []
        for clip in selected:
            out_path = self.clips_dir / f"clip_{clip.clip_number:02d}.mp4"
            print(f"      rendering {out_path.name} ({clip.final_end - clip.final_start:.1f}s)...", flush=True)
            entry = {
                "clip": out_path.name,
                "candidate_id": clip.candidate.id,
                "start": clip.final_start,
                "end": clip.final_end,
                "viral_score": clip.viral_score,
                "title": clip.analysis.title,
                "reasoning": clip.analysis.reasoning,
                "ok": False,
                "issues": [],
            }
            item = None
            try:
                item = renderer.render(
                    self.input_path, meta, clip, transcript, out_path, scene_cuts=scene_cuts
                )
                entry["issues"] = validate_clip(
                    out_path, item.duration, self.cfg.video.width, self.cfg.video.height
                )
            except FFmpegError as exc:
                entry["issues"] = [f"render failed: {exc}"]
            entry["ok"] = not entry["issues"]
            report.append(entry)
            if entry["ok"] and item is not None:
                rendered.append(item)
            else:
                log.error("%s failed: %s", out_path.name, "; ".join(entry["issues"]))
                print(f"      ! {out_path.name} failed: {'; '.join(entry['issues'])}")
        _dump(
            self.analysis_dir / "report.json",
            {
                "input": str(self.input_path),
                "profile": self.cfg.analysis.profile,
                "phase2_enabled": self.phase2_cfg is not None,
                "requested_clips": self.cfg.selection.num_clips,
                "rendered_clips": len(rendered),
                "clips": report,
            },
        )
        if not rendered:
            raise PipelineError("All clip renders failed validation. See analysis/report.json and logs/app.log.")
        _dump(self.analysis_dir / "rendered.json", [r.model_dump(mode="json") for r in rendered])
        return rendered