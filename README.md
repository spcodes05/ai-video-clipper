# AI Video Clipper — Phase 1

Turns a long video (podcast, interview, livestream, lecture) into vertical 9:16 short-form clips with burned-in captions.

```
VIDEO -> transcript -> candidate windows (local, cheap) -> LLM analysis -> ranking -> top N
      -> refined start/end -> 9:16 crop -> captions -> output/clips/clip_XX.mp4
```

## Architecture

| Module | Responsibility |
|---|---|
| `src/pipeline` | config loading + orchestration of the 8 steps |
| `src/transcription` | Faster-Whisper behind a replaceable `TranscriptionService` |
| `src/detection` | scene detection, audio features, candidate window generation (no LLM) |
| `src/llm` | provider interface (`LLMProvider`) + OpenAI implementation |
| `src/analysis` | LLM analysis (batched, validated, retried), scoring, selection, boundary refinement |
| `src/video` | cutter, reframer (center crop now; face/speaker tracking later), renderer (one FFmpeg pass) |
| `src/captions` | word grouping + ASS subtitle generation with a style registry |
| `src/models` | Pydantic schemas |

Design choices: only 20-50 locally generated candidates ever reach the LLM (cost); every provider/reframer/style/transcriber is behind a small interface so it can be swapped; cut + crop + captions + encode happen in a single FFmpeg pass (no intermediate files).

## Installation (Windows)

1. Install Python 3.11+ from python.org (tick "Add Python to PATH").
2. Install FFmpeg:
```powershell
   winget install Gyan.FFmpeg
```
   Close and reopen the terminal, then verify:
```powershell
   ffmpeg -version
   ffprobe -version
```
3. Set up the project:
```powershell
   cd ai-video-clipper
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   copy .env.example .env
```
   (If script execution is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.)
4. Open `.env` and set `OPENAI_API_KEY`.

macOS: `brew install ffmpeg` — Ubuntu: `sudo apt install ffmpeg`.

## Environment variables

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | required for LLM analysis |
| `WHISPER_MODEL` | `tiny`, `base`, `small` (default), `medium`, `large-v3` |
| `OUTPUT_DIR` | output directory (default `output`) |

All tunables (scoring weights, durations, caption style, encoder settings) live in `config.yaml`. Precedence: defaults < `config.yaml` < environment < CLI flags.

## Run

```powershell
python main.py --input input/podcast.mp4
python main.py --input input/podcast.mp4 --num-clips 10 --whisper-model small --caption-style bold --debug
```

Flags: `--input --output --num-clips --min-duration --max-duration --whisper-model --caption-style --config --debug`

## Output

```
output/
  clips/clip_01.mp4 ...
  transcripts/transcript.json, transcript.txt
  analysis/video_metadata.json, scenes.json, candidates.json, rankings.json, rendered.json, captions/*.ass
logs/app.log
```

`rankings.json` contains every analyzed candidate with its score, the LLM's reasoning, final boundaries, and whether it was selected.

## How scoring works

1. **Candidate generation (local):** sliding windows over transcript segments (15-60 s). Each is scored from: speech density, emotional words, question/answer structure, punchline-like endings, hook-like openings, audio energy/peaks, speech activity, and scene-boundary alignment (weights in `config.yaml -> candidates.signal_weights`). Non-max suppression keeps up to 40 diverse windows.
2. **LLM analysis:** candidates are sent in batches of 5. The LLM returns humor, hook, surprise, emotion, shareability, visual interest, context completeness, payoff, overall, recommended start/end, title, and reasoning as JSON, validated with Pydantic, with retries for invalid or missing entries.
3. **ViralScore:**
   `0.20*hook + 0.20*humor + 0.15*surprise + 0.15*emotion + 0.10*shareability + 0.10*visual_interest + 0.10*context_completeness`, multiplied by `context_penalty` when the clip needs too much context. Weights are configurable and are a placeholder for a trained model.
4. **Selection:** top N by score, skipping clips that overlap an already selected one.
5. **Boundary refinement:** the LLM's times are snapped to sentence boundaries, given a small lead-in/tail, and prevented from bleeding into neighbouring speech.

The score is a ranking heuristic. It does not predict or guarantee virality.

## Known limitations

- Center crop only; speakers off-center can be cut off (face/speaker tracking is Phase 2).
- Audio analysis is loudness/speech activity only, with no real laughter or cheer detection.
- The LLM judges text only (no visuals); `visual_interest` is a guess.
- Transcription on CPU is slow for long videos; use `tiny`/`base` while developing.
- Whisper accuracy affects captions; no speaker diarization.
- Scene detection on very long videos can take several minutes.
- If captions do not appear or fonts look wrong on Windows, ensure Arial is installed or change the font in `src/captions/ass_renderer.py`.

## Roadmap

- **Phase 2:** face detection, active speaker detection, dynamic crop, laughter/emotion detection.
- **Phase 3:** dynamic zoom, jump cuts, silence removal, B-roll, animated/highlighted captions, hook text, creator styles.
- **Phase 4:** collect real performance data (views, likes, shares, saves, completion rate, ...), train XGBoost/LightGBM virality ranker, replace the fixed weights.
- Later: FastAPI, task queue, publishing, frontend.

## Tests

```powershell
python -m pytest
```
`tests/test_pipeline_smoke.py` renders a real 9:16 clip from a generated video using a fake LLM (needs FFmpeg, no API key, no Whisper).