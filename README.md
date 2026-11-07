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

## Phase 2 (optional)

Set `phase2.enabled: true` in `config.yaml`, or pass `--phase2` / `--no-phase2` (overrides the file).
When enabled, `ClipRenderer` hands each clip to `phase2.pipeline.render_clip`: face detection and
tracking with a smoothed dynamic 9:16 crop (falls back to a centered crop, with a warning, if OpenCV
4.x Haar detection is unavailable), phase2 captions (`phase2.captions`), audio filters
(`phase2.render.loudnorm`/`limiter`) and optional silence removal (`phase2.edit`).
Output size and encoder settings still come from `video:`. `--caption-style` and `captions:` apply to the
legacy renderer only. Clip boundaries are set once, by `src/analysis/boundaries.py`, before selection;
clip length limits are `candidates.min_duration`/`max_duration` (`--min-duration`/`--max-duration`).
OpenCV 5.x has no Haar cascade, so `requirements.txt` pins `opencv-python<5`.
A Phase 2 render error fails that clip (see `output/analysis/report.json` and `logs/app.log`);
there is no silent fallback to the legacy renderer.


## Phase 3 — professional short-form editing

Enable with any Phase 3 flag or `phase3.enabled: true`. Without them nothing changes (Phase 1/2 paths are untouched).

```powershell
python main.py --input input/video.mp4 --num-clips 5 --editing-style podcast --aspect-ratio 9:16 `
  --foreground-ratio 1:1 --fit-mode auto --caption-style karaoke --dynamic-zoom --remove-silence --hook-text
```

| Flag | Meaning |
|---|---|
| `--phase3/--no-phase3` | master switch (any flag below turns it on) |
| `--editing-style podcast\|comedy\|interview\|general` | preset (sections in `config.yaml` override it) |
| `--aspect-ratio 9:16`, `--foreground-ratio 1:1` | canvas / sharp panel ratio (1080x1920 -> 1080x1080 panel) |
| `--fit-mode auto\|crop\|contain` | square-panel framing |
| `--caption-style classic\|bold\|karaoke\|highlighted\|minimal` | caption look |
| `--dynamic-zoom/--no-dynamic-zoom`, `--remove-silence/--no-remove-silence` | toggles |
| `--hook-text [TEXT]`, `--no-hook` | opening hook (auto from spoken words, or your text) |
| `--b-roll` | plans B-roll inserts into the report only |

Architecture (`phase3/`): `config` (typed + presets) · `layout` (canvas/panel geometry) · `framing` (AUTO/CROP/CONTAIN, group logic, reuses phase2 face sampling/tracking/smoothing) · `silence` + `jumpcuts` + `timeline` (word-safe cuts) · `zoom` · `reactions` · `captions` · `hooks` · `broll` · `filters` (one FFmpeg pass, encoder selection) · `render` · `qa`.

Decisions
- **AUTO is conservative.** It crops only when detected faces provably fit; 3+ faces, weak/no detections, or far-apart faces -> CONTAIN (whole frame, blurred fill, never stretched). Zoom is disabled in CONTAIN so nobody is cut off; in crop mode every zoom is capped so tracked faces stay inside.
- **Silence removal is audio-confirmed and word-safe.** Only the audio-silent part of a between-word gap is cut, with padding and a kept natural pause, micro-fades at joins, comedic pauses (setup->short payoff, questions, laughs) shortened only beyond `comedic_keep`, and a final check that no word straddles a cut.
- **Reaction shots** (cut to the whole group) happen only when a laugh is *measured* in the audio and >= 2 faces in the picture (needs `analysis.laughter: true`).
- **Hooks** are quotes of words actually spoken, or "WAIT FOR THE REACTION" only if a laugh was detected.
- Pixel values in caption/hook settings are authored for 1080 wide and scale with the output.

Output per clip: `output/clips/clip_XX.mp4`, `output/analysis/phase3/clip_XX.report.json` (framing mode + reason, cuts, zooms, hook, caption box, QA), `clip_XX.contact.jpg` (frames with the panel outlined). A clip failing a critical QA check is reported in `analysis/report.json` and not counted as rendered.

Known limitations
- Faces use OpenCV Haar (frontal only, CPU). Side-on or small faces are missed, so AUTO then prefers CONTAIN; there is no true active-speaker or expression/emotion recognition. Reactions rely on audio laughter, not facial expressions.
- B-roll is interface + offline folder matcher + report only; it is not composited.
- Silence audio uses micro-fades at joins (not overlapping crossfades) so audio stays in sync with the video.
- GPU encoding (`encoder: auto`) is used only if a test encode succeeds; filters run on CPU.
- Phase 3 does not run Whisper again; it reuses the transcript from the pipeline.


## Phase 3.1 — what changed

- **Hook:** default `clean_white` (white bold, black outline, soft shadow, no background block, fade + slight scale-in, max 2 lines, placed above the panel inside the safe area and never over captions or faces). Styles: `clean_white | minimal_shadow | subtle_pill | creator_custom`. A hook is rendered only when one exists: LLM suggestion (optional advisor, validated against the transcript: no invented names/numbers, no clickbait, no repeating the transcript) or a verified rule (a question asked later in the clip; "Wait For The Reaction" only if a laugh was measured). Otherwise **no hook**. `--hook-text "TEXT"` forces text, `--no-hook` disables.
- **Captions:** default clean bold, as-spoken case, trailing punctuation dropped, phrase grouping that avoids dangling words ("going | to"), balanced two-line wrapping, one event per phrase with smooth per-word colour tween from Whisper timestamps; only the active word is coloured, emphasis is size-only.
- **Zoom:** scored from independent evidence (punchline after a beat, laugh follows, loud sentence end, optional LLM-marked moments). A lone "!" never zooms. Levels 1.04 / 1.08 / 1.12 (max once per clip), face-safe, off in contain mode.
- **Reactions:** audio laugh or >= 2 faces smiling together / visibly moving (OpenCV smile cascade + frame difference) while >= 2 faces are visible -> cut to the group. These are smile/movement proxies, not emotion recognition; surprise, shock, gaze and gestures are NOT detected. MediaPipe was considered but not added (wheel availability on Python 3.13 / Windows is uncertain and it needs a model download).
- **Cuts:** dramatic pauses before a laugh are kept (up to `dramatic_pause_keep`), cut edges snap to the quietest instant, pacing guard (`max_cuts_per_10s`), jump cuts skip anything near a laugh and emphatic repeats.
- **Framing:** frontal + profile (mirrored) Haar faces, so side-on single speakers can be cropped; unsure -> contain.
- **Presets:** added `gaming` (never face-crops, no zoom) and `storytelling` (longer pauses, slow gentle zooms).
- **B-roll:** now really composited for a user-supplied folder (exact keyword = file-name word, never in first 1.5 s / last 1 s, never during reactions/laughs, panel only, audio untouched). Semantic matching and stock-footage APIs are NOT implemented.
- **QA:** adds hook/caption overlap + safe-area checks, audio level checks, per-clip `*.preview.md` (framing, hook, captions, zooms, cuts, reactions, B-roll, feature status incl. what was disabled/unavailable).
- **Advisor (optional):** one cached LLM call per clip for hook + zoom moments using your existing `llm:` section and API key (`phase3.advisor.mode: off` to disable; `auto` uses it only when a key/local endpoint exists). Not exercised against a live API in tests (fake provider only).
