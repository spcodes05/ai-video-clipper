"""CLI entry point:  python main.py --input input/video.mp4"""
import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.captions.ass_renderer import STYLES
from src.pipeline.config import DEFAULT_PROFILES, load_config
from phase3.config import CAPTION_STYLES, EDITING_STYLES
from src.pipeline.orchestrator import Pipeline
from src.utils.errors import PipelineError
from src.utils.logging import setup_logging

log = logging.getLogger("main")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Turn long videos into vertical short-form clips.")
    p.add_argument("--input", required=True, help="Path to the source video")
    p.add_argument("--output", help="Output directory (default: output)")
    p.add_argument("--num-clips", type=int, help="Number of clips to export (default: 5)")
    p.add_argument("--min-duration", type=float, help="Minimum clip length in seconds")
    p.add_argument("--max-duration", type=float, help="Maximum clip length in seconds")
    p.add_argument("--whisper-model", choices=["tiny", "base", "small", "medium", "large-v3"])
    p.add_argument("--caption-style", choices=sorted(set(STYLES) | set(CAPTION_STYLES)))
    p.add_argument("--profile", choices=sorted(DEFAULT_PROFILES.keys()), help="Content profile (default: general)")
    p.add_argument("--phase2", action=argparse.BooleanOptionalAction, default=None,
                   help="Enable/disable Phase 2 rendering (overrides phase2.enabled in config.yaml)")
    p.add_argument("--phase3", action=argparse.BooleanOptionalAction, default=None,
                   help="Enable/disable Phase 3 editing (any other phase3 flag below enables it)")
    p.add_argument("--editing-style", choices=EDITING_STYLES, help="Phase 3 preset: podcast | comedy | interview | general")
    p.add_argument("--aspect-ratio", help="Output canvas ratio, e.g. 9:16 (Phase 3)")
    p.add_argument("--foreground-ratio", help="Sharp foreground panel ratio, e.g. 1:1 (Phase 3)")
    p.add_argument("--fit-mode", choices=["auto", "crop", "contain"], help="Square panel framing (Phase 3)")
    p.add_argument("--dynamic-zoom", action=argparse.BooleanOptionalAction, default=None,
                   help="Emphasis zooms on/off (Phase 3)")
    p.add_argument("--remove-silence", action=argparse.BooleanOptionalAction, default=None,
                   help="Word-safe silence removal on/off (Phase 3)")
    p.add_argument("--hook-text", nargs="?", const="__auto__", default=None, metavar="TEXT",
                   help="Show an opening hook; optional TEXT overrides the generated one (Phase 3)")
    p.add_argument("--no-hook", action="store_true", help="Disable the opening hook (Phase 3)")
    p.add_argument("--b-roll", action="store_true", help="Plan B-roll inserts (report only; Phase 3)")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--debug", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv()
    args = parse_args(argv)
    p3_flags = (args.editing_style, args.aspect_ratio, args.foreground_ratio, args.fit_mode,
                args.dynamic_zoom, args.remove_silence, args.hook_text, args.b_roll)
    phase3 = args.phase3
    if phase3 is None and (any(v is not None and v is not False for v in p3_flags)):
        phase3 = True
    overrides = {
        "output": args.output,
        "num_clips": args.num_clips,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "whisper_model": args.whisper_model,
        "caption_style": args.caption_style,
        "profile": args.profile,
        "phase2": args.phase2,
        "phase3": phase3,
        "editing_style": args.editing_style,
        "aspect_ratio": args.aspect_ratio,
        "foreground_ratio": args.foreground_ratio,
        "fit_mode": args.fit_mode,
        "dynamic_zoom": args.dynamic_zoom,
        "remove_silence": args.remove_silence,
        "hook_text": args.hook_text,
        "no_hook": args.no_hook,
        "b_roll": args.b_roll,
    }
    try:
        cfg = load_config(args.config, overrides)
        setup_logging(cfg.logs_dir, args.debug)
        Pipeline(cfg, Path(args.input)).run()
        return 0
    except PipelineError as exc:
        log.error("Pipeline failed: %s", exc)
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error")
        print(f"\nUNEXPECTED ERROR: {exc}\nSee logs/app.log for the full traceback.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())