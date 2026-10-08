"""CLI entry point:  python main.py --input input/video.mp4"""
import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.captions.ass_renderer import STYLES
from src.pipeline.config import load_config
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
    p.add_argument("--caption-style", choices=sorted(STYLES.keys()))
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--debug", action="store_true", help="Verbose logging")
    return p.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv()
    args = parse_args(argv)
    overrides = {
        "output": args.output,
        "num_clips": args.num_clips,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "whisper_model": args.whisper_model,
        "caption_style": args.caption_style,
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