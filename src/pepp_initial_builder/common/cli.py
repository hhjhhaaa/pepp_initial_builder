from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def config_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def workflow_parser(default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    return parser


def mode_from_args(args: argparse.Namespace, default: str = "main") -> str:
    return "tiny" if args.tiny else "pilot" if args.pilot else default
