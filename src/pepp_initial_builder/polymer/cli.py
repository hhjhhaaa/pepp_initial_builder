from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pepp_initial_builder.core import build_systems, export_manifest, load_config, validate_systems, write_matrix


REPO_ROOT = Path(__file__).resolve().parents[3]


def _config_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _mode(args: argparse.Namespace) -> str:
    return "tiny" if args.tiny else "pilot" if args.pilot else "matrix"


def _parser(default_config: str = "configs/polymer.yaml") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--max-systems", type=int)
    return parser


def generate_matrix_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(write_matrix(load_config(_config_path(args.config)), _mode(args)))


def build_structures_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_config(_config_path(args.config))
    for path in build_systems(config, args.tiny, args.pilot, args.max_systems):
        print(path)


def validate_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(validate_systems(load_config(_config_path(args.config)), args.tiny, args.pilot))


def export_manifest_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/polymer.yaml")
    args = parser.parse_args(argv)
    print(export_manifest(load_config(_config_path(args.config))))
