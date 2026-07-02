from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pepp_initial_builder.export.all_manifests import export_all_manifests
from pepp_initial_builder.pore_workflow import (
    build_porems_pores,
    crop_silica_patches,
    load_pore_config,
    validate_pore_structures,
    write_porems_discovery,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _config_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pore.yaml")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    return parser


def _mode(args: argparse.Namespace) -> str:
    return "tiny" if args.tiny else "pilot" if args.pilot else "main"


def discover_porems_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(write_porems_discovery(load_pore_config(_config_path(args.config))))


def build_pore_models_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(build_porems_pores(load_pore_config(_config_path(args.config)), _mode(args)))


def crop_patches_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(crop_silica_patches(load_pore_config(_config_path(args.config)), _mode(args)))


def validate_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(validate_pore_structures(load_pore_config(_config_path(args.config))))


def export_manifest_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(export_all_manifests(load_pore_config(_config_path(args.config))))
