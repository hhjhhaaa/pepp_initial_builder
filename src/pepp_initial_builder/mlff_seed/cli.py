from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pepp_initial_builder.export.all_manifests import export_all_manifests
from pepp_initial_builder.pore_workflow import build_full_pore_seed_structures, load_pore_config, validate_full_pore_seed_structures


REPO_ROOT = Path(__file__).resolve().parents[3]


def _config_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mlff_seed.yaml")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    return parser


def _mode(args: argparse.Namespace) -> str:
    return "tiny" if args.tiny else "pilot" if args.pilot else "main"


def build_packmol_inputs_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(build_full_pore_seed_structures(load_pore_config(_config_path(args.config)), _mode(args)))


def pack_polymer_into_pore_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(build_full_pore_seed_structures(load_pore_config(_config_path(args.config)), _mode(args)))


def write_lammps_relax_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(build_full_pore_seed_structures(load_pore_config(_config_path(args.config)), _mode(args)))


def validate_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(validate_full_pore_seed_structures(load_pore_config(_config_path(args.config))))


def export_manifest_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(export_all_manifests(load_pore_config(_config_path(args.config))))
