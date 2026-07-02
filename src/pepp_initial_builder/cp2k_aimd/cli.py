from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pepp_initial_builder.cp2k_workflow import (
    build_aimd_dataset,
    export_aimd_dataset_manifest,
    load_cp2k_config,
    make_hpc_cp2k_jobs,
    parse_cp2k_outputs,
    validate_aimd_dataset,
    write_cp2k_label_inputs,
)
from pepp_initial_builder.pore_workflow import build_aimd_local_structures, load_pore_config
from pepp_initial_builder.reuse.lmp_proj_discovery import discover_lmp_proj_modules


REPO_ROOT = Path(__file__).resolve().parents[3]


def _config_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cp2k_aimd.yaml")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    return parser


def _mode(args: argparse.Namespace) -> str:
    return "tiny" if args.tiny else "pilot" if args.pilot else "main"


def discover_lmp_proj_reuse_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(discover_lmp_proj_modules(load_cp2k_config(_config_path(args.config))))


def build_seed_structures_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(build_aimd_local_structures(load_pore_config(_config_path(args.config)), _mode(args)))


def write_cp2k_inputs_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(write_cp2k_label_inputs(load_cp2k_config(_config_path(args.config)), _mode(args)))


def make_hpc_jobs_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(make_hpc_cp2k_jobs(load_cp2k_config(_config_path(args.config)), _mode(args)))


def parse_cp2k_outputs_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(parse_cp2k_outputs(load_cp2k_config(_config_path(args.config))))


def build_dataset_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(build_aimd_dataset(load_cp2k_config(_config_path(args.config))))


def validate_dataset_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(validate_aimd_dataset(load_cp2k_config(_config_path(args.config))))


def export_dataset_manifest_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(export_aimd_dataset_manifest(load_cp2k_config(_config_path(args.config))))
