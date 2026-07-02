from __future__ import annotations

from typing import Sequence

from pepp_initial_builder.common.cli import config_path, mode_from_args, workflow_parser
from pepp_initial_builder.cp2k_aimd.config import load_cp2k_config
from pepp_initial_builder.cp2k_aimd.dataset_builder import build_aimd_dataset
from pepp_initial_builder.cp2k_aimd.hpc_jobs import make_hpc_cp2k_jobs
from pepp_initial_builder.cp2k_aimd.input_writer import write_cp2k_label_inputs
from pepp_initial_builder.cp2k_aimd.manifest import export_aimd_dataset_manifest
from pepp_initial_builder.cp2k_aimd.parser import parse_cp2k_outputs
from pepp_initial_builder.cp2k_aimd.seed_patch_builder import build_aimd_local_structures
from pepp_initial_builder.cp2k_aimd.validation import validate_aimd_dataset
from pepp_initial_builder.pore.config import load_pore_config
from pepp_initial_builder.reuse.lmp_proj_discovery import discover_lmp_proj_modules


def _parser():
    return workflow_parser("configs/cp2k_aimd.yaml")


def discover_lmp_proj_reuse_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(discover_lmp_proj_modules(load_cp2k_config(config_path(args.config))))


def build_seed_structures_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(build_aimd_local_structures(load_pore_config(config_path(args.config)), mode_from_args(args)))


def write_cp2k_inputs_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(write_cp2k_label_inputs(load_cp2k_config(config_path(args.config)), mode_from_args(args)))


def make_hpc_jobs_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(make_hpc_cp2k_jobs(load_cp2k_config(config_path(args.config)), mode_from_args(args)))


def parse_cp2k_outputs_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(parse_cp2k_outputs(load_cp2k_config(config_path(args.config))))


def build_dataset_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(build_aimd_dataset(load_cp2k_config(config_path(args.config))))


def validate_dataset_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(validate_aimd_dataset(load_cp2k_config(config_path(args.config))))


def export_dataset_manifest_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(export_aimd_dataset_manifest(load_cp2k_config(config_path(args.config))))
