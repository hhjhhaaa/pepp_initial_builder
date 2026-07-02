from __future__ import annotations

from typing import Sequence

from pepp_initial_builder.common.cli import config_path, mode_from_args, workflow_parser
from pepp_initial_builder.export.all_manifests import export_all_manifests
from pepp_initial_builder.mlff_seed.full_pore_seed import build_full_pore_seed_structures
from pepp_initial_builder.mlff_seed.lammps_relax import write_lammps_relax_inputs
from pepp_initial_builder.mlff_seed.validation import validate_full_pore_seed_structures
from pepp_initial_builder.pore.config import load_pore_config


def _parser():
    return workflow_parser("configs/mlff_seed.yaml")


def build_packmol_inputs_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(build_full_pore_seed_structures(load_pore_config(config_path(args.config)), mode_from_args(args)))


def pack_polymer_into_pore_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(build_full_pore_seed_structures(load_pore_config(config_path(args.config)), mode_from_args(args)))


def write_lammps_relax_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(write_lammps_relax_inputs(load_pore_config(config_path(args.config)), mode_from_args(args)))


def validate_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(validate_full_pore_seed_structures(load_pore_config(config_path(args.config))))


def export_manifest_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(export_all_manifests(load_pore_config(config_path(args.config))))
