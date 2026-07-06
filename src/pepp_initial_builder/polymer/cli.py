from __future__ import annotations

import argparse
from typing import Sequence

from pepp_initial_builder.common.cli import config_path, mode_from_args, workflow_parser
from pepp_initial_builder.common.config import load_config
from pepp_initial_builder.polymer.emc_builder import build_systems
from pepp_initial_builder.polymer.emc_library import build_pure_library
from pepp_initial_builder.polymer.manifest import export_manifest
from pepp_initial_builder.polymer.matrix import write_matrix
from pepp_initial_builder.polymer.validation import validate_systems


def _parser(default_config: str = "configs/polymer.yaml") -> argparse.ArgumentParser:
    parser = workflow_parser(default_config)
    parser.add_argument("--max-systems", type=int)
    return parser


def generate_matrix_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(write_matrix(load_config(config_path(args.config)), mode_from_args(args, "matrix")))


def build_structures_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_config(config_path(args.config))
    for path in build_systems(config, args.tiny, args.pilot, args.max_systems):
        print(path)


def build_emc_library_main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    parser.add_argument("--run-relax", action="store_true")
    args = parser.parse_args(argv)
    print(build_pure_library(load_config(config_path(args.config)), mode_from_args(args, "pilot"), args.run_relax, args.max_systems))


def validate_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(validate_systems(load_config(config_path(args.config)), args.tiny, args.pilot))


def export_manifest_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/polymer.yaml")
    args = parser.parse_args(argv)
    print(export_manifest(load_config(config_path(args.config))))
