from __future__ import annotations

from typing import Sequence

from pepp_initial_builder.common.cli import config_path, mode_from_args, workflow_parser
from pepp_initial_builder.export.all_manifests import export_all_manifests
from pepp_initial_builder.pore.config import load_pore_config
from pepp_initial_builder.pore.manifest import export_pore_aimd_manifests
from pepp_initial_builder.pore.patch_crop import crop_silica_patches
from pepp_initial_builder.pore.porems_builder import build_porems_pores
from pepp_initial_builder.pore.porems_discovery import write_porems_discovery
from pepp_initial_builder.pore.validation import validate_pore_structures


def _parser():
    return workflow_parser("configs/pore.yaml")


def discover_porems_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(write_porems_discovery(load_pore_config(config_path(args.config))))


def build_pore_models_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(build_porems_pores(load_pore_config(config_path(args.config)), mode_from_args(args)))


def crop_patches_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(crop_silica_patches(load_pore_config(config_path(args.config)), mode_from_args(args)))


def validate_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(validate_pore_structures(load_pore_config(config_path(args.config))))


def export_manifest_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(export_all_manifests(load_pore_config(config_path(args.config))))
