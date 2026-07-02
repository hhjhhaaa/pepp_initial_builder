from __future__ import annotations

import argparse

from pepp_initial_builder.common.cli import config_path
from pepp_initial_builder.common.config import load_config
from pepp_initial_builder.export.all_manifests import export_all_manifests
from pepp_initial_builder.export.summary import summarize_outputs


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cp2k_aimd.yaml")
    return parser


def export_all_manifests_main() -> None:
    args = parser().parse_args()
    print(export_all_manifests(load_config(config_path(args.config))))


def summarize_outputs_main() -> None:
    args = parser().parse_args()
    print(summarize_outputs(load_config(config_path(args.config))))
