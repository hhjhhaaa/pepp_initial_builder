from __future__ import annotations

from pepp_initial_builder.export.all_manifests import export_all_manifests
from pepp_initial_builder.export.summary import summarize_outputs


def export_all_manifests_main() -> None:
    print(export_all_manifests())


def summarize_outputs_main() -> None:
    print(summarize_outputs())
