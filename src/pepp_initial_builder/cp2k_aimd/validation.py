from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from pepp_initial_builder.cp2k_aimd.config import ensure_dirs, p, read_rows, write_rows
from pepp_initial_builder.cp2k_aimd.dataset_builder import frame_count
from pepp_initial_builder.pore.validation import validate_aimd_local_structures


def validate_aimd_dataset(config: Dict[str, Any]) -> Path:
    ensure_dirs(config)
    summary_path = p(config, "aimd_dataset_dir") / "dataset_summary.yaml"
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"dataset_status": "insufficient_real_cp2k_frames", "usable_for_mlff_training": False, "failure_reason": "dataset not built"}
    rows = []
    for row in read_rows(p(config, "aimd_dataset_dir") / "dataset_manifest.csv"):
        path = Path(row.get("extxyz_path", ""))
        rows.append({"split": row["split"], "extxyz_path": str(path) if row.get("extxyz_path") else "", "exists": bool(row.get("extxyz_path")) and path.exists(), "frame_count": frame_count(path) if row.get("extxyz_path") else 0, "valid": bool(summary.get("usable_for_mlff_training")) and bool(row.get("extxyz_path")) and path.exists()})
    if not rows:
        rows.append({"split": "none", "exists": False, "frame_count": 0, "valid": False})
    (p(config, "logs_dir") / "aimd_dataset_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    return write_rows(p(config, "logs_dir") / "aimd_dataset_validation.csv", rows)
