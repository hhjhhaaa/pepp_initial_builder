from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from pepp_initial_builder.cp2k_aimd.config import ensure_dirs, p, read_rows, write_rows


def export_aimd_dataset_manifest(config: Dict[str, Any]) -> Path:
    ensure_dirs(config)
    summary_path = p(config, "aimd_dataset_dir") / "dataset_summary.yaml"
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"dataset_status": "insufficient_real_cp2k_frames", "usable_for_mlff_training": False, "failure_reason": "dataset not built"}
    by_split = {row["split"]: row for row in read_rows(p(config, "aimd_dataset_dir") / "dataset_manifest.csv")}
    row = {"dataset_id": config["dataset"].get("dataset_id", "pepp_silica_aimd_cp2k"), "train_extxyz_path": by_split.get("train", {}).get("extxyz_path", ""), "val_extxyz_path": by_split.get("val", {}).get("extxyz_path", ""), "test_extxyz_path": by_split.get("test", {}).get("extxyz_path", ""), "dataset_manifest_path": str(p(config, "aimd_dataset_dir") / "dataset_manifest.csv"), "num_train_frames": by_split.get("train", {}).get("frame_count", 0), "num_val_frames": by_split.get("val", {}).get("frame_count", 0), "num_test_frames": by_split.get("test", {}).get("frame_count", 0), "elements": ";".join(config["dataset_scope"]["elements"]), "cp2k_method": config["cp2k"]["method"], "basis_set": config["cp2k"]["basis_set"], "potential": config["cp2k"]["potential"], "dispersion": config["cp2k"]["dispersion"], "aimd_structure_source_manifest": str(p(config, "aimd_structure_manifest")), "has_energy": bool(config["dataset"]["require_energy"]), "has_forces": bool(config["dataset"]["require_forces"]), "has_stress": False, "dataset_status": summary["dataset_status"], "usable_for_mlff_training": bool(summary["usable_for_mlff_training"]), "failure_reason": summary.get("failure_reason", "")}
    out_csv = p(config, "exports_dir") / "aimd_dataset_manifest.csv"
    write_rows(out_csv, [row])
    clean = {key: (value.replace("\\", "/") if isinstance(value, str) else value) for key, value in row.items()}
    (p(config, "exports_dir") / "aimd_dataset_manifest.json").write_text(json.dumps([clean], indent=2) + "\n", encoding="utf-8")
    return out_csv
