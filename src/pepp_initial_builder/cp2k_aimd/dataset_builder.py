from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from pepp_initial_builder.cp2k_aimd.config import ensure_dirs, p, read_rows, write_rows


def frame_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip().isdigit())


def build_aimd_dataset(config: Dict[str, Any]) -> Path:
    ensure_dirs(config)
    parsed = read_rows(p(config, "cp2k_parsed_dir") / "cp2k_parsed_manifest.csv")
    real = [row for row in parsed if row.get("status") == "parsed_real_cp2k_output"]
    min_frames = int(config["dataset"]["min_frames_for_training"])
    total = sum(int(row.get("usable_frame_count", 0)) for row in real)
    failed = [row for row in parsed if row.get("status") != "parsed_real_cp2k_output"]
    def _count_by(key: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in real:
            label = str(row.get(key, "") or "unknown")
            counts[label] = counts.get(label, 0) + int(row.get("usable_frame_count", 0))
        return counts

    failure_reason_counts: Dict[str, int] = {}
    for row in failed:
        reason = str(row.get("failure_reason", "") or row.get("status", "unknown"))
        failure_reason_counts[reason] = failure_reason_counts.get(reason, 0) + 1
    num_sp = sum(int(row.get("usable_frame_count", 0)) for row in real if row.get("label_mode") == "sp_force" or row.get("cp2k_run_type") == "ENERGY_FORCE")
    num_aimd = sum(int(row.get("usable_frame_count", 0)) for row in real if row.get("label_mode") == "short_aimd" or row.get("cp2k_run_type") == "MD")
    source_counts = _count_by("source_stage")
    relaxed_count = source_counts.get("lammps_relaxed_full_pore", 0)
    non_relaxed_count = total - relaxed_count
    accepted_source_stages = list(config.get("dataset", {}).get("accepted_source_stages", ["lammps_relaxed_full_pore"]))
    accepted_source_count = sum(source_counts.get(stage, 0) for stage in accepted_source_stages)
    rejected_source_count = total - accepted_source_count
    common_summary = {
        "num_sp_frames": num_sp,
        "num_aimd_frames": num_aimd,
        "num_frames_by_crop_family": _count_by("crop_family"),
        "num_frames_by_polymer_architecture": _count_by("polymer_architecture"),
        "num_frames_by_source_stage": source_counts,
        "num_frames_from_lammps_relaxed_full_pore": relaxed_count,
        "num_frames_from_non_relaxed_source": non_relaxed_count,
        "accepted_source_stages": accepted_source_stages,
        "num_frames_from_accepted_source_stages": accepted_source_count,
        "num_frames_from_rejected_source_stages": rejected_source_count,
        "num_failed_cp2k_jobs": len(failed),
        "failure_reason_counts": failure_reason_counts,
    }
    rows = [{"split": split, "extxyz_path": "", "frame_count": 0} for split in ["train", "val", "test"]]
    if total < min_frames:
        summary = {"dataset_status": "insufficient_real_cp2k_frames", "usable_for_mlff_training": False, "failure_reason": f"real frames < {min_frames}", **common_summary}
    elif rejected_source_count != 0:
        summary = {"dataset_status": "failed_rejected_source_stage_detected", "usable_for_mlff_training": False, "failure_reason": "accepted frames include source_stage outside accepted_source_stages", **common_summary}
    else:
        block_keys = list(dict.fromkeys("|".join([str(row.get("family", "")), str(row.get("patch_id", "")), str(row.get("aimd_structure_id", ""))]) for row in real))
        split_for = {}
        for i, key in enumerate(block_keys):
            frac = i / max(len(block_keys), 1)
            split_for[key] = "train" if frac < float(config["dataset"]["train_fraction"]) else "val" if frac < float(config["dataset"]["train_fraction"]) + float(config["dataset"]["val_fraction"]) else "test"
        rows = []
        for split in ["train", "val", "test"]:
            part = [row for row in real if split_for.get("|".join([str(row.get("family", "")), str(row.get("patch_id", "")), str(row.get("aimd_structure_id", ""))])) == split]
            out = p(config, "aimd_dataset_dir") / f"{split}.extxyz"
            if part:
                out.write_text("".join(Path(row["frames_extxyz_path"]).read_text(encoding="utf-8") for row in part), encoding="utf-8")
            rows.append({"split": split, "extxyz_path": str(out) if part else "", "frame_count": sum(int(row.get("usable_frame_count", 0)) for row in part)})
        summary = {"dataset_status": "ready", "usable_for_mlff_training": True, "failure_reason": "", **common_summary}
    (p(config, "aimd_dataset_dir") / "dataset_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    return write_rows(p(config, "aimd_dataset_dir") / "dataset_manifest.csv", rows)
