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
    rows = [{"split": split, "extxyz_path": "", "frame_count": 0} for split in ["train", "val", "test"]]
    if total < min_frames:
        summary = {"dataset_status": "insufficient_real_cp2k_frames", "usable_for_mlff_training": False, "failure_reason": f"real frames < {min_frames}"}
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
        summary = {"dataset_status": "ready", "usable_for_mlff_training": True, "failure_reason": ""}
    (p(config, "aimd_dataset_dir") / "dataset_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    return write_rows(p(config, "aimd_dataset_dir") / "dataset_manifest.csv", rows)
