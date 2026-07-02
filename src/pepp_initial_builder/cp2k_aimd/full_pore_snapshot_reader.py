from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from pepp_initial_builder.cp2k_aimd.config import p, read_rows


def _existing_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    for row in read_rows(path):
        rows.append({str(key): "" if value is None else str(value) for key, value in row.items()})
    return rows


def _stage_from_row(row: Dict[str, str], default: str) -> str:
    text = row.get("source_stage") or row.get("mlff_start_structure_kind") or default
    if text == "raw_full_pore_seed":
        return "raw_full_pore_seed"
    if "exploration" in text:
        return "lammps_exploration_snapshot"
    if "relaxed" in text:
        return "lammps_relaxed_full_pore"
    return default


def _snapshot_path(row: Dict[str, str]) -> str:
    for key in ["source_snapshot_path", "snapshot_path", "relaxed_extxyz_path", "mlff_start_extxyz_path", "extxyz_path", "seed_extxyz_path"]:
        value = row.get(key, "")
        if value:
            return value
    return ""


def _normalize_row(row: Dict[str, str], source_manifest: Path, default_stage: str) -> Dict[str, str]:
    source_stage = _stage_from_row(row, default_stage)
    full_pore_id = row.get("source_full_pore_id") or row.get("full_pore_seed_id") or row.get("snapshot_id") or row.get("id", "")
    path = _snapshot_path(row)
    status = row.get("status", "")
    usable = str(row.get("usable_for_mlff_start", "true")).lower() not in {"false", "0", "no"}
    return {
        **row,
        "source_stage": source_stage,
        "source_full_pore_id": full_pore_id,
        "source_snapshot_path": path,
        "source_frame_index": row.get("source_frame_index", row.get("frame_index", "0")),
        "source_manifest": str(source_manifest),
        "recommended_label_mode": row.get("recommended_label_mode", "energy_force_and_short_aimd"),
        "confidence": row.get("confidence", "production_candidate"),
        "status": status,
        "usable_for_mlff_start": str(usable).lower(),
    }


def read_full_pore_snapshot_sources(config: Dict[str, Any]) -> List[Dict[str, str]]:
    paths = config.get("paths", {})
    allowed_stages = set(config.get("cp2k_crop", {}).get("source_priority", ["lammps_relaxed_full_pore", "lammps_exploration_snapshot"]))
    allowed_stages.discard("raw_full_pore_seed")
    candidates = [
        p(config, "exports_dir") / "full_pore_snapshot_manifest.csv",
        p(config, "exports_dir") / "mlff_seed_manifest.csv",
        p(config, "full_pore_seed_structures_dir") / "lammps_relax_manifest.csv",
        p(config, "full_pore_seed_structures_dir") / "full_pore_seed_manifest.csv",
    ]
    if paths.get("full_pore_snapshot_manifest"):
        candidates.insert(1, Path(paths["full_pore_snapshot_manifest"]))
    rows: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for manifest in candidates:
        if not str(manifest) or manifest.is_dir():
            continue
        for row in _existing_manifest(manifest):
            default_stage = "raw_full_pore_seed" if "mlff_seed" in manifest.name or "full_pore_seed" in manifest.name else "lammps_exploration_snapshot"
            normalized = _normalize_row(row, manifest, default_stage)
            if normalized["source_stage"] not in allowed_stages:
                continue
            path = normalized["source_snapshot_path"]
            key = (normalized["source_full_pore_id"], path, normalized["source_frame_index"])
            if not path or key in seen:
                continue
            status = normalized["status"]
            status_ok = (not status) or status.startswith("available") or status == "lammps_relaxed_full_pore"
            if not status_ok:
                continue
            if normalized["usable_for_mlff_start"] == "false":
                continue
            rows.append(normalized)
            seen.add(key)
    return rows


def write_source_audit(config: Dict[str, Any], rows: List[Dict[str, str]]) -> Path:
    out = p(config, "logs_dir") / "full_pore_snapshot_source_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
