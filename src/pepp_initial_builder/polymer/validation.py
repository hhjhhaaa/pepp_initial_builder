from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pepp_initial_builder.common.paths import ensure_dirs, project_root
from pepp_initial_builder.polymer.emc_builder import select_rows


def validate_system(system_dir: Path, heavy_threshold: float = 1.8) -> Dict[str, Any]:
    required = ["polymer.xyz", "polymer.extxyz", "polymer.pdb", "polymer.data", "metadata.yaml"]
    result: Dict[str, Any] = {"system_id": system_dir.name, "usable_for_mlff_start": True}
    missing = [name for name in required if not (system_dir / name).exists()]
    result["missing_files"] = ";".join(missing)
    if missing:
        result["usable_for_mlff_start"] = False
        return result
    atom_count = max(0, len((system_dir / "polymer.xyz").read_text(encoding="utf-8", errors="ignore").splitlines()) - 2)
    forbidden = []
    for name in ["prod.lammpstrj", "production.lammpstrj"]:
        if list(system_dir.rglob(name)):
            forbidden.append(name)
    result["atom_count"] = atom_count
    result["forbidden_production_trajectory_files"] = ";".join(forbidden)
    result["usable_for_mlff_start"] = atom_count > 0 and not forbidden
    return result


def validate_systems(config: Dict[str, Any], tiny: bool = False, pilot: bool = False) -> Path:
    ensure_dirs(config)
    records = []
    for row in select_rows(config, tiny, pilot, None):
        system_dir = project_root(config) / config["paths"]["systems_dir"] / row["system_id"]
        if system_dir.exists():
            records.append(validate_system(system_dir, float(config["builder"].get("min_heavy_atom_distance_A", 1.8))))
        else:
            records.append({"system_id": row["system_id"], "usable_for_mlff_start": False, "missing_files": "system_dir"})
    out = project_root(config) / config["paths"]["logs_dir"] / "initial_structure_validation.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    return out
