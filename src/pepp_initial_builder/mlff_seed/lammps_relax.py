from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pepp_initial_builder.pore.config import ensure_pore_dirs, pore_root


def write_lammps_relax_inputs(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_pore_dirs(config)
    root = pore_root(config)
    out = root / config["paths"]["full_pore_seed_structures_dir"] / "lammps_relax_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    seed_manifest = root / config["paths"]["full_pore_seed_structures_dir"] / "full_pore_seed_manifest.csv"
    rows = []
    if seed_manifest.exists():
        for row in pd.read_csv(seed_manifest).to_dict("records"):
            rows.append(
                {
                    "full_pore_seed_id": row.get("full_pore_seed_id", ""),
                    "status": "not_generated_requires_lammps_ready_combined_topology",
                    "raw_seed_extxyz_path": row.get("extxyz_path", ""),
                    "relaxed_extxyz_path": "",
                    "relax_is_training_data": False,
                    "relax_is_production_md": False,
                }
            )
    else:
        rows.append({"full_pore_seed_id": "no_full_pore_seed_manifest", "status": "skipped_no_full_pore_seed_manifest", "raw_seed_extxyz_path": "", "relaxed_extxyz_path": "", "relax_is_training_data": False, "relax_is_production_md": False})
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
