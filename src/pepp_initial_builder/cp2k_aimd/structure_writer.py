from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from pepp_initial_builder.common.io import write_xyz
from pepp_initial_builder.pore.config import ensure_pore_dirs, pore_root
from pepp_initial_builder.pore.porems_builder import AIMD_ELEMENTS, atoms_from_elements, read_xyz_like


def write_cp2k_structure_inputs(config: Dict[str, Any]) -> Path:
    ensure_pore_dirs(config)
    outbase = pore_root(config) / config["paths"]["aimd_exports_dir"]
    rows: List[Dict[str, Any]] = []
    manifest = pore_root(config) / config["paths"]["aimd_local_structures_dir"] / "aimd_local_manifest.csv"
    if manifest.exists():
        df = pd.read_csv(manifest)
        for _, row in df[df["status"].astype(str) == "available"].iterrows():
            src = Path(row["extxyz_path"])
            structure_dir = outbase / row["aimd_structure_id"]
            structure_dir.mkdir(parents=True, exist_ok=True)
            dst = structure_dir / "structure.xyz"
            elems, coords, box = read_xyz_like(src)
            atoms = atoms_from_elements(elems, coords)
            write_xyz(dst, atoms, box, ext=False)
            (structure_dir / "cp2k_structure_metadata.yaml").write_text(yaml.safe_dump({"cp2k_run_performed": False, "aimd_trajectory_generated": False, "source_extxyz": str(src)}, sort_keys=False), encoding="utf-8")
            rows.append({"aimd_structure_id": row["aimd_structure_id"], "status": "structure_input_written_no_cp2k_run", "cp2k_xyz_path": str(dst)})
    if not rows:
        rows.append({"aimd_structure_id": "no_available_aimd_structure", "status": "skipped_no_available_aimd_structure", "cp2k_xyz_path": ""})
    out = outbase / "cp2k_structure_input_manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
