from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from pepp_initial_builder.common.paths import ensure_dirs, project_root
from pepp_initial_builder.polymer.validation import validate_system


def export_manifest(config: Dict[str, Any]):
    ensure_dirs(config)
    rows = []
    systems_dir = project_root(config) / config["paths"]["systems_dir"]
    for system_dir in sorted(systems_dir.glob("pepp_*")):
        meta_path = system_dir / "metadata.yaml"
        if not meta_path.exists():
            continue
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        validation = validate_system(system_dir)
        rows.append({"system_id": meta["system_id"], "builder_used": meta["builder"]["builder_used"], "topology_source": meta["builder"]["topology_source"], "coordinate_source": meta["builder"]["coordinate_source"], "mlff_start_structure_kind": "emc_polymer", "mlff_start_xyz_path": meta["paths"]["xyz"], "mlff_start_extxyz_path": meta["paths"]["extxyz"], "mlff_start_pdb_path": meta["paths"]["pdb"], "mlff_start_lammps_data_path": meta["paths"]["lammps_data"], "metadata_yaml_path": str(meta_path), "pe_fraction_actual": meta["composition"]["pe_fraction_actual"], "pp_fraction_actual": meta["composition"]["pp_fraction_actual"], "total_atoms_actual": validation.get("atom_count"), "usable_for_mlff_start": validation.get("usable_for_mlff_start", False)})
    out = project_root(config) / config["paths"]["exports_dir"]
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "mlff_start_manifest.csv"
    json_path = out / "mlff_start_manifest.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path
