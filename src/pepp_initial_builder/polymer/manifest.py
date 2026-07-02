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
        cleanup_status = json.loads((system_dir / "cleanup_status.json").read_text()) if (system_dir / "cleanup_status.json").exists() else {}
        cleaned = bool(cleanup_status.get("cleanup_performed", meta.get("cleanup", {}).get("cleanup_performed", False)))
        kind = "cleaned_initial" if cleaned else "raw_initial"
        validation = validate_system(system_dir)
        rows.append({"system_id": meta["system_id"], "builder_used": meta["builder"]["builder_used"], "topology_source": meta["builder"]["topology_source"], "coordinate_source": meta["builder"]["coordinate_source"], "mlff_start_structure_kind": kind, "mlff_start_xyz_path": str(system_dir / f"{kind}.xyz"), "mlff_start_extxyz_path": str(system_dir / f"{kind}.extxyz"), "mlff_start_pdb_path": str(system_dir / f"{kind}.pdb"), "mlff_start_lammps_data_path": str(system_dir / f"{kind}.data"), "raw_initial_xyz_path": str(system_dir / "raw_initial.xyz"), "cleaned_initial_xyz_path": str(system_dir / "cleaned_initial.xyz"), "metadata_yaml_path": str(meta_path), "atom_table_path": str(system_dir / "atom_table.csv"), "bond_table_path": str(system_dir / "bond_table.csv"), "segment_table_path": str(system_dir / "segment_table.csv"), "chain_table_path": str(system_dir / "chain_table.csv"), "pe_fraction_actual": meta["composition"]["pe_fraction_actual"], "pp_fraction_actual": meta["composition"]["pp_fraction_actual"], "pe_mass_fraction_actual": meta["composition"]["pe_mass_fraction_actual"], "pp_mass_fraction_actual": meta["composition"]["pp_mass_fraction_actual"], "chain_length_backbone": meta["polymer_metadata"]["chain_length_definition"], "total_atoms_actual": validation.get("atom_count"), "density_definition": meta["density"]["density_definition"], "initial_packing_density_g_cm3": meta["density"]["initial_packing_density_g_cm3"], "not_equilibrium_density": meta["density"]["not_equilibrium_density"], "planned_downstream_density_scales": json.dumps(meta["condition_for_later_mlff"]["planned_downstream_density_scales"]), "target_temperature_K_for_later_mlff": meta["condition_for_later_mlff"]["target_temperature_K"], "target_pressure_atm_for_later_mlff": meta["condition_for_later_mlff"]["target_pressure_atm"], "pp_tacticity": meta["polymer_metadata"]["pp_tacticity"], "cleanup_performed": cleaned, "cleanup_is_training_data": False, "usable_for_mlff_start": validation.get("usable_for_mlff_start", False), "box_lx_A": meta["box"]["box_lx_A"], "box_ly_A": meta["box"]["box_ly_A"], "box_lz_A": meta["box"]["box_lz_A"], "pbc": True})
    out = project_root(config) / config["paths"]["exports_dir"]
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "mlff_start_manifest.csv"
    json_path = out / "mlff_start_manifest.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path
