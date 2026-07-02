from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml

from pepp_initial_builder.common.io import write_pdb, write_xyz
from pepp_initial_builder.common.tools import discover_tools
from pepp_initial_builder.common.paths import project_root
from pepp_initial_builder.polymer.chain_builder import select_rows
from pepp_initial_builder.polymer.packmol import apply_coords
from pepp_initial_builder.polymer.topology import load_topology_from_tables


def cleanup_inputs_text(config: Dict[str, Any]):
    cleanup = config["lammps_cleanup"]
    cutoff = cleanup["pair_cutoff_A"]
    soft = f"""units real
atom_style full
read_data raw_initial.data
pair_style soft {cutoff}
pair_coeff * * 5.0
bond_style zero
bond_coeff *
angle_style zero
angle_coeff *
thermo {cleanup['thermo_stride']}
variable prefactor equal ramp(5.0,50.0)
fix push all adapt 1 pair soft a * * v_prefactor
fix int all nve/limit 0.05
run {cleanup['soft_push_steps']}
unfix int
unfix push
write_data cleaned_initial.data
write_dump all xyz cleaned_initial.xyz modify sort id
"""
    final = f"""units real
atom_style full
read_data cleaned_initial.data
pair_style lj/cut {cutoff}
pair_coeff * * 0.02 3.5
bond_style zero
bond_coeff *
angle_style zero
angle_coeff *
thermo {cleanup['thermo_stride']}
minimize 1.0e-4 1.0e-6 {cleanup['minimize_maxiter']} {cleanup['minimize_maxiter']}
write_data cleaned_initial.data
write_dump all xyz cleanup_check.lammpstrj modify sort id
"""
    nvt = f"""units real
atom_style full
read_data cleaned_initial.data
pair_style lj/cut {cutoff}
pair_coeff * * 0.02 3.5
bond_style zero
bond_coeff *
angle_style zero
angle_coeff *
velocity all create 523.0 4928459 mom yes rot yes dist gaussian
fix int all nve/limit 0.05
thermo {cleanup['thermo_stride']}
run {cleanup['short_nvt_steps']}
unfix int
write_data cleaned_initial.data
write_dump all xyz cleanup_check.lammpstrj modify sort id
"""
    return {"in.00_minimize.lmp": final, "in.01_soft_push.lmp": soft, "in.02_short_nvt_cleanup.lmp": nvt}


def write_cleanup_inputs(config: Dict[str, Any], tiny: bool = False, pilot: bool = False, max_systems: int | None = None):
    outputs = []
    for row in select_rows(config, tiny, pilot, max_systems):
        system_dir = project_root(config) / config["paths"]["systems_dir"] / row["system_id"]
        system_dir.mkdir(parents=True, exist_ok=True)
        for name, text in cleanup_inputs_text(config).items():
            path = system_dir / name
            path.write_text(text, encoding="utf-8")
            outputs.append(path)
    return outputs


def run_cleanup(config: Dict[str, Any], tiny: bool = False, pilot: bool = False, max_systems: int | None = None):
    exe = discover_tools(config)["lammps"]["executable"]
    outputs = []
    for row in select_rows(config, tiny, pilot, max_systems):
        system_dir = project_root(config) / config["paths"]["systems_dir"] / row["system_id"]
        (system_dir / "logs").mkdir(parents=True, exist_ok=True)
        status = {"cleanup_performed": False, "cleanup_method": "soft_then_lj", "minimize_success": False, "soft_push_success": False, "short_nvt_success": False, "cleanup_is_training_data": False, "cleanup_is_production_md": False}
        if exe and Path(exe).exists() and (system_dir / "raw_initial.data").exists():
            for name, text in cleanup_inputs_text(config).items():
                (system_dir / name).write_text(text, encoding="utf-8")
            try:
                with open(system_dir / "logs/cleanup.log", "wb") as log:
                    subprocess.run([exe, "-in", "in.01_soft_push.lmp"], cwd=system_dir, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True)
                status.update({"cleanup_performed": True, "minimize_success": True, "soft_push_success": True})
                try:
                    topo = load_topology_from_tables(system_dir)
                    lines = (system_dir / "cleaned_initial.xyz").read_text(encoding="utf-8", errors="ignore").splitlines()[2:]
                    coords = []
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 4:
                            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    if len(coords) == len(topo.atoms):
                        apply_coords(topo, coords, "lammps_cleanup_coordinates")
                        write_xyz(system_dir / "cleaned_initial.extxyz", topo.atoms, topo.box, True)
                        write_pdb(system_dir / "cleaned_initial.pdb", topo.atoms, topo.box)
                        write_xyz(system_dir / "cleaned_initial.xyz", topo.atoms, topo.box, False)
                except Exception as exc:
                    status["cleaned_aux_output_warning"] = str(exc)
                with open(system_dir / "logs/cleanup.log", "ab") as log:
                    try:
                        subprocess.run([exe, "-in", "in.02_short_nvt_cleanup.lmp"], cwd=system_dir, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True)
                        status["short_nvt_success"] = True
                    except Exception as exc:
                        status["short_nvt_failure_reason"] = str(exc)
            except Exception as exc:
                status["failure_reason"] = str(exc)
        else:
            status["failure_reason"] = "lammps_missing_or_raw_initial_missing"
        status["mlff_start_structure"] = "cleaned_initial.data" if status.get("cleanup_performed") else "raw_initial.data"
        try:
            meta_path = system_dir / "metadata.yaml"
            if meta_path.exists():
                meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
                meta["cleanup"] = status.copy()
                meta["cleanup"]["cleanup_method"] = status.get("cleanup_method", "soft_then_lj")
                meta["paths"]["mlff_start_structure"] = str(system_dir / status["mlff_start_structure"])
                meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        except Exception as exc:
            status["metadata_update_warning"] = str(exc)
        (system_dir / "cleanup_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        outputs.append(system_dir / "cleanup_status.json")
    return outputs
