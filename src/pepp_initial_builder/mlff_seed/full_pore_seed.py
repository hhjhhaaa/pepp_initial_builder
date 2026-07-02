from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from pepp_initial_builder.common.io import write_pdb, write_xyz
from pepp_initial_builder.pore.config import ensure_pore_dirs, pore_root
from pepp_initial_builder.pore.porems_builder import atoms_from_elements, available_pore_rows, read_xyz_like
from pepp_initial_builder.pore.surface_classifier import estimate_pore_radius_A, inside_pore_fraction, min_cross_distance, pore_center
from pepp_initial_builder.polymer.chain_builder import build_python_topology


def build_full_pore_seed_structures(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_pore_dirs(config)
    pores = available_pore_rows(config)
    outbase = pore_root(config) / config["paths"]["full_pore_seed_structures_dir"]
    maxn = int(config["full_pore_seed_matrix"][f"{mode}_max_systems"])
    rows: List[Dict[str, Any]] = []
    if pores.empty:
        rows.append({"full_pore_seed_id": "no_available_pore_model", "status": "skipped_no_available_pore_model", "source_pore_model_id": "", "extxyz_path": ""})
    else:
        made = 0
        for _, pore in pores.iterrows():
            elems, coords, box = read_xyz_like(Path(pore["pore_model_extxyz_path"]))
            seed_cfg = config.get("full_pore_seed", {})
            wall_buffer = float(seed_cfg.get("wall_buffer_A", 3.0))
            end_buffer = float(seed_cfg.get("end_buffer_A", 3.0))
            min_inside = float(seed_cfg.get("min_polymer_inside_pore_fraction", 0.95))
            min_silica = float(seed_cfg.get("min_polymer_silica_distance_A", 1.6))
            pore_radius = estimate_pore_radius_A(pore, box, coords)
            for pe, pp in config["full_pore_seed_matrix"]["pe_pp_compositions"]:
                if made >= maxn:
                    break
                seed = int(config["full_pore_seed_matrix"]["seeds"][0])
                sid = f"full_pore_seed_{made + 1:04d}_PE{int(pe * 100):02d}_PP{int(pp * 100):02d}_seed{seed}"
                structure_dir = outbase / sid
                structure_dir.mkdir(parents=True, exist_ok=True)
                row = {"system_id": sid, "seed": seed, "chain_length_backbone": 80, "n_pe_chains": 1 if pe > 0 else 0, "n_pp_chains": 1 if pp > 0 else 0, "initial_packing_density_g_cm3": 0.5}
                topo = build_python_topology(row)
                all_elems = list(elems)
                all_coords = [x for x in coords]
                polymer_raw = np.array([[atom.x, atom.y, atom.z] for atom in topo.atoms], dtype=float)
                polymer_raw -= polymer_raw.mean(axis=0)
                radial = np.linalg.norm(polymer_raw[:, :2], axis=1)
                allowed_radius = max(pore_radius - wall_buffer, 0.5)
                max_radial = max(float(radial.max()), 1.0e-8)
                if max_radial > allowed_radius:
                    polymer_raw[:, :2] *= allowed_radius / max_radial * 0.95
                z_span = float(polymer_raw[:, 2].max() - polymer_raw[:, 2].min()) if len(polymer_raw) else 0.0
                allowed_z = max(box[2] - 2.0 * end_buffer, 1.0)
                if z_span > allowed_z:
                    polymer_raw[:, 2] *= allowed_z / z_span * 0.95
                placed_polymer = polymer_raw + pore_center(box)
                all_elems.extend(atom.element for atom in topo.atoms)
                all_coords.extend(placed_polymer)
                combined_coords = np.array(all_coords, dtype=float)
                inside_fraction = inside_pore_fraction(placed_polymer, box, pore_radius, wall_buffer, end_buffer)
                min_distance = min_cross_distance(all_elems, combined_coords, len(elems))
                usable = inside_fraction >= min_inside and min_distance >= min_silica
                packing_status = "packed_inside_pore" if usable else "polymer_not_inside_pore_or_overlap"
                atoms = atoms_from_elements(all_elems, combined_coords, 0)
                write_xyz(structure_dir / "seed.extxyz", atoms, box, ext=True)
                write_pdb(structure_dir / "seed.pdb", atoms, box)
                relaxation = {"lammps_relax_performed": False, "relax_is_training_data": False, "relax_is_production_md": False, "mlff_start_structure_kind": "raw_full_pore_seed", "mlff_start_extxyz_path": str(structure_dir / "seed.extxyz")}
                (structure_dir / "metadata.yaml").write_text(yaml.safe_dump({"full_pore_seed_id": sid, "status": "available", "source_pore_model_id": pore["pore_model_id"], "purpose": "mlff_exploration_seed_only_no_mlff_run", "packing_method": "deterministic_cylindrical_sampler", "packing_status": packing_status, "usable_for_mlff_start": bool(usable), "failure_reason": "" if usable else "polymer_not_inside_pore_or_overlap", "polymer_inside_pore_fraction": inside_fraction, "min_polymer_silica_distance_A": min_distance, "relaxation": relaxation}, sort_keys=False), encoding="utf-8")
                rows.append({"full_pore_seed_id": sid, "status": "available", "source_pore_model_id": pore["pore_model_id"], "extxyz_path": str(structure_dir / "seed.extxyz"), "polymer_inside_pore_fraction": inside_fraction, "min_polymer_silica_distance_A": min_distance, "packing_method": "deterministic_cylindrical_sampler", "packing_status": packing_status, "usable_for_mlff_start": bool(usable), "failure_reason": "" if usable else "polymer_not_inside_pore_or_overlap", "mlff_start_structure_kind": "raw_full_pore_seed"})
                made += 1
    manifest = outbase / "full_pore_seed_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


pack_polymer_into_pore = build_full_pore_seed_structures
write_packmol_inputs = build_full_pore_seed_structures
write_lammps_relax_inputs = build_full_pore_seed_structures
