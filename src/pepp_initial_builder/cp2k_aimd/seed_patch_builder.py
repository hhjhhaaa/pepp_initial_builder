from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from pepp_initial_builder.common.io import write_pdb, write_xyz
from pepp_initial_builder.pore.config import ensure_pore_dirs, pore_root
from pepp_initial_builder.pore.patch_crop import patch_rows
from pepp_initial_builder.pore.porems_builder import atoms_from_elements, read_xyz_like
from pepp_initial_builder.pore.surface_classifier import heavy_min_distance, normal_from_row
from pepp_initial_builder.polymer.chain_builder import build_python_topology


def build_aimd_local_structures(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_pore_dirs(config)
    patches = patch_rows(config)
    outbase = pore_root(config) / config["paths"]["aimd_local_structures_dir"]
    maxn = int(config["aimd_local_matrix"][f"{mode}_max_structures"])
    rows: List[Dict[str, Any]] = []
    if patches.empty:
        rows.append({"aimd_structure_id": "no_available_silica_patch", "status": "skipped_no_available_silica_patch", "source_patch_id": "", "extxyz_path": ""})
    else:
        families = config["aimd_local_matrix"]["families"]
        seeds = config["aimd_local_matrix"]["seeds"]
        made = 0
        for _, patch in patches.iterrows():
            elems, coords, box = read_xyz_like(Path(patch["patch_extxyz_path"]))
            for family in families:
                for seed in seeds:
                    if made >= maxn:
                        break
                    sid = f"aimd_{made + 1:04d}_{family}_seed{seed}"
                    structure_dir = outbase / sid
                    structure_dir.mkdir(parents=True, exist_ok=True)
                    pe = "pe" in family
                    pp = "pp" in family
                    all_elems = list(elems)
                    all_coords = [x for x in coords]
                    rng = np.random.default_rng(seed)
                    placement_status = "silica_patch_only"
                    min_polymer_silica_distance = float("inf")
                    if pe or pp:
                        row = {"system_id": "fragment", "seed": seed, "chain_length_backbone": 8, "n_pe_chains": 1 if pe else 0, "n_pp_chains": 1 if pp else 0, "initial_packing_density_g_cm3": 0.5}
                        topo = build_python_topology(row)
                        normal = normal_from_row(patch)
                        patch_center = np.array([box[0] / 2.0, box[1] / 2.0, box[2] / 2.0])
                        distances = [float(x) for x in config["aimd_local_matrix"].get("polymer_wall_distances_A", [3.5])]
                        distance = distances[0] if "compressed" in family else distances[min(seed - 1, len(distances) - 1)]
                        polymer_coords = np.array([[atom.x, atom.y, atom.z] for atom in topo.atoms], dtype=float)
                        polymer_coords -= polymer_coords.mean(axis=0)
                        placed = polymer_coords + patch_center + normal * distance + rng.normal(0, 0.03, polymer_coords.shape)
                        min_allowed = float(config.get("packing", {}).get("min_heavy_atom_distance_A", 1.6))
                        for _attempt in range(40):
                            trial_coords = np.vstack([np.array(all_coords, dtype=float), placed])
                            trial_elems = all_elems + [atom.element for atom in topo.atoms]
                            min_polymer_silica_distance = heavy_min_distance(trial_elems, trial_coords, len(all_elems))
                            if min_polymer_silica_distance >= min_allowed:
                                placement_status = "placed_along_inward_normal"
                                break
                            placed = placed + normal * 0.25
                        if placement_status != "placed_along_inward_normal":
                            rows.append({"aimd_structure_id": sid, "status": "skipped_overlap_unresolved", "source_patch_id": patch["patch_id"], "family": family, "extxyz_path": "", "min_polymer_silica_distance_A": min_polymer_silica_distance})
                            made += 1
                            continue
                        for atom, xyz in zip(topo.atoms, placed):
                            all_elems.append(atom.element)
                            all_coords.append(xyz)
                    atoms = atoms_from_elements(all_elems, np.array(all_coords), 0)
                    write_xyz(structure_dir / "structure.extxyz", atoms, box, ext=True)
                    write_pdb(structure_dir / "structure.pdb", atoms, box)
                    (structure_dir / "metadata.yaml").write_text(yaml.safe_dump({"aimd_structure_id": sid, "family": family, "source_patch_id": patch["patch_id"], "patch_type": patch.get("patch_type", ""), "status": "available", "placement_status": placement_status, "placement_direction": "inward_normal", "inward_normal_xyz": patch.get("inward_normal_xyz", ""), "min_polymer_silica_distance_A": min_polymer_silica_distance if math.isfinite(min_polymer_silica_distance) else None, "purpose": "aimd_local_training_structure_only_no_cp2k_run"}, sort_keys=False), encoding="utf-8")
                    rows.append({"aimd_structure_id": sid, "status": "available", "source_patch_id": patch["patch_id"], "patch_id": patch["patch_id"], "patch_type": patch.get("patch_type", ""), "family": family, "extxyz_path": str(structure_dir / "structure.extxyz"), "placement_status": placement_status, "inward_normal_xyz": patch.get("inward_normal_xyz", ""), "min_polymer_silica_distance_A": min_polymer_silica_distance if math.isfinite(min_polymer_silica_distance) else ""})
                    made += 1
                if made >= maxn:
                    break
    manifest = outbase / "aimd_local_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


build_seed_structures = build_aimd_local_structures
