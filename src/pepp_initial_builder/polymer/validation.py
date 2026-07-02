from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from pepp_initial_builder.common.paths import ensure_dirs, project_root
from pepp_initial_builder.polymer.chain_builder import select_rows
from pepp_initial_builder.polymer.topology import load_topology_from_tables


def relation_exclusions(topo, exclude_14: bool = True):
    exclusions = set()
    neighbors = {}
    for bond in topo.bonds:
        exclusions.add(tuple(sorted((bond.atom1, bond.atom2))))
        neighbors.setdefault(bond.atom1, set()).add(bond.atom2)
        neighbors.setdefault(bond.atom2, set()).add(bond.atom1)
    for center, ns in neighbors.items():
        for atom1, atom2 in combinations(ns, 2):
            exclusions.add(tuple(sorted((atom1, atom2))))
    if exclude_14:
        for atom1 in list(neighbors):
            for atom2 in neighbors.get(atom1, []):
                for atom3 in neighbors.get(atom2, []):
                    if atom3 == atom1:
                        continue
                    for atom4 in neighbors.get(atom3, []):
                        if atom4 not in (atom1, atom2):
                            exclusions.add(tuple(sorted((atom1, atom4))))
    return exclusions


def validate_system(system_dir: Path, heavy_threshold: float = 1.8) -> Dict[str, Any]:
    required = ["raw_initial.xyz", "raw_initial.extxyz", "raw_initial.pdb", "raw_initial.data", "metadata.yaml", "atom_table.csv", "bond_table.csv", "segment_table.csv"]
    result: Dict[str, Any] = {"system_id": system_dir.name, "usable_for_mlff_start": True}
    missing = [name for name in required if not (system_dir / name).exists()]
    result["missing_files"] = ";".join(missing)
    if missing:
        result["usable_for_mlff_start"] = False
        return result
    topo = load_topology_from_tables(system_dir)
    result.update({"atom_count": len(topo.atoms), "bond_count": len(topo.bonds), "segment_center_count": sum(atom.is_segment_center for atom in topo.atoms), "backbone_carbon_count": sum(atom.is_backbone and atom.element == "C" for atom in topo.atoms), "box_positive": all(x > 0 for x in topo.box)})
    neighbors = {atom.atom_id: [] for atom in topo.atoms}
    atom_map = topo.atom_by_id()
    for bond in topo.bonds:
        neighbors[bond.atom1].append(bond.atom2)
        neighbors[bond.atom2].append(bond.atom1)
    bad_c = [atom.atom_id for atom in topo.atoms if atom.element == "C" and len(neighbors[atom.atom_id]) != 4]
    bad_h = [atom.atom_id for atom in topo.atoms if atom.element == "H" and (len(neighbors[atom.atom_id]) != 1 or atom_map[neighbors[atom.atom_id][0]].element != "C")]
    result["bad_carbon_valence_count"] = len(bad_c)
    result["bad_hydrogen_count"] = len(bad_h)
    pp_side = sum(atom.atom_type == "PP_CH3_SIDE" for atom in topo.atoms)
    pp_backbone = sum(atom.polymer_type == "PP" and atom.is_backbone for atom in topo.atoms)
    result["pp_side_methyl_count"] = pp_side
    result["pp_side_methyl_expected"] = pp_backbone // 2
    exclusions = relation_exclusions(topo, True)
    heavy = [atom for atom in topo.atoms if atom.element != "H"]
    overlaps = 0
    min_distance = 999.0
    for i, atom in enumerate(heavy):
        va = np.array([atom.x, atom.y, atom.z])
        for other in heavy[i + 1 :]:
            if tuple(sorted((atom.atom_id, other.atom_id))) in exclusions:
                continue
            distance = float(np.linalg.norm(va - np.array([other.x, other.y, other.z])))
            min_distance = min(min_distance, distance)
            if distance < heavy_threshold:
                overlaps += 1
            if overlaps > 20:
                break
        if overlaps > 20:
            break
    forbidden = []
    for name in ["prod.lammpstrj", "production.lammpstrj"]:
        if list(system_dir.rglob(name)):
            forbidden.append(name)
    result["nonbonded_heavy_overlap_count"] = overlaps
    result["min_nonbonded_heavy_distance_A"] = None if min_distance == 999.0 else min_distance
    result["forbidden_production_trajectory_files"] = ";".join(forbidden)
    result["usable_for_mlff_start"] = not bad_c and not bad_h and result["box_positive"] and not forbidden and result["segment_center_count"] == result["backbone_carbon_count"]
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
