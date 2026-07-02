from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import yaml

from pepp_initial_builder.common.constants import AMU_TO_G, MASS
from pepp_initial_builder.common.paths import ensure_dirs, project_root
from pepp_initial_builder.common.tables import Angle, Atom, Bond, Chain, SystemTopology
from pepp_initial_builder.common.tools import write_discovery_report
from pepp_initial_builder.polymer.matrix import matrix_rows


def unit(rng: random.Random) -> np.ndarray:
    vec = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)], float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1.0e-12 else np.array([1.0, 0.0, 0.0])


def walk(n: int, rng: random.Random, step: float = 1.54, min_nonbond: float = 2.0) -> List[np.ndarray]:
    coords = [np.zeros(3)]
    direction = unit(rng)
    for i in range(1, n):
        accepted = None
        for _ in range(400):
            new_direction = unit(rng)
            if np.dot(new_direction, direction) < -0.35:
                continue
            trial = coords[-1] + step * new_direction
            if all(i - j <= 2 or np.linalg.norm(trial - old) >= min_nonbond for j, old in enumerate(coords[:-1])):
                accepted = trial
                direction = new_direction
                break
        coords.append(accepted if accepted is not None else coords[-1] + step * unit(rng))
    arr = np.array(coords)
    arr -= arr.mean(axis=0)
    return [np.array(x) for x in arr]


def add_atom(atoms: List[Atom], element: str, atom_type: str, polymer_type: str, chain: Chain, backbone_index: int, is_backbone: bool, is_side_group: bool, parent: int, pos: Sequence[float]) -> int:
    atom_id = len(atoms) + 1
    is_hydrogen = element == "H"
    atoms.append(
        Atom(
            atom_id,
            element,
            atom_type,
            polymer_type,
            chain.chain_id,
            chain.chain_type,
            backbone_index,
            is_backbone,
            is_backbone and element == "C",
            is_side_group,
            is_hydrogen,
            parent,
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            chain.chain_id,
            0.0,
        )
    )
    chain.atom_ids.append(atom_id)
    if is_backbone:
        chain.backbone_atom_ids.append(atom_id)
    return atom_id


def make_chain(chain_type: str, n: int, chain_id: int, rng: random.Random):
    if chain_type == "PP" and n % 2:
        raise ValueError("PP chain_length_backbone must be even for -CH2-CH(CH3)- repeat pattern")
    atoms: List[Atom] = []
    bonds: List[Bond] = []
    chain = Chain(chain_id, chain_type, n)
    backbone = walk(n, rng)
    ids = []
    for i, pos in enumerate(backbone):
        atom_type = "PE_C" if chain_type == "PE" else ("PP_CH2" if i % 2 == 0 else "PP_CH")
        atom_id = add_atom(atoms, "C", atom_type, chain_type, chain, i + 1, True, False, 0, pos)
        atoms[-1].parent_segment_id = atom_id
        ids.append(atom_id)
    for atom1, atom2 in zip(ids[:-1], ids[1:]):
        bonds.append(Bond(len(bonds) + 1, atom1, atom2, "C-C"))
    if chain_type == "PP":
        for i in range(1, n, 2):
            tangent = (backbone[i + 1] - backbone[i - 1]) if 0 < i < n - 1 else (backbone[i] - backbone[i - 1])
            tangent = tangent / (np.linalg.norm(tangent) + 1.0e-12)
            best = None
            for _try in range(200):
                direction = unit(rng)
                direction = direction - np.dot(direction, tangent) * tangent
                direction = direction / (np.linalg.norm(direction) + 1.0e-12)
                direction *= -1 if rng.random() < 0.5 else 1
                trial = backbone[i] + 1.54 * direction
                existing = [pos for j, pos in enumerate(backbone) if abs(j - i) > 2]
                for old_atom in atoms:
                    if old_atom.element == "C" and old_atom.atom_id != ids[i]:
                        existing.append(np.array([old_atom.x, old_atom.y, old_atom.z]))
                if all(np.linalg.norm(trial - pos) >= 2.0 for pos in existing):
                    best = direction
                    break
            if best is None:
                best = direction
            side_id = add_atom(atoms, "C", "PP_CH3_SIDE", chain_type, chain, i + 1, False, True, ids[i], backbone[i] + 1.54 * best)
            bonds.append(Bond(len(bonds) + 1, ids[i], side_id, "C-C"))
    neighbors = {atom.atom_id: [] for atom in atoms}
    for bond in bonds:
        neighbors[bond.atom1].append(bond.atom2)
        neighbors[bond.atom2].append(bond.atom1)
    for carbon in list(atoms):
        for _ in range(4 - len(neighbors[carbon.atom_id])):
            hydrogen_id = add_atom(atoms, "H", "H", chain_type, chain, carbon.backbone_index, False, False, carbon.atom_id, np.array([carbon.x, carbon.y, carbon.z]) + 1.09 * unit(rng))
            bonds.append(Bond(len(bonds) + 1, carbon.atom_id, hydrogen_id, "C-H"))
    return atoms, bonds, chain


def offset_chain(atoms: Sequence[Atom], bonds: Sequence[Bond], chain: Chain, atom_offset: int, bond_offset: int, new_chain_id: int):
    id_map = {atom.atom_id: atom.atom_id + atom_offset for atom in atoms}
    new_chain = Chain(new_chain_id, chain.chain_type, chain.chain_length_backbone)
    out_atoms = []
    for atom in atoms:
        new_atom = Atom(
            id_map[atom.atom_id],
            atom.element,
            atom.atom_type,
            atom.polymer_type,
            new_chain_id,
            atom.chain_type,
            atom.backbone_index,
            atom.is_backbone,
            atom.is_segment_center,
            atom.is_side_group,
            atom.is_hydrogen,
            id_map.get(atom.parent_segment_id, atom.parent_segment_id),
            atom.x,
            atom.y,
            atom.z,
            new_chain_id,
            0.0,
        )
        out_atoms.append(new_atom)
        new_chain.atom_ids.append(new_atom.atom_id)
        if new_atom.is_backbone:
            new_chain.backbone_atom_ids.append(new_atom.atom_id)
    out_bonds = [Bond(i + bond_offset + 1, id_map[bond.atom1], id_map[bond.atom2], bond.bond_type) for i, bond in enumerate(bonds)]
    return out_atoms, out_bonds, new_chain


def build_angles(bonds: Sequence[Bond]) -> List[Angle]:
    neighbors: Dict[int, List[int]] = {}
    for bond in bonds:
        neighbors.setdefault(bond.atom1, []).append(bond.atom2)
        neighbors.setdefault(bond.atom2, []).append(bond.atom1)
    angles = []
    for center, ns in sorted(neighbors.items()):
        for atom1, atom3 in combinations(sorted(ns), 2):
            angles.append(Angle(len(angles) + 1, atom1, center, atom3, "generic"))
    return angles


def box_length_from_atoms(atoms: Sequence[Atom], density: float) -> float:
    return (sum(MASS[atom.element] for atom in atoms) * AMU_TO_G / density * 1.0e24) ** (1 / 3)


def build_python_topology(row: Dict[str, Any]) -> SystemTopology:
    rng = random.Random(int(row["seed"]))
    atoms: List[Atom] = []
    bonds: List[Bond] = []
    chains: List[Chain] = []
    chain_id = 1
    n = int(row["chain_length_backbone"])
    for chain_type, count in [("PE", int(row["n_pe_chains"])), ("PP", int(row["n_pp_chains"]))]:
        for _ in range(count):
            chain_atoms, chain_bonds, chain = make_chain(chain_type, n, chain_id, rng)
            offset_atoms, offset_bonds, offset = offset_chain(chain_atoms, chain_bonds, chain, len(atoms), len(bonds), chain_id)
            atoms += offset_atoms
            bonds += offset_bonds
            chains.append(offset)
            chain_id += 1
    length = box_length_from_atoms(atoms, float(row["initial_packing_density_g_cm3"]))
    return SystemTopology(atoms, bonds, build_angles(bonds), chains, (length, length, length), row["system_id"])


def emc_attempt(config: Dict[str, Any], row: Dict[str, Any], system_dir: Path) -> Dict[str, Any]:
    emc_dir = system_dir / "emc"
    emc_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "attempted": True,
        "success": False,
        "timeout_seconds": int(config.get("emc", {}).get("attempt_timeout_seconds", 300)),
        "failure_reason": "emc_no_verified_all_atom_pe_pp_recipe; switched_to_packmol_fallback",
    }
    (emc_dir / "emc_build.log").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def composition_masses(topo: SystemTopology) -> Tuple[float, float]:
    pe = sum(MASS[atom.element] for atom in topo.atoms if atom.polymer_type == "PE")
    pp = sum(MASS[atom.element] for atom in topo.atoms if atom.polymer_type == "PP")
    total = pe + pp
    return (pe / total if total else 0.0, pp / total if total else 0.0)


def write_metadata(system_dir: Path, topo: SystemTopology, row: Dict[str, Any], config: Dict[str, Any], emc: Dict[str, Any], cleanup: Dict[str, Any] | None = None) -> None:
    pe_mass, pp_mass = composition_masses(topo)
    cleanup = cleanup or {"cleanup_performed": False, "cleanup_method": "none", "cleanup_is_training_data": False, "cleanup_is_production_md": False}
    kind = "cleaned_initial" if cleanup.get("cleanup_performed") else "raw_initial"
    scales = json.loads(row["planned_downstream_density_scales"]) if isinstance(row["planned_downstream_density_scales"], str) else row["planned_downstream_density_scales"]
    meta = {
        "system_id": topo.system_id,
        "builder": {
            "builder_used": topo.builder_used,
            "representation": "all_atom_C_H",
            "emc_attempted": bool(emc.get("attempted")),
            "emc_success": bool(emc.get("success")),
            "emc_failure_reason": emc.get("failure_reason"),
            "packmol_attempted": True,
            "packmol_success": topo.coordinate_source == "packmol_coordinates",
            "packmol_fallback_used": True,
            "topology_source": topo.topology_source,
            "coordinate_source": topo.coordinate_source,
        },
        "condition_for_later_mlff": {
            "target_temperature_K": config["conditions_for_later_mlff"]["target_temperature_K"],
            "target_pressure_atm": config["conditions_for_later_mlff"]["target_pressure_atm"],
            "planned_downstream_density_scales": scales,
        },
        "density": {"density_definition": "initial_packing_density_only", "initial_packing_density_g_cm3": row["initial_packing_density_g_cm3"], "not_equilibrium_density": True, "rho_eq_generated_here": False},
        "composition": {"pe_fraction_actual": row["pe_fraction_actual"], "pp_fraction_actual": row["pp_fraction_actual"], "pe_mass_fraction_actual": pe_mass, "pp_mass_fraction_actual": pp_mass},
        "polymer_metadata": {"chain_length_definition": "backbone_carbon_number", "segment_center_definition": "backbone_carbons", "pp_tacticity": "atactic_like_v0", "atom_table_path": str(system_dir / "atom_table.csv"), "bond_table_path": str(system_dir / "bond_table.csv"), "segment_table_path": str(system_dir / "segment_table.csv"), "chain_table_path": str(system_dir / "chain_table.csv")},
        "box": {"box_lx_A": topo.box[0], "box_ly_A": topo.box[1], "box_lz_A": topo.box[2], "pbc": True},
        "paths": {"raw_initial_extxyz": str(system_dir / "raw_initial.extxyz"), "raw_initial_xyz": str(system_dir / "raw_initial.xyz"), "raw_initial_pdb": str(system_dir / "raw_initial.pdb"), "raw_initial_lammps_data": str(system_dir / "raw_initial.data"), "cleaned_initial_extxyz": str(system_dir / "cleaned_initial.extxyz"), "cleaned_initial_xyz": str(system_dir / "cleaned_initial.xyz"), "cleaned_initial_pdb": str(system_dir / "cleaned_initial.pdb"), "cleaned_initial_lammps_data": str(system_dir / "cleaned_initial.data"), "mlff_start_structure": str(system_dir / f"{kind}.data")},
        "cleanup": cleanup,
    }
    (system_dir / "metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


def append_emc_report(config: Dict[str, Any], row: Dict[str, Any], emc: Dict[str, Any], message: str) -> None:
    path = write_discovery_report(config)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\nEMC_ATTEMPT\n" + json.dumps({"system_id": row["system_id"], "emc": emc, "fallback": message}, indent=2) + "\n")


def build_system(config: Dict[str, Any], row: Dict[str, Any]) -> Path:
    from pepp_initial_builder.polymer.lammps_data import write_structure_outputs, write_tables
    from pepp_initial_builder.polymer.packmol import run_packmol

    system_dir = project_root(config) / config["paths"]["systems_dir"] / row["system_id"]
    system_dir.mkdir(parents=True, exist_ok=True)
    emc = emc_attempt(config, row, system_dir)
    topo = build_python_topology(row)
    _ok, message = run_packmol(system_dir, topo, config, row)
    topo.builder_used = "packmol_fallback"
    topo.topology_source = "python_all_atom_builder"
    write_structure_outputs(system_dir, topo)
    write_tables(system_dir, topo)
    write_metadata(system_dir, topo, row, config, emc)
    (system_dir / "system_summary.json").write_text(
        json.dumps({"system_id": topo.system_id, "total_atoms_actual": len(topo.atoms), "total_bonds_actual": len(topo.bonds), "total_angles_actual": len(topo.angles), "n_chains": len(topo.chains), "box_lx_A": topo.box[0], "box_ly_A": topo.box[1], "box_lz_A": topo.box[2], "pbc": True, "builder_used": topo.builder_used, "topology_source": topo.topology_source, "coordinate_source": topo.coordinate_source}, indent=2),
        encoding="utf-8",
    )
    append_emc_report(config, row, emc, message)
    return system_dir


def select_rows(config: Dict[str, Any], tiny: bool = False, pilot: bool = False, max_systems: int | None = None):
    rows = matrix_rows(config, "tiny" if tiny else "pilot" if pilot else "matrix")
    return rows[:max_systems] if max_systems is not None else rows


def build_systems(config: Dict[str, Any], tiny: bool = False, pilot: bool = False, max_systems: int | None = None):
    ensure_dirs(config)
    return [build_system(config, row) for row in select_rows(config, tiny, pilot, max_systems)]
