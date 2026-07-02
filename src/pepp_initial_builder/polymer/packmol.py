from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from pepp_initial_builder.common.io import write_pdb
from pepp_initial_builder.common.tables import Atom, SystemTopology
from pepp_initial_builder.common.tools import discover_tools


def write_chain_template(path: Path, atoms: Sequence[Atom]) -> None:
    local_atoms = [Atom(i, atom.element, atom.atom_type, atom.polymer_type, 1, atom.chain_type, atom.backbone_index, atom.is_backbone, atom.is_segment_center, atom.is_side_group, atom.is_hydrogen, 0, atom.x, atom.y, atom.z, 1, 0.0) for i, atom in enumerate(atoms, 1)]
    write_pdb(path, local_atoms, (80.0, 80.0, 80.0))


def write_packmol_inputs(system_dir: Path, topo: SystemTopology, config: Dict[str, Any], row: Dict[str, Any]) -> Path:
    packmol_dir = system_dir / "packmol"
    template_dir = packmol_dir / "chain_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    atom_map = topo.atom_by_id()
    lines = [f"tolerance {config['packmol']['tolerance_A']}", "filetype pdb", f"output {packmol_dir / 'raw_packmol.pdb'}", f"seed {int(row['seed'])}", f"maxit {int(config['packmol']['maxit'])}", ""]
    for chain in topo.chains:
        path = template_dir / f"chain_{chain.chain_id:04d}_{chain.chain_type}.pdb"
        write_chain_template(path, [atom_map[i] for i in chain.atom_ids])
        lx, ly, lz = topo.box
        lines += [f"structure {path}", "  number 1", f"  inside box 0.0 0.0 0.0 {lx:.6f} {ly:.6f} {lz:.6f}", "end structure", ""]
    inp = packmol_dir / "packmol.inp"
    inp.write_text("\n".join(lines), encoding="utf-8")
    return inp


def parse_pdb_coords(path: Path):
    coords = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return coords


def apply_coords(topo: SystemTopology, coords, source: str) -> None:
    if len(coords) != len(topo.atoms):
        raise ValueError(f"Coordinate count mismatch: got {len(coords)} expected {len(topo.atoms)}")
    for atom, (x, y, z) in zip(topo.atoms, coords):
        atom.x = float(x)
        atom.y = float(y)
        atom.z = float(z)
    topo.coordinate_source = source


def internal_pack(topo: SystemTopology, seed: int) -> None:
    rng = random.Random(seed + 77)
    atom_map = topo.atom_by_id()
    lx, ly, lz = topo.box
    for chain in topo.chains:
        arr = np.array([[atom_map[i].x, atom_map[i].y, atom_map[i].z] for i in chain.atom_ids])
        shift = np.array([rng.uniform(0.1 * lx, 0.9 * lx), rng.uniform(0.1 * ly, 0.9 * ly), rng.uniform(0.1 * lz, 0.9 * lz)]) - arr.mean(axis=0)
        for atom_id in chain.atom_ids:
            atom = atom_map[atom_id]
            atom.x = (atom.x + shift[0]) % lx
            atom.y = (atom.y + shift[1]) % ly
            atom.z = (atom.z + shift[2]) % lz
    topo.coordinate_source = "python_internal_coordinates"


def run_packmol(system_dir: Path, topo: SystemTopology, config: Dict[str, Any], row: Dict[str, Any]):
    inp = write_packmol_inputs(system_dir, topo, config, row)
    exe = discover_tools(config)["packmol"]["executable"]
    log = system_dir / "packmol" / "packmol.log"
    if not exe or not Path(exe).exists():
        internal_pack(topo, int(row["seed"]))
        log.write_text("Packmol missing; used internal coordinates\n", encoding="utf-8")
        return False, "packmol_missing_internal_coordinates_used"
    try:
        with open(inp, "rb") as fin, open(log, "wb") as fout:
            subprocess.run([exe], stdin=fin, stdout=fout, stderr=subprocess.STDOUT, cwd=inp.parent, timeout=int(config["packmol"].get("timeout_seconds", 300)), check=True)
        apply_coords(topo, parse_pdb_coords(inp.parent / "raw_packmol.pdb"), "packmol_coordinates")
        return True, "packmol_success"
    except Exception as exc:
        internal_pack(topo, int(row["seed"]))
        prior = log.read_text(errors="ignore") if log.exists() else ""
        log.write_text(prior + f"\nPackmol failed; used internal coordinates: {exc}\n", encoding="utf-8")
        return False, f"packmol_failed_internal_coordinates_used: {exc}"
