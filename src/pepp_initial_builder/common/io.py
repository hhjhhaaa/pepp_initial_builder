from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

from pepp_initial_builder.common.constants import ATOM_TYPE_IDS, ATOM_TYPE_MASS
from pepp_initial_builder.common.tables import Atom, SystemTopology


def write_pdb(path: str | Path, atoms: Sequence[Atom], box: Tuple[float, float, float]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"CRYST1{box[0]:9.3f}{box[1]:9.3f}{box[2]:9.3f}{90.0:7.2f}{90.0:7.2f}{90.0:7.2f} P 1           1\n")
        for atom in atoms:
            handle.write(
                f"HETATM{atom.atom_id:5d} {atom.atom_type[:4].ljust(4)} {atom.chain_type[:3].rjust(3)} A{atom.chain_id % 10000:4d}    "
                f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}  1.00  0.00          {atom.element:>2s}\n"
            )
        handle.write("END\n")


def write_xyz(path: str | Path, atoms: Sequence[Atom], box: Tuple[float, float, float], ext: bool = False) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{len(atoms)}\n")
        if ext:
            handle.write(f'Lattice="{box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]}" Properties=species:S:1:pos:R:3 pbc="T T T"\n')
        else:
            handle.write(f"box_lx_A={box[0]:.6f} box_ly_A={box[1]:.6f} box_lz_A={box[2]:.6f} pbc=true\n")
        for atom in atoms:
            handle.write(f"{atom.element} {atom.x:.8f} {atom.y:.8f} {atom.z:.8f}\n")


def write_lammps_data(path: str | Path, topo: SystemTopology) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            f"LAMMPS data for {topo.system_id}; atom_style full; cleanup-only zero charges\n\n"
            f"{len(topo.atoms)} atoms\n{len(topo.bonds)} bonds\n{len(topo.angles)} angles\n\n"
            f"{len(ATOM_TYPE_IDS)} atom types\n2 bond types\n1 angle types\n\n"
        )
        lx, ly, lz = topo.box
        handle.write(f"0.0 {lx:.8f} xlo xhi\n0.0 {ly:.8f} ylo yhi\n0.0 {lz:.8f} zlo zhi\n\nMasses\n\n")
        for type_id, (name, mass) in ATOM_TYPE_MASS.items():
            handle.write(f"{type_id} {mass:.6f} # {name}\n")
        handle.write("\nAtoms # full\n\n")
        for atom in topo.atoms:
            handle.write(f"{atom.atom_id} {atom.molecule_id} {ATOM_TYPE_IDS[atom.atom_type]} 0.000000 {atom.x:.8f} {atom.y:.8f} {atom.z:.8f}\n")
        handle.write("\nBonds\n\n")
        for bond in topo.bonds:
            handle.write(f"{bond.bond_id} {1 if bond.bond_type == 'C-C' else 2} {bond.atom1} {bond.atom2}\n")
        handle.write("\nAngles\n\n")
        for angle in topo.angles:
            handle.write(f"{angle.angle_id} 1 {angle.atom1} {angle.atom2} {angle.atom3}\n")
