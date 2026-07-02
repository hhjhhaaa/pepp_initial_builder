from __future__ import annotations

from pathlib import Path

import pandas as pd

from pepp_initial_builder.common.io import write_lammps_data, write_pdb, write_xyz
from pepp_initial_builder.common.tables import SystemTopology


def write_tables(system_dir: Path, topo: SystemTopology) -> None:
    pd.DataFrame([atom.__dict__ for atom in topo.atoms]).to_csv(system_dir / "atom_table.csv", index=False)
    pd.DataFrame([bond.__dict__ for bond in topo.bonds]).to_csv(system_dir / "bond_table.csv", index=False)
    pd.DataFrame([angle.__dict__ for angle in topo.angles]).to_csv(system_dir / "angle_table.csv", index=False)
    pd.DataFrame([{"chain_id": chain.chain_id, "chain_type": chain.chain_type, "chain_length_backbone": chain.chain_length_backbone, "n_atoms": len(chain.atom_ids), "n_backbone_carbons": len(chain.backbone_atom_ids)} for chain in topo.chains]).to_csv(system_dir / "chain_table.csv", index=False)
    pd.DataFrame([{"segment_id": atom.atom_id, "atom_id": atom.atom_id, "chain_id": atom.chain_id, "polymer_type": atom.polymer_type, "backbone_index": atom.backbone_index, "is_segment_center": True} for atom in topo.atoms if atom.is_segment_center]).to_csv(system_dir / "segment_table.csv", index=False)


def write_structure_outputs(system_dir: Path, topo: SystemTopology, prefix: str = "raw_initial") -> None:
    write_xyz(system_dir / f"{prefix}.extxyz", topo.atoms, topo.box, True)
    write_pdb(system_dir / f"{prefix}.pdb", topo.atoms, topo.box)
    write_lammps_data(system_dir / f"{prefix}.data", topo)
    write_xyz(system_dir / f"{prefix}.xyz", topo.atoms, topo.box, False)
