from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from pepp_initial_builder.common.tables import Angle, Atom, Bond, Chain, SystemTopology
from pepp_initial_builder.polymer.chain_builder import build_angles


def load_topology_from_tables(system_dir: str | Path) -> SystemTopology:
    system_dir = Path(system_dir)
    atom_df = pd.read_csv(system_dir / "atom_table.csv")
    bond_df = pd.read_csv(system_dir / "bond_table.csv")
    angle_df = pd.read_csv(system_dir / "angle_table.csv") if (system_dir / "angle_table.csv").exists() else pd.DataFrame()
    meta = yaml.safe_load((system_dir / "metadata.yaml").read_text(encoding="utf-8"))
    atoms = [
        Atom(int(row.atom_id), str(row.element), str(row.atom_type), str(row.polymer_type), int(row.chain_id), str(row.chain_type), int(row.backbone_index), bool(row.is_backbone), bool(row.is_segment_center), bool(row.is_side_group), bool(row.is_hydrogen), int(row.parent_segment_id), float(row.x), float(row.y), float(row.z), int(row.molecule_id), float(row.charge))
        for _, row in atom_df.iterrows()
    ]
    bonds = [Bond(int(row.bond_id), int(row.atom1), int(row.atom2), str(row.bond_type)) for _, row in bond_df.iterrows()]
    angles = [Angle(int(row.angle_id), int(row.atom1), int(row.atom2), int(row.atom3), str(row.angle_type)) for _, row in angle_df.iterrows()] if not angle_df.empty else build_angles(bonds)
    chains = []
    for chain_id, group in atom_df.groupby("chain_id"):
        chain = Chain(int(chain_id), str(group.iloc[0]["chain_type"]), int(group["backbone_index"].max()))
        chain.atom_ids = [int(x) for x in group["atom_id"]]
        chain.backbone_atom_ids = [int(x) for x in group[group["is_backbone"] == True]["atom_id"]]
        chains.append(chain)
    box = meta["box"]
    return SystemTopology(atoms, bonds, angles, chains, (float(box["box_lx_A"]), float(box["box_ly_A"]), float(box["box_lz_A"])), meta["system_id"], meta["builder"]["builder_used"], meta["builder"]["topology_source"], meta["builder"]["coordinate_source"])
