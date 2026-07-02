from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Atom:
    atom_id: int
    element: str
    atom_type: str
    polymer_type: str
    chain_id: int
    chain_type: str
    backbone_index: int
    is_backbone: bool
    is_segment_center: bool
    is_side_group: bool
    is_hydrogen: bool
    parent_segment_id: int
    x: float
    y: float
    z: float
    molecule_id: int = 0
    charge: float = 0.0


@dataclass
class Bond:
    bond_id: int
    atom1: int
    atom2: int
    bond_type: str = "generic"


@dataclass
class Angle:
    angle_id: int
    atom1: int
    atom2: int
    atom3: int
    angle_type: str = "generic"


@dataclass
class Chain:
    chain_id: int
    chain_type: str
    chain_length_backbone: int
    atom_ids: List[int] = field(default_factory=list)
    backbone_atom_ids: List[int] = field(default_factory=list)


@dataclass
class SystemTopology:
    atoms: List[Atom]
    bonds: List[Bond]
    angles: List[Angle]
    chains: List[Chain]
    box: Tuple[float, float, float]
    system_id: str
    builder_used: str = "emc"
    topology_source: str = "emc"
    coordinate_source: str = "emc"

    def atom_by_id(self) -> dict[int, Atom]:
        return {atom.atom_id: atom for atom in self.atoms}
