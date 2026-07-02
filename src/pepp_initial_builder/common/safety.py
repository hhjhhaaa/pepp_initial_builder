from __future__ import annotations

from pathlib import Path
from typing import List


def assert_no_fake_cp2k_or_mlff(root: Path) -> List[str]:
    forbidden = []
    for name in ["aimd.traj", "cp2k.out", "mlff_production", "production.lammpstrj", "prod.lammpstrj"]:
        forbidden.extend(str(path) for path in root.rglob(name))
    return forbidden
