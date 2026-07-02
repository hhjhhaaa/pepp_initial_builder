from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def apply_boundary_caps(elems: Sequence[str], coords: np.ndarray, selected_indices: Sequence[int], config: Dict[str, Any]) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    selected = set(int(i) for i in selected_indices)
    if not selected:
        return [], np.zeros((0, 3), dtype=float), {
            "polymer_cut_and_capped": False,
            "silica_boundary_hydroxylated": False,
            "boundary_atom_count": 0,
            "cap_atom_count": 0,
            "boundary_status": "unresolved_crop_boundary",
        }
    crop_elems = [elems[int(i)] for i in selected_indices]
    crop_coords = coords[list(selected_indices)].copy()
    cap_coords: List[np.ndarray] = []
    for idx in selected:
        if elems[idx] != "C":
            continue
        near_unselected_c = [
            j
            for j, elem in enumerate(elems)
            if elem == "C" and j not in selected and float(np.linalg.norm(coords[idx] - coords[j])) <= 1.8
        ]
        if not near_unselected_c:
            continue
        outside_idx = min(near_unselected_c, key=lambda j: float(np.linalg.norm(coords[idx] - coords[j])))
        direction = coords[outside_idx] - coords[idx]
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-8:
            continue
        cap_coords.append(coords[idx] + direction / norm * 1.09)
    silica_boundary = any(elems[idx] in {"Si", "O"} for idx in selected)
    if cap_coords:
        crop_elems.extend("H" for _ in cap_coords)
        crop_coords = np.vstack([crop_coords, np.array(cap_coords, dtype=float)])
    cap_count = len(cap_coords)
    return crop_elems, crop_coords, {
        "polymer_cut_and_capped": bool(cap_count),
        "silica_boundary_hydroxylated": bool(silica_boundary),
        "boundary_atom_count": cap_count,
        "cap_atom_count": cap_count,
        "boundary_status": "resolved_by_distance_heuristic",
    }


def boundary_is_usable(treatment: Dict[str, Any]) -> bool:
    return treatment.get("boundary_status") != "unresolved_crop_boundary"
