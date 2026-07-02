from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np


def mlff_cell_min(config: Dict[str, Any]) -> float:
    assumptions = config.get("mlff_assumptions", {})
    cutoff = float(assumptions.get("planned_mlff_cutoff_A", 5.0))
    multiple = float(assumptions.get("min_cell_multiple_of_cutoff", 2.0))
    absolute = float(assumptions.get("min_cell_length_A", 12.0))
    return max(absolute, multiple * cutoff)


def vec3_text(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.8f}" for value in values)


def pore_center(box: Tuple[float, float, float]) -> np.ndarray:
    return np.array([box[0] / 2.0, box[1] / 2.0, box[2] / 2.0], dtype=float)


def anchor_indices(coords: np.ndarray, center: np.ndarray, patch_type: str, radius: float, max_atoms: int) -> np.ndarray:
    xy = coords[:, :2] - center[:2]
    radial = np.linalg.norm(xy, axis=1)
    order = np.argsort(radial)
    anchor = order[-1] if "concave" in patch_type or "wall" in patch_type or len(order) else 0
    if "flat" in patch_type and len(order):
        anchor = order[len(order) // 2]
    distances = np.linalg.norm(coords - coords[anchor], axis=1)
    keep = np.where(distances <= radius)[0]
    if len(keep) < min(max_atoms, len(coords)):
        keep = np.argsort(distances)[: min(max_atoms, len(coords))]
    return np.array(sorted(keep[: min(max_atoms, len(keep))]), dtype=int)


def normal_metadata(coords: np.ndarray, box: Tuple[float, float, float], keep: np.ndarray) -> Dict[str, Any]:
    center = pore_center(box)
    patch_center = coords[keep].mean(axis=0) if len(keep) else center
    radial = np.array([patch_center[0] - center[0], patch_center[1] - center[1], 0.0], dtype=float)
    norm = float(np.linalg.norm(radial))
    if norm > 1.0e-8:
        outward = radial / norm
        status = "box_center_radial"
    else:
        outward = np.array([1.0, 0.0, 0.0], dtype=float)
        radial = outward.copy()
        status = "weak_default_box_center"
    inward = -outward
    return {"pore_axis": "z", "pore_center_xyz": vec3_text(center), "radial_vector_xyz": vec3_text(radial), "inward_normal_xyz": vec3_text(inward), "normal_definition_status": status, "patch_center_xyz": vec3_text(patch_center), "patch_center_array": patch_center, "inward_normal_array": inward}


def surface_classification(elems: Sequence[str], coords: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    o_idx = [i for i, element in enumerate(elems) if element == "O"]
    h_idx = [i for i, element in enumerate(elems) if element == "H"]
    si_idx = [i for i, element in enumerate(elems) if element == "Si"]
    n_silanol = 0
    n_siloxane = 0
    for oi in o_idx:
        has_oh = any(float(np.linalg.norm(coords[oi] - coords[hi])) <= 1.25 for hi in h_idx)
        si_neighbors = sum(1 for si in si_idx if float(np.linalg.norm(coords[oi] - coords[si])) <= 2.1)
        if has_oh:
            n_silanol += 1
        if not has_oh and si_neighbors >= 2:
            n_siloxane += 1
    n_o = len(o_idx)
    min_oh = int(config.get("silica_patch", {}).get("min_surface_oh_count_for_silanol_rich", 2))
    oh_ratio = n_silanol / n_o if n_o else 0.0
    silox_ratio = n_siloxane / n_o if n_o else 0.0
    if n_silanol >= min_oh and n_siloxane >= min_oh:
        surface_class = "mixed_oh"
    elif n_silanol >= min_oh and oh_ratio >= silox_ratio:
        surface_class = "silanol_rich"
    elif n_siloxane > n_silanol:
        surface_class = "siloxane_rich"
    else:
        surface_class = "mixed_oh"
    confidence = "high" if n_o >= 12 and (n_silanol or n_siloxane) else "medium" if n_o >= 4 else "low"
    return {"surface_class": surface_class, "surface_classification_method": "distance_based_heuristic", "classification_confidence": confidence, "n_silanol_oh": n_silanol, "n_siloxane_o": n_siloxane, "n_surface_si": len(si_idx), "n_surface_o": n_o, "local_oh_count": n_silanol}


def rebuild_local_cell(coords: np.ndarray, box: Tuple[float, float, float], config: Dict[str, Any]):
    patch_cfg = config.get("silica_patch", {})
    min_cell = mlff_cell_min(config)
    vacuum = float(patch_cfg.get("vacuum_buffer_A", 5.0))
    if len(coords):
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        extent = maxs - mins
        local_box = tuple(float(max(min_cell, extent[i] + 2.0 * vacuum)) for i in range(3))
        centroid = coords.mean(axis=0)
        target = np.array([local_box[0] / 2.0, local_box[1] / 2.0, local_box[2] / 2.0])
        shift = target - centroid
        shifted = coords + shift
        status = "centered_on_local_cell_center"
    else:
        local_box = (min_cell, min_cell, min_cell)
        shift = np.zeros(3)
        shifted = coords
        status = "no_atoms_to_center"
    meta = {"cell_source": "rebuilt_local_patch_cell" if patch_cfg.get("rebuild_local_cell", True) else "original_pore_cell", "original_pore_cell_lx_A": float(box[0]), "original_pore_cell_ly_A": float(box[1]), "original_pore_cell_lz_A": float(box[2]), "coordinate_centering_status": status, "local_cell_origin_shift_xyz": vec3_text(shift)}
    return shifted, local_box, meta


def cell_failure_reason(box: Tuple[float, float, float], config: Dict[str, Any]) -> str:
    min_cell = mlff_cell_min(config)
    return "" if all(float(x) >= min_cell for x in box) else "cell_smaller_than_2x_mlff_cutoff"


def normal_from_row(row: Any) -> np.ndarray:
    text = str(row.get("inward_normal_xyz", "-1 0 0"))
    try:
        values = np.array([float(x) for x in text.split()[:3]], dtype=float)
    except Exception:
        values = np.array([-1.0, 0.0, 0.0], dtype=float)
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 1.0e-8 else np.array([-1.0, 0.0, 0.0], dtype=float)


def heavy_min_distance(elems: Sequence[str], coords: np.ndarray, left_count: int) -> float:
    left = [i for i in range(left_count) if elems[i] != "H"]
    right = [i for i in range(left_count, len(elems)) if elems[i] != "H"]
    if not left or not right:
        return float("inf")
    return min(float(np.linalg.norm(coords[i] - coords[j])) for i in left for j in right)


def estimate_pore_radius_A(pore_row: Any, box: Tuple[float, float, float], coords: np.ndarray) -> float:
    try:
        diameter_nm = float(pore_row.get("pore_diameter_nm", ""))
        if diameter_nm > 0:
            return diameter_nm * 10.0 / 2.0
    except Exception:
        pass
    center = pore_center(box)
    radial = np.linalg.norm(coords[:, :2] - center[:2], axis=1) if len(coords) else np.array([min(box[:2]) / 2.0])
    return max(3.0, float(np.percentile(radial, 80)))


def inside_pore_fraction(coords: np.ndarray, box: Tuple[float, float, float], radius: float, wall_buffer: float, end_buffer: float) -> float:
    if len(coords) == 0:
        return 0.0
    center = pore_center(box)
    radial = np.linalg.norm(coords[:, :2] - center[:2], axis=1)
    inside = (radial < max(radius - wall_buffer, 0.1)) & (coords[:, 2] > end_buffer) & (coords[:, 2] < box[2] - end_buffer)
    return float(np.count_nonzero(inside) / len(coords))


def min_cross_distance(elems: Sequence[str], coords: np.ndarray, split: int) -> float:
    left = [i for i in range(split) if elems[i] != "H"]
    right = [i for i in range(split, len(elems)) if elems[i] != "H"]
    if not left or not right:
        return float("inf")
    return min(float(np.linalg.norm(coords[i] - coords[j])) for i in left for j in right)
