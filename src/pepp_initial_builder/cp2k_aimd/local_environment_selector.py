from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from pepp_initial_builder.pore.surface_classifier import surface_classification


@dataclass
class LocalEnvironment:
    selection_reason: str
    what_local_environment_it_teaches: str
    center_atom_index: int
    center_atom_element: str
    nearest_wall_distance_A: float
    local_polymer_density: float
    local_PE_fraction: float
    local_PP_fraction: float
    surface_class: str


def silica_indices(elems: Sequence[str]) -> List[int]:
    return [idx for idx, elem in enumerate(elems) if elem in {"Si", "O"}]


def polymer_indices(elems: Sequence[str]) -> List[int]:
    return [idx for idx, elem in enumerate(elems) if elem == "C"]


def nearest_silica_distance(coords: np.ndarray, polymer_idx: int, silica: Sequence[int]) -> float:
    if not silica:
        return float("inf")
    return min(float(np.linalg.norm(coords[polymer_idx] - coords[j])) for j in silica)


def composition_from_source(source: Dict[str, str]) -> tuple[float, float]:
    text = source.get("source_full_pore_id", "")
    if "PE100" in text:
        return 1.0, 0.0
    if "PP100" in text or "PE00_PP100" in text:
        return 0.0, 1.0
    if "PE50_PP50" in text:
        return 0.5, 0.5
    if "PE75_PP25" in text:
        return 0.75, 0.25
    if "PE25_PP75" in text:
        return 0.25, 0.75
    return 0.5, 0.5


def reason_for_environment(elems: Sequence[str], coords: np.ndarray, center_idx: int, local_indices: Sequence[int], source: Dict[str, str]) -> tuple[str, str]:
    pe_fraction, pp_fraction = composition_from_source(source)
    local_elements = [elems[i] for i in local_indices]
    n_polymer_c = sum(1 for elem in local_elements if elem == "C")
    n_silica_o = sum(1 for elem in local_elements if elem == "O")
    if n_polymer_c == 0:
        return "silica_only_wall_region", "baseline wall vibration and silanol flexibility"
    if pe_fraction > 0.0 and pp_fraction > 0.0:
        return "pe_pp_mixed_near_wall_contact", "near-wall PE/PP mixed local environment"
    if pp_fraction > pe_fraction:
        return "pp_methyl_wall_contact", "PP-rich polymer-wall contact environment"
    if pe_fraction > pp_fraction:
        return "pe_ch2_wall_contact", "PE-rich polymer-wall contact environment"
    if n_silica_o:
        return "polymer_silanol_or_siloxane_contact", "polymer contact with silica oxygen surface sites"
    return "crowded_polymer_wall_contact", "local polymer crowding near silica wall"


def select_local_environments(elems: Sequence[str], coords: np.ndarray, source: Dict[str, str], config: Dict[str, Any], limit: int) -> List[LocalEnvironment]:
    silica = silica_indices(elems)
    polymer = polymer_indices(elems)
    crop_cfg = config.get("cp2k_crop", {})
    radius = float(crop_cfg.get("crop_radius_A", crop_cfg.get("tiny_crop_radius_A", 6.0)))
    ranked = sorted(polymer, key=lambda idx: nearest_silica_distance(coords, idx, silica))
    centers = ranked[: max(limit, 1)] if ranked else silica[: max(limit, 1)]
    environments: List[LocalEnvironment] = []
    pe_fraction, pp_fraction = composition_from_source(source)
    for center_idx in centers:
        local = np.where(np.linalg.norm(coords - coords[center_idx], axis=1) <= radius)[0]
        local_polymer_count = sum(1 for idx in local if elems[int(idx)] == "C")
        local_volume = max(4.0 / 3.0 * np.pi * radius**3, 1.0)
        surface = surface_classification([elems[int(i)] for i in local], coords[local], config)
        nearest = nearest_silica_distance(coords, center_idx, silica) if elems[center_idx] == "C" else 0.0
        reason, teaches = reason_for_environment(elems, coords, center_idx, local, source)
        if nearest < 2.2 and elems[center_idx] == "C":
            reason = "compressed_polymer_wall_contact"
            teaches = "short non-pathological polymer-silica contact"
        environments.append(
            LocalEnvironment(
                selection_reason=reason,
                what_local_environment_it_teaches=teaches,
                center_atom_index=int(center_idx),
                center_atom_element=elems[center_idx],
                nearest_wall_distance_A=float(nearest),
                local_polymer_density=float(local_polymer_count / local_volume),
                local_PE_fraction=pe_fraction,
                local_PP_fraction=pp_fraction,
                surface_class=str(surface["surface_class"]),
            )
        )
    return environments
