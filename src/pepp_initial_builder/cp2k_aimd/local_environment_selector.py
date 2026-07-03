from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

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
    metadata: Dict[str, Any] | None = None


def silica_indices(elems: Sequence[str]) -> List[int]:
    return [idx for idx, elem in enumerate(elems) if elem in {"Si", "O"}]


def polymer_indices(elems: Sequence[str]) -> List[int]:
    return [idx for idx, elem in enumerate(elems) if elem == "C"]


def nearest_silica_distance(coords: np.ndarray, polymer_idx: int, silica: Sequence[int]) -> float:
    if not silica:
        return float("inf")
    return min(float(np.linalg.norm(coords[polymer_idx] - coords[j])) for j in silica)


def surface_site_metadata(elems: Sequence[str], coords: np.ndarray, center_idx: int) -> Dict[str, Any]:
    center = coords[center_idx]
    oxygen = [idx for idx, elem in enumerate(elems) if elem == "O"]
    hydrogen = [idx for idx, elem in enumerate(elems) if elem == "H"]
    silicon = [idx for idx, elem in enumerate(elems) if elem == "Si"]
    silanol_o: List[int] = []
    siloxane_o: List[int] = []
    for idx in oxygen:
        oh = any(float(np.linalg.norm(coords[idx] - coords[h_idx])) <= 1.25 for h_idx in hydrogen)
        si_neighbors = sum(1 for si_idx in silicon if float(np.linalg.norm(coords[idx] - coords[si_idx])) <= 2.15)
        if oh:
            silanol_o.append(idx)
        elif si_neighbors > 0:
            siloxane_o.append(idx)

    def distances(indices: Sequence[int]) -> List[float]:
        return [float(np.linalg.norm(center - coords[idx])) for idx in indices]

    oh_dist = distances(silanol_o)
    siloxane_dist = distances(siloxane_o)
    nearest_oh = min(oh_dist) if oh_dist else float("inf")
    nearest_siloxane = min(siloxane_dist) if siloxane_dist else float("inf")
    if nearest_oh == float("inf") and nearest_siloxane == float("inf"):
        nearest_type = "unknown"
    elif nearest_oh <= nearest_siloxane:
        nearest_type = "silanol_OH"
    else:
        nearest_type = "siloxane_O"
    return {
        "nearest_site_type": nearest_type,
        "n_silanol_OH_within_5A": sum(1 for value in oh_dist if value <= 5.0),
        "n_siloxane_O_within_5A": sum(1 for value in siloxane_dist if value <= 5.0),
        "nearest_OH_distance_A": "" if nearest_oh == float("inf") else nearest_oh,
        "nearest_siloxane_O_distance_A": "" if nearest_siloxane == float("inf") else nearest_siloxane,
    }


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


def _load_atom_roles(source: Dict[str, str]) -> pd.DataFrame:
    path = source.get("atom_roles_path", "")
    if not path:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _surface_family(surface_class: str) -> str:
    text = surface_class.lower()
    if "siloxane" in text:
        return "siloxane"
    return "silanol"


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
    per_family = int(crop_cfg.get("patches_per_family_per_system", 3))
    environments: List[LocalEnvironment] = []
    pe_fraction, pp_fraction = composition_from_source(source)
    roles = _load_atom_roles(source)
    requested_families = config.get("aimd_local_matrix", {}).get("families", [])
    role_by_atom = {}
    if not roles.empty and "atom_id" in roles.columns:
        role_by_atom = {int(row["atom_id"]): row for _, row in roles.iterrows()}

    def add_environment(center_idx: int, family: str, teaches: str, extra: Dict[str, Any] | None = None) -> None:
        if len([env for env in environments if env.selection_reason == family]) >= per_family:
            return
        local = np.where(np.linalg.norm(coords - coords[center_idx], axis=1) <= radius)[0]
        local_polymer_count = sum(1 for idx in local if elems[int(idx)] == "C")
        local_volume = max(4.0 / 3.0 * np.pi * radius**3, 1.0)
        surface = surface_classification([elems[int(i)] for i in local], coords[local], config)
        nearest = nearest_silica_distance(coords, center_idx, silica) if elems[center_idx] == "C" else 0.0
        nearest_silica = min(silica, key=lambda idx: float(np.linalg.norm(coords[center_idx] - coords[idx]))) if silica else -1
        n_pp_methyl = 0
        n_pe_c = 0
        for idx in local:
            atom_id = int(idx) + 1
            role = role_by_atom.get(atom_id)
            if role is None:
                continue
            if str(role.get("atom_role", "")) == "PP_side_methyl_C":
                n_pp_methyl += 1
            if str(role.get("polymer_type", "")) == "PE" and elems[int(idx)] == "C":
                n_pe_c += 1
        meta = {
            "crop_family": family,
            **surface_site_metadata(elems, coords, center_idx),
            "center_atom_role": extra.get("center_atom_role", "") if extra else "",
            "parent_backbone_atom_id": extra.get("parent_backbone_atom_id", "") if extra else "",
            "nearest_silica_atom_id": "" if nearest_silica < 0 else nearest_silica + 1,
            "nearest_silica_element": "" if nearest_silica < 0 else elems[nearest_silica],
            "methyl_wall_alignment": extra.get("methyl_wall_alignment", "") if extra else "",
            "methyl_orientation_class": extra.get("methyl_orientation_class", "") if extra else "",
            "n_PP_side_methyl_within_5A": n_pp_methyl,
            "n_PE_backbone_C_within_5A": n_pe_c,
            **(extra or {}),
        }
        environments.append(
            LocalEnvironment(
                selection_reason=family,
                what_local_environment_it_teaches=teaches,
                center_atom_index=int(center_idx),
                center_atom_element=elems[center_idx],
                nearest_wall_distance_A=float(nearest),
                local_polymer_density=float(local_polymer_count / local_volume),
                local_PE_fraction=pe_fraction,
                local_PP_fraction=pp_fraction,
                surface_class=str(surface["surface_class"]),
                metadata=meta,
            )
        )

    if any(f.startswith("PP_methyl") for f in requested_families):
        if not roles.empty and {"atom_id", "atom_role", "is_side_group", "parent_backbone_atom_id"}.issubset(set(roles.columns)):
            methyl_rows = roles[
                (roles["atom_role"].astype(str) == "PP_side_methyl_C")
                & (roles["is_side_group"].astype(str).str.lower().isin(["true", "1", "yes"]))
                & (roles["parent_backbone_atom_id"].astype(str) != "")
            ]
            for _, role in methyl_rows.iterrows():
                idx = int(role["atom_id"]) - 1
                if idx < 0 or idx >= len(elems):
                    continue
                nearest = nearest_silica_distance(coords, idx, silica)
                if nearest < 1.8 or nearest > 5.0:
                    continue
                local = np.where(np.linalg.norm(coords - coords[idx], axis=1) <= radius)[0]
                surface = surface_classification([elems[int(i)] for i in local], coords[local], config)
                family = f"PP_methyl_{_surface_family(str(surface['surface_class']))}_contact"
                if family not in requested_families:
                    family = "PP_methyl_silanol_contact" if "PP_methyl_silanol_contact" in requested_families else family
                if family in requested_families:
                    add_environment(
                        idx,
                        family,
                        "PP side methyl steric/dispersion contact with silanol or siloxane silica wall.",
                        {
                            "center_atom_role": "PP_side_methyl_C",
                            "parent_backbone_atom_id": role.get("parent_backbone_atom_id", ""),
                            "methyl_orientation_class": role.get("methyl_orientation_class", ""),
                        },
                    )

    ranked = sorted(polymer, key=lambda idx: nearest_silica_distance(coords, idx, silica))
    for center_idx in ranked:
        if len(environments) >= limit:
            break
        local = np.where(np.linalg.norm(coords - coords[center_idx], axis=1) <= radius)[0]
        reason, teaches = reason_for_environment(elems, coords, center_idx, local, source)
        family_map = {
            "pe_ch2_wall_contact": "PE_HDPE_CH2_silanol_contact",
            "pe_pp_mixed_near_wall_contact": "PE_PP_mixed_near_wall",
            "compressed_polymer_wall_contact": "crowded_polymer_wall_contact",
            "pp_methyl_wall_contact": "PP_backbone_CH_silanol_contact",
        }
        nearest = nearest_silica_distance(coords, center_idx, silica)
        if nearest < 2.2:
            reason = "compressed_polymer_wall_contact"
            teaches = "crowded polymer-wall contact under full-pore packing."
        family = family_map.get(reason, "crowded_polymer_wall_contact")
        if family in requested_families:
            add_environment(center_idx, family, teaches)

    if "silica_only_wall_baseline" in requested_families and len(environments) < limit and silica:
        add_environment(silica[0], "silica_only_wall_baseline", "baseline wall vibration and silanol flexibility")
    return environments
