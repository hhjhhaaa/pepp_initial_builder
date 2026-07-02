from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from pepp_initial_builder.common.io import write_pdb, write_xyz
from pepp_initial_builder.cp2k_aimd.boundary_capping import apply_boundary_caps, boundary_is_usable
from pepp_initial_builder.cp2k_aimd.local_environment_selector import LocalEnvironment
from pepp_initial_builder.pore.porems_builder import atoms_from_elements
from pepp_initial_builder.pore.surface_classifier import rebuild_local_cell, vec3_text


def crop_limits(config: Dict[str, Any], mode: str) -> tuple[int, float]:
    cfg = config.get("cp2k_crop", {})
    max_atoms = int(cfg.get(f"{mode}_max_atoms", cfg.get("max_atoms", 100 if mode == "tiny" else 200)))
    radius = float(cfg.get(f"{mode}_crop_radius_A", cfg.get("crop_radius_A", 6.0 if mode == "tiny" else 8.0)))
    return max_atoms, radius


def selected_indices(coords: np.ndarray, center_idx: int, max_atoms: int, radius: float) -> np.ndarray:
    distances = np.linalg.norm(coords - coords[center_idx], axis=1)
    keep = np.where(distances <= radius)[0]
    if len(keep) > max_atoms:
        keep = np.argsort(distances)[:max_atoms]
    return np.array(sorted(int(i) for i in keep), dtype=int)


def write_atom_mapping(path: Path, selected: Sequence[int], elems: Sequence[str], n_crop_atoms: int) -> None:
    rows = [
        {"crop_atom_id": idx + 1, "source_atom_index": int(source_idx), "source_atom_id": int(source_idx) + 1, "element": elems[int(source_idx)]}
        for idx, source_idx in enumerate(selected)
    ]
    for idx in range(len(rows), n_crop_atoms):
        rows.append({"crop_atom_id": idx + 1, "source_atom_index": "", "source_atom_id": "", "element": "H", "source_role": "cap_generated"})
    pd.DataFrame(rows).to_csv(path, index=False)


def build_crop(
    source: Dict[str, str],
    env: LocalEnvironment,
    elems: Sequence[str],
    coords: np.ndarray,
    box: Tuple[float, float, float],
    outdir: Path,
    aimd_seed_id: str,
    config: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    max_atoms, radius = crop_limits(config, mode)
    base_atom_budget = max(1, int(max_atoms * 0.9))
    keep = selected_indices(coords, env.center_atom_index, base_atom_budget, radius)
    crop_elems, crop_coords, treatment = apply_boundary_caps(elems, coords, keep, config)
    if len(crop_elems) > max_atoms:
        crop_elems = crop_elems[:max_atoms]
        crop_coords = crop_coords[:max_atoms]
        treatment["boundary_status"] = "unresolved_crop_boundary"
    shifted, local_box, cell_meta = rebuild_local_cell(crop_coords, box, config)
    usable = boundary_is_usable(treatment) and len(crop_elems) <= max_atoms
    failure_reason = "" if usable else "unresolved_crop_boundary"
    outdir.mkdir(parents=True, exist_ok=True)
    atoms = atoms_from_elements(crop_elems, shifted, 0)
    extxyz = outdir / "input.extxyz"
    pdb = outdir / "input.pdb"
    write_xyz(extxyz, atoms, local_box, ext=True)
    write_pdb(pdb, atoms, local_box)
    write_atom_mapping(outdir / "atom_mapping.csv", keep, elems, len(crop_elems))
    boundary_meta = {
        "polymer_cut_and_capped": treatment["polymer_cut_and_capped"],
        "silica_boundary_hydroxylated": treatment["silica_boundary_hydroxylated"],
        "boundary_atom_count": treatment["boundary_atom_count"],
        "cap_atom_count": treatment["cap_atom_count"],
        "boundary_status": treatment["boundary_status"],
    }
    metadata = {
        "aimd_seed_id": aimd_seed_id,
        "family": env.selection_reason,
        "crop_source": "full_pore_snapshot",
        "source_stage": source["source_stage"],
        "source_full_pore_id": source["source_full_pore_id"],
        "source_snapshot_path": source["source_snapshot_path"],
        "source_frame_index": source["source_frame_index"],
        "selection_reason": env.selection_reason,
        "what_local_environment_it_teaches": env.what_local_environment_it_teaches,
        "center_atom_id": env.center_atom_index + 1,
        "center_atom_element": env.center_atom_element,
        "nearest_wall_distance_A": env.nearest_wall_distance_A,
        "local_polymer_density": env.local_polymer_density,
        "local_PE_fraction": env.local_PE_fraction,
        "local_PP_fraction": env.local_PP_fraction,
        "surface_class": env.surface_class,
        "boundary_treatment": boundary_meta,
        "cap_atom_count": treatment["cap_atom_count"],
        "n_atoms": len(crop_elems),
        "crop_radius_A": radius,
        "usable_for_cp2k_aimd": bool(usable),
        "failure_reason": failure_reason,
        "status": "available" if usable else "unusable",
        "input_extxyz_path": str(extxyz),
        "input_pdb_path": str(pdb),
        "local_cell_lx_A": local_box[0],
        "local_cell_ly_A": local_box[1],
        "local_cell_lz_A": local_box[2],
        "local_cell_origin_shift_xyz": cell_meta["local_cell_origin_shift_xyz"],
        "coordinate_centering_status": cell_meta["coordinate_centering_status"],
        "crop_center_xyz": vec3_text(coords[env.center_atom_index]),
    }
    (outdir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    (outdir / "crop_summary.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return metadata
