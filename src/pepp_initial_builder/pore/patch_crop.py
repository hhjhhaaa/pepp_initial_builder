from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from pepp_initial_builder.common.io import write_pdb, write_xyz
from pepp_initial_builder.pore.config import ensure_pore_dirs, pore_root
from pepp_initial_builder.pore.porems_builder import PORE_ELEMENTS, atoms_from_elements, available_pore_rows, read_xyz_like
from pepp_initial_builder.pore.surface_classifier import anchor_indices, cell_failure_reason, normal_metadata, pore_center, rebuild_local_cell, surface_classification


def crop_silica_patches(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_pore_dirs(config)
    rows: List[Dict[str, Any]] = []
    pores = available_pore_rows(config)
    outbase = pore_root(config) / config["paths"]["silica_patches_dir"]
    if pores.empty:
        pd.DataFrame([{"patch_id": "no_available_pore_model", "status": "skipped_no_available_pore_model", "source_pore_model_id": "", "patch_extxyz_path": ""}]).to_csv(outbase / "silica_patch_manifest.csv", index=False)
        return outbase / "silica_patch_manifest.csv"
    patch_cfg = config["silica_patch"]
    max_atoms = int(patch_cfg["max_atoms_silica_patch"])
    min_silica_atoms = int(patch_cfg.get("min_silica_atoms_for_valid_patch", 40))
    radius = float(patch_cfg.get("target_patch_radius_A", 8.0))
    for _, row in pores.iterrows():
        elems, coords, box = read_xyz_like(Path(row["pore_model_extxyz_path"]))
        for patch_idx, patch_type in enumerate(patch_cfg.get("patch_types", ["concave_pore_wall_patch"]), start=1):
            keep = anchor_indices(coords, pore_center(box), str(patch_type), radius, max_atoms)
            patch_elems = [elems[i] for i in keep]
            patch_coords = coords[keep]
            normal_meta = normal_metadata(coords, box, keep)
            local_coords, local_box, cell_meta = rebuild_local_cell(patch_coords, box, config)
            classification = surface_classification(patch_elems, local_coords, config)
            cell_failure = cell_failure_reason(local_box, config)
            silica_atoms = sum(1 for element in patch_elems if element in PORE_ELEMENTS)
            coverage_failure = silica_atoms < min_silica_atoms
            usable = not cell_failure and not coverage_failure
            failure_reason = cell_failure or ("patch_too_small_or_poor_surface_coverage" if coverage_failure else "")
            patch_id = f"patch_{row['pore_model_id']}_{patch_idx:03d}_{patch_type}"
            patch_dir = outbase / patch_id
            patch_dir.mkdir(parents=True, exist_ok=True)
            atoms = atoms_from_elements(patch_elems, local_coords)
            write_xyz(patch_dir / "patch.extxyz", atoms, local_box, ext=True)
            write_pdb(patch_dir / "patch.pdb", atoms, local_box)
            meta = {"patch_id": patch_id, "patch_type": patch_type, "source_pore_model_id": row["pore_model_id"], "status": "available", "source": row["source"], "usable_for_cp2k_aimd": bool(usable), "failure_reason": failure_reason, "geometry_status": "available_geometry" if usable else "available_geometry_not_usable_for_cp2k_aimd", "n_atoms": len(patch_elems), "selection_radius_A": radius, **{k: v for k, v in normal_meta.items() if not k.endswith("_array")}, **cell_meta, **classification}
            (patch_dir / "patch_metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
            rows.append({"patch_id": patch_id, "patch_type": patch_type, "status": "available", "usable_for_cp2k_aimd": bool(usable), "failure_reason": failure_reason, "source_pore_model_id": row["pore_model_id"], "patch_extxyz_path": str(patch_dir / "patch.extxyz"), "n_atoms": len(patch_elems), "selection_radius_A": radius, **{k: v for k, v in normal_meta.items() if not k.endswith("_array")}, **cell_meta, **classification})
    manifest = outbase / "silica_patch_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def patch_rows(config: Dict[str, Any]) -> pd.DataFrame:
    path = pore_root(config) / config["paths"]["silica_patches_dir"] / "silica_patch_manifest.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "usable_for_cp2k_aimd" in df.columns:
        usable = df["usable_for_cp2k_aimd"].astype(str).str.lower().isin({"true", "1", "yes"})
        return df[(df["status"].astype(str) == "available") & usable]
    return df[df["status"].astype(str) == "available"]
