from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from pepp_initial_builder.pore.config import ensure_pore_dirs, pore_root
from pepp_initial_builder.pore.porems_builder import AIMD_ELEMENTS, PORE_ELEMENTS, read_xyz_like, validate_pore_elements


def validate_extxyz(path: Path, allowed: set[str], require_h: bool = False) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "valid": False, "reason": "missing"}
    elems, _coords, box = read_xyz_like(path)
    return {"exists": True, "valid": set(elems).issubset(allowed) and (not require_h or "H" in elems) and all(x > 0 for x in box), "n_atoms": len(elems), "elements": ";".join(sorted(set(elems))), "has_explicit_h": "H" in elems, "box_lx_A": box[0], "box_ly_A": box[1], "box_lz_A": box[2], "pbc": True}


def validate_from_manifest(config: Dict[str, Any], manifest_rel: str, id_col: str, path_col: str, out_name: str, allowed: set[str], require_h: bool = False) -> Path:
    ensure_pore_dirs(config)
    manifest = pore_root(config) / manifest_rel
    rows = []
    if manifest.exists():
        df = pd.read_csv(manifest)
        for _, row in df.iterrows():
            rec = {id_col: row.get(id_col, ""), "status": row.get("status", "")}
            path = str(row.get(path_col, "") or "")
            rec.update(validate_extxyz(Path(path), allowed, require_h) if path else {"exists": False, "valid": False, "reason": "no_structure_path"})
            rows.append(rec)
    if not rows:
        rows.append({id_col: "no_manifest_rows", "status": "skipped", "exists": False, "valid": False})
    out = pore_root(config) / config["paths"]["logs_dir"] / out_name
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def validate_pore_structures(config: Dict[str, Any]) -> Path:
    manifest = f"{config['paths']['porems_models_dir']}/pore_model_manifest.csv"
    return validate_from_manifest(config, manifest, "pore_model_id", "pore_model_extxyz_path", "pore_structure_validation.csv", PORE_ELEMENTS, True)


def validate_aimd_local_structures(config: Dict[str, Any]) -> Path:
    manifest = f"{config['paths']['aimd_local_structures_dir']}/aimd_local_manifest.csv"
    return validate_from_manifest(config, manifest, "aimd_structure_id", "extxyz_path", "aimd_local_validation.csv", AIMD_ELEMENTS, True)
