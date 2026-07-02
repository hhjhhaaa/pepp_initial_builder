from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from pepp_initial_builder.pore.config import ensure_pore_dirs, pore_root


def export_pore_aimd_manifests(config: Dict[str, Any]) -> Tuple[Path, Path]:
    ensure_pore_dirs(config)
    root = pore_root(config)
    outdir = root / config["paths"]["aimd_exports_dir"]
    pieces = []
    for rel, kind in [
        (f"{config['paths']['porems_models_dir']}/pore_model_manifest.csv", "pore_model"),
        (f"{config['paths']['silica_patches_dir']}/silica_patch_manifest.csv", "silica_patch"),
        (f"{config['paths']['aimd_local_structures_dir']}/aimd_local_manifest.csv", "aimd_local"),
        (f"{config['paths']['full_pore_seed_structures_dir']}/full_pore_seed_manifest.csv", "full_pore_seed"),
        (f"{config['paths']['aimd_exports_dir']}/cp2k_structure_input_manifest.csv", "cp2k_structure_input"),
    ]:
        path = root / rel
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "manifest_kind", kind)
            pieces.append(df)
    combined = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame([{"manifest_kind": "none", "status": "no_pore_aimd_outputs_available"}])
    csv_path = outdir / "pore_aimd_master_manifest.csv"
    json_path = outdir / "pore_aimd_master_manifest.json"
    combined.to_csv(csv_path, index=False)
    json_path.write_text(combined.to_json(orient="records", indent=2), encoding="utf-8")
    return csv_path, json_path
