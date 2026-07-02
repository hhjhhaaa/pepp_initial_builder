from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


def _root(config: Dict[str, Any] | None = None) -> Path:
    if config and "paths" in config and "root" in config["paths"]:
        return Path(config["paths"]["root"])
    return Path(__file__).resolve().parents[3]


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def _display(path: Path) -> str:
    text = str(path).replace("\\", "/")
    prefix = "//wsl.localhost/Ubuntu-22.04"
    return text.removeprefix(prefix)


def export_all_manifests(config: Dict[str, Any] | None = None) -> Tuple[Path, Path]:
    root = _root(config)
    exports = root / "data" / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    paths = config.get("paths", {}) if config else {}
    porems_dir = root / paths.get("porems_models_dir", "data/pore/porems_models")
    patches_dir = root / paths.get("silica_patches_dir", "data/pore/silica_patches")
    mlff_seed_dir = root / paths.get("full_pore_seed_structures_dir", "data/mlff_seed/structures")
    aimd_seed_dir = root / paths.get("aimd_local_structures_dir", "data/cp2k_aimd/seed_structures")
    jobs_dir = root / paths.get("jobs_dir", "outputs/jobs")
    mapping = [
        (root / "data" / "exports" / "mlff_start_manifest.csv", exports / "polymer_initial_manifest.csv"),
        (porems_dir / "pore_model_manifest.csv", exports / "pore_model_manifest.csv"),
        (patches_dir / "silica_patch_manifest.csv", exports / "silica_patch_manifest.csv"),
        (mlff_seed_dir / "full_pore_seed_manifest.csv", exports / "mlff_seed_manifest.csv"),
        (aimd_seed_dir / "aimd_local_manifest.csv", exports / "aimd_seed_manifest.csv"),
        (jobs_dir / "cp2k_hpc_job_manifest.csv", exports / "cp2k_job_manifest.csv"),
    ]
    rows = []
    for src, dst in mapping:
        copied = _copy_if_exists(src, dst)
        rows.append({"manifest": dst.name, "source": _display(src), "path": _display(dst), "status": "available" if copied else "missing_source"})
    combined = exports / "all_data_generation_manifest.csv"
    pd.DataFrame(rows).to_csv(combined, index=False)
    summary = exports / "all_data_generation_manifest.json"
    summary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return combined, summary
