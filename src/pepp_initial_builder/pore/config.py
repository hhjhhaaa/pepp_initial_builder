from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from pepp_initial_builder.common.run import apply_run_namespace


def load_pore_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if isinstance(config, dict) and isinstance(config.get("paths"), dict):
        config = apply_run_namespace(config)
        root = config["paths"].get("root")
        if root in {None, "", "."}:
            config["paths"]["root"] = str(path.resolve().parents[1])
        else:
            config["paths"]["root"] = str(Path(str(root)).expanduser())
    return config


def pore_root(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["root"]).expanduser()


def ensure_pore_dirs(config: Dict[str, Any]) -> None:
    root = pore_root(config)
    for key in ["porems_models_dir", "silica_patches_dir", "surface_sites_dir", "aimd_local_structures_dir", "full_pore_seed_structures_dir", "aimd_exports_dir", "logs_dir", "figures_dir"]:
        path = config["paths"].get(key)
        if path:
            (root / path).mkdir(parents=True, exist_ok=True)
