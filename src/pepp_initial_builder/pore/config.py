from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_pore_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def pore_root(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["root"])


def ensure_pore_dirs(config: Dict[str, Any]) -> None:
    root = pore_root(config)
    for key in ["porems_models_dir", "silica_patches_dir", "aimd_local_structures_dir", "full_pore_seed_structures_dir", "aimd_exports_dir", "logs_dir", "figures_dir"]:
        path = config["paths"].get(key)
        if path:
            (root / path).mkdir(parents=True, exist_ok=True)
