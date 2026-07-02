from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def project_root(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["root"]).expanduser()


def ensure_dirs(config: Dict[str, Any]) -> None:
    root = project_root(config)
    for key in ["systems_dir", "matrix_dir", "exports_dir", "logs_dir"]:
        if key in config.get("paths", {}):
            (root / config["paths"][key]).mkdir(parents=True, exist_ok=True)
