from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from pepp_initial_builder.common.run import apply_run_namespace


def load_yaml(path: str | Path) -> Dict[str, Any]:
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


load_config = load_yaml
