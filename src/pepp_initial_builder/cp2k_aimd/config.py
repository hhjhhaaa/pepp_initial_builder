from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import yaml

from pepp_initial_builder.common.run import apply_run_namespace


def load_cp2k_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if isinstance(config, dict) and isinstance(config.get("paths"), dict):
        config = apply_run_namespace(config)
        root_value = config["paths"].get("root")
        if root_value in {None, "", "."}:
            config["paths"]["root"] = str(path.resolve().parents[1])
        else:
            config["paths"]["root"] = str(Path(str(root_value)).expanduser())
        if config["paths"].get("lmp_proj_root"):
            config["paths"]["lmp_proj_root"] = str(Path(str(config["paths"]["lmp_proj_root"])).expanduser())
    return config


def root(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["root"]).expanduser()


def p(config: Dict[str, Any], key: str) -> Path:
    value = Path(config["paths"][key])
    return value if value.is_absolute() else root(config) / value


def ensure_dirs(config: Dict[str, Any]) -> None:
    for key in ["cp2k_jobs_dir", "cp2k_parsed_dir", "aimd_dataset_dir", "exports_dir", "logs_dir", "jobs_dir"]:
        p(config, key).mkdir(parents=True, exist_ok=True)


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (value.replace("\\", "/") if isinstance(value, str) else value) for key, value in row.items()})
    return path
