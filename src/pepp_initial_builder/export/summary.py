from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def summarize_outputs(config: Dict[str, Any] | None = None) -> Path:
    root = Path(config["paths"]["root"]).expanduser() if config and "paths" in config else Path(__file__).resolve().parents[3]
    paths = config.get("paths", {}) if config else {}
    exports_dir = _resolve(root, paths.get("exports_dir", "data/exports"))
    logs_dir = _resolve(root, paths.get("logs_dir", "outputs/logs"))
    figures_dir = _resolve(root, paths.get("figures_dir", "outputs/figures"))
    jobs_dir = _resolve(root, paths.get("jobs_dir", "outputs/jobs"))
    exports = sorted(exports_dir.glob("*manifest*")) if exports_dir.exists() else []
    reports = sorted(logs_dir.glob("*report*")) if logs_dir.exists() else []
    validations = sorted(logs_dir.glob("*validation*")) if logs_dir.exists() else []
    figures = sorted(figures_dir.glob("*")) if figures_dir.exists() else []
    jobs = sorted(jobs_dir.glob("*")) if jobs_dir.exists() else []
    out = logs_dir / "data_generation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "exports": [str(path) for path in exports],
                "reports": [str(path) for path in reports],
                "validations": [str(path) for path in validations],
                "figures": [str(path) for path in figures],
                "jobs": [str(path) for path in jobs],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out
