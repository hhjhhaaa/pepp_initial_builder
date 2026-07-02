from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def summarize_outputs(config: Dict[str, Any] | None = None) -> Path:
    root = Path(config["paths"]["root"]).expanduser() if config and "paths" in config else Path(__file__).resolve().parents[3]
    exports = sorted((root / "data" / "exports").glob("*manifest*"))
    reports = sorted((root / "outputs" / "logs").glob("*report*"))
    validations = sorted((root / "outputs" / "logs").glob("*validation*"))
    out = root / "outputs" / "reports" / "data_generation_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "exports": [str(path) for path in exports],
                "reports": [str(path) for path in reports],
                "validations": [str(path) for path in validations],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out
