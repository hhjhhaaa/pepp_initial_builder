from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from pepp_initial_builder.common.paths import ensure_dirs, project_root


def configured_path(value: object) -> Path | None:
    if value in {None, ""}:
        return None
    return Path(str(value)).expanduser()


def existing_executable(path: Path | None) -> str | None:
    return str(path) if path and path.exists() else None


def discover_tools(config: Dict[str, Any]) -> Dict[str, Any]:
    tools = config.get("tools", {})
    emc = configured_path(tools.get("known_emc_root"))
    packmol = configured_path(tools.get("known_packmol_executable"))
    lammps = configured_path(tools.get("known_lammps_executable"))
    report = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_modules": {},
        "emc": {
            "root": str(emc) if emc and emc.exists() else None,
            "executable": existing_executable(emc / "bin/emc_linux_x86_64" if emc else None) or shutil.which("emc"),
            "emc_pl": existing_executable(emc / "scripts/emc.pl" if emc else None) or shutil.which("emc.pl"),
            "emc_setup": existing_executable(emc / "scripts/emc_setup.pl" if emc else None) or shutil.which("emc_setup"),
            "scripts_dir": str(emc / "scripts") if emc and (emc / "scripts").exists() else None,
            "examples_dir": str(emc / "examples") if emc and (emc / "examples").exists() else None,
            "field_dir": str(emc / "field") if emc and (emc / "field").exists() else None,
            "preferred_fields_found": [],
        },
        "packmol": {"executable": existing_executable(packmol) or shutil.which("packmol")},
        "lammps": {"executable": existing_executable(lammps) or shutil.which("lmp") or shutil.which("lammps")},
        "obabel": {"executable": shutil.which("obabel")},
    }
    for module in ["numpy", "pandas", "yaml", "ase", "MDAnalysis", "rdkit", "openbabel"]:
        try:
            __import__(module)
            report["python_modules"][module] = "FOUND"
        except Exception as exc:
            report["python_modules"][module] = f"MISSING: {exc}"
    field_dir = report["emc"]["field_dir"]
    if field_dir:
        field = Path(field_dir)
        preferred = {"OPLS-AA": ["opls/2024/opls-aa.prm", "opls/2012/opls-aa.prm"], "PCFF": ["pcff/pcff.frc"], "TraPPE": ["trappe/2014/trappe-ua.prm"]}
        for name, rels in preferred.items():
            if any((field / rel).exists() for rel in rels):
                report["emc"]["preferred_fields_found"].append(name)
    return report


def write_discovery_report(config: Dict[str, Any], extra: Dict[str, Any] | None = None) -> Path:
    ensure_dirs(config)
    report = discover_tools(config)
    if extra:
        report.update(extra)
    path = project_root(config) / config["paths"]["logs_dir"] / "emc_discovery_report.txt"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path
