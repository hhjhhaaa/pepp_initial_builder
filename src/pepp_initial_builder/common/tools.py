from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

from pepp_initial_builder.common.paths import ensure_dirs, project_root


def discover_tools(config: Dict[str, Any]) -> Dict[str, Any]:
    tools = config.get("tools", {})
    emc = Path(tools.get("known_emc_root", "/home/jinhao/software/EMC"))
    packmol = Path(tools.get("known_packmol_executable", "/home/jinhao/software/packmol/packmol-21.1.1/packmol"))
    lammps = Path(tools.get("known_lammps_executable", "/home/jinhao/software/lammps/build-cmake/lmp"))
    report = {"python_executable": sys.executable, "python_version": sys.version.split()[0], "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""), "python_modules": {}, "emc": {"root": str(emc) if emc.exists() else None, "executable": str(emc / "bin/emc_linux_x86_64") if (emc / "bin/emc_linux_x86_64").exists() else shutil.which("emc"), "emc_pl": str(emc / "scripts/emc.pl") if (emc / "scripts/emc.pl").exists() else shutil.which("emc.pl"), "emc_setup": str(emc / "scripts/emc_setup.pl") if (emc / "scripts/emc_setup.pl").exists() else shutil.which("emc_setup"), "scripts_dir": str(emc / "scripts") if (emc / "scripts").exists() else None, "examples_dir": str(emc / "examples") if (emc / "examples").exists() else None, "field_dir": str(emc / "field") if (emc / "field").exists() else None, "preferred_fields_found": []}, "packmol": {"executable": str(packmol) if packmol.exists() else shutil.which("packmol")}, "lammps": {"executable": str(lammps) if lammps.exists() else shutil.which("lmp") or shutil.which("lammps")}, "obabel": {"executable": shutil.which("obabel")}}
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
