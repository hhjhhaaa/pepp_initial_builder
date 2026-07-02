from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepp_initial_builder.common.config import load_config
from pepp_initial_builder.cp2k_aimd.config import load_cp2k_config


def _module_status(name: str) -> str:
    try:
        importlib.import_module(name)
        return "FOUND"
    except Exception as exc:
        return f"MISSING: {exc}"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    if path.name == "cp2k_aimd.yaml":
        return load_cp2k_config(path)
    return load_config(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polymer-config", default="configs/polymer.yaml")
    parser.add_argument("--cp2k-config", default="configs/cp2k_aimd.yaml")
    args = parser.parse_args()

    polymer_config = _load_yaml(ROOT / args.polymer_config)
    cp2k_config = _load_yaml(ROOT / args.cp2k_config)
    tools = polymer_config.get("tools", {})
    cp2k_paths = cp2k_config.get("paths", {})
    hpc = cp2k_config.get("hpc", {})

    report = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "package_import_pepp_initial_builder": _module_status("pepp_initial_builder"),
        "python_modules": {
            "numpy": _module_status("numpy"),
            "pandas": _module_status("pandas"),
            "yaml": _module_status("yaml"),
            "ase": _module_status("ase"),
            "openbabel": _module_status("openbabel"),
        },
        "emc": {
            "configured_root": tools.get("known_emc_root"),
            "root_exists": Path(str(tools.get("known_emc_root", ""))).expanduser().exists(),
            "executable": str(Path(str(tools.get("known_emc_root", ""))).expanduser() / "bin/emc_linux_x86_64") if (Path(str(tools.get("known_emc_root", ""))).expanduser() / "bin/emc_linux_x86_64").exists() else shutil.which("emc"),
            "emc_pl": str(Path(str(tools.get("known_emc_root", ""))).expanduser() / "scripts/emc.pl") if (Path(str(tools.get("known_emc_root", ""))).expanduser() / "scripts/emc.pl").exists() else shutil.which("emc.pl"),
            "emc_setup_pl": str(Path(str(tools.get("known_emc_root", ""))).expanduser() / "scripts/emc_setup.pl") if (Path(str(tools.get("known_emc_root", ""))).expanduser() / "scripts/emc_setup.pl").exists() else shutil.which("emc_setup.pl") or shutil.which("emc_setup"),
        },
        "packmol": {
            "configured": tools.get("known_packmol_executable"),
            "found": shutil.which("packmol") or (tools.get("known_packmol_executable") if Path(str(tools.get("known_packmol_executable", ""))).exists() else None),
        },
        "openbabel": {
            "configured": tools.get("known_openbabel_executable"),
            "found": shutil.which("obabel") or (tools.get("known_openbabel_executable") if Path(str(tools.get("known_openbabel_executable", ""))).exists() else None),
        },
        "lammps": {
            "configured": tools.get("known_lammps_executable"),
            "found": shutil.which("lmp") or shutil.which("lammps") or (tools.get("known_lammps_executable") if Path(str(tools.get("known_lammps_executable", ""))).exists() else None),
        },
        "porems_external_python": {
            "hint": cp2k_config.get("tools", {}).get("porems_path_hint"),
            "python_import": _module_status("porems"),
        },
        "cp2k_local_command_optional": {
            "cp2k.psmp": shutil.which("cp2k.psmp"),
            "cp2k": shutil.which("cp2k"),
            "required_locally": False,
        },
        "hpc_cp2k_module_placeholder": hpc.get("cp2k_module_placeholder", "__SET_CP2K_MODULE_ON_HPC__"),
        "lmp_proj": {
            "path": str(Path(cp2k_paths.get("lmp_proj_root", "~/lmp-proj")).expanduser()),
            "exists": Path(cp2k_paths.get("lmp_proj_root", "~/lmp-proj")).expanduser().exists(),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
