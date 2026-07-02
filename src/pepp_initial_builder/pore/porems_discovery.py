from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pepp_initial_builder.pore.config import ensure_pore_dirs, pore_root


def safe_find(root: Path, patterns: Sequence[str], max_results: int = 200) -> List[str]:
    if not root.exists():
        return []
    cmd = ["find", str(root), "-maxdepth", "9", "("]
    for i, pattern in enumerate(patterns):
        if i:
            cmd.append("-o")
        cmd.extend(["-iname", pattern])
    cmd.extend([")"])
    try:
        out = subprocess.run(cmd, text=True, capture_output=True, timeout=30, check=False)
    except Exception:
        return []
    return [line for line in out.stdout.splitlines()[:max_results] if line]


def python_for_site_package(path: Path) -> Optional[str]:
    parts = path.parts
    if "site-packages" not in parts:
        return None
    idx = parts.index("site-packages")
    lib = Path(*parts[: idx - 1]) if idx >= 1 else None
    if lib is None:
        return None
    env_root = lib.parent
    py = env_root / "bin" / "python"
    return str(py) if py.exists() else None


def scan_porems_python_packages(search_roots: Sequence[Path]) -> List[Dict[str, Any]]:
    packages: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for root in search_roots:
        for hit in safe_find(root, ["porems-*.dist-info", "porems"], max_results=500):
            path = Path(hit)
            if path.name == "porems" and not path.is_dir():
                continue
            if "site-packages" not in path.parts:
                continue
            site_idx = path.parts.index("site-packages")
            site = Path(*path.parts[: site_idx + 1])
            pkg = site / "porems"
            if not pkg.exists():
                continue
            key = str(pkg)
            if key in seen:
                continue
            seen.add(key)
            version = None
            for dist in site.glob("porems-*.dist-info"):
                version = dist.name.replace("porems-", "").replace(".dist-info", "")
                break
            packages.append({"package_path": str(pkg), "site_packages": str(site), "python_executable": python_for_site_package(pkg), "version": version})
    return packages


def discover_porems(config: Dict[str, Any]) -> Dict[str, Any]:
    ensure_pore_dirs(config)
    if config.get("porems", {}).get("enabled") is False:
        return {"available": False, "source": "disabled_by_config", "executable": None, "python_package_found": False, "python_package_error": "disabled_by_config", "version": None, "python_executable": None, "discovered_python_packages": [], "candidates": [], "examples": [], "templates": [], "manual_user_input_allowed": True, "manual_user_input_note": "Place extxyz/pdb pore models under data/pore/porems_models/manual_* and rerun builders."}
    tools = config["tools"]
    hint_value = tools.get("porems_path_hint")
    hint = Path(str(hint_value)).expanduser() if hint_value else None
    roots = [Path(str(x)).expanduser() for x in tools.get("porems_search_roots", [])]
    candidates: List[str] = []
    if hint and hint.exists():
        candidates.append(str(hint))
    for exe in ["porems", "PoreMS", "porems.py"]:
        found = shutil.which(exe)
        if found:
            candidates.append(found)
    for root in roots:
        candidates.extend(safe_find(root, ["*porems*", "*PoreMS*"]))
    unique = sorted(dict.fromkeys(candidates))
    discovered_packages = scan_porems_python_packages(roots)
    package_found = False
    package_error = None
    try:
        __import__("porems")
        package_found = True
    except Exception as exc:
        package_error = str(exc)
    executable = next((path for path in unique if os.access(path, os.X_OK) and Path(path).is_file()), None)
    version = None
    if executable:
        try:
            out = subprocess.run([executable, "--version"], text=True, capture_output=True, timeout=15, check=False)
            version = (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr).strip() else None
        except Exception as exc:
            version = f"version_probe_failed: {exc}"
    external_python = next((pkg["python_executable"] for pkg in discovered_packages if pkg.get("python_executable")), None)
    external_version = next((pkg["version"] for pkg in discovered_packages if pkg.get("version")), None)
    return {"available": bool(executable or package_found or discovered_packages), "source": "installed_porems" if (executable or package_found or discovered_packages) else "not_available", "executable": executable, "python_package_found": package_found, "python_package_error": package_error, "version": version or external_version, "python_executable": sys.executable if package_found else external_python, "discovered_python_packages": discovered_packages, "candidates": unique, "examples": [path for path in unique if "example" in path.lower()], "templates": [path for path in unique if "template" in path.lower()], "manual_user_input_allowed": True, "manual_user_input_note": "Place extxyz/pdb pore models under data/pore/porems_models/manual_* and rerun builders."}


def write_porems_discovery(config: Dict[str, Any]) -> Tuple[Path, Path]:
    report = discover_porems(config)
    log_dir = pore_root(config) / config["paths"]["logs_dir"]
    txt = log_dir / "porems_discovery_report.txt"
    js = log_dir / "porems_discovery.json"
    txt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    js.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return txt, js
