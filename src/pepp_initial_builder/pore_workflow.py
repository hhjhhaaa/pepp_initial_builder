from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from .core import Atom, SystemTopology, build_python_topology, matrix_rows, write_pdb, write_xyz

PORE_ELEMENTS = {"Si", "O", "H"}
AIMD_ELEMENTS = {"C", "H", "O", "Si"}


def load_pore_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pore_root(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["root"])


def ensure_pore_dirs(config: Dict[str, Any]) -> None:
    root = pore_root(config)
    for key in [
        "porems_models_dir",
        "silica_patches_dir",
        "aimd_local_structures_dir",
        "full_pore_seed_structures_dir",
        "aimd_exports_dir",
        "logs_dir",
        "figures_dir",
    ]:
        path = config["paths"].get(key)
        if path:
            (root / path).mkdir(parents=True, exist_ok=True)


def _safe_find(root: Path, patterns: Sequence[str], max_results: int = 200) -> List[str]:
    if not root.exists():
        return []
    cmd = ["find", str(root), "-maxdepth", "9", "("]
    for i, pat in enumerate(patterns):
        if i:
            cmd.append("-o")
        cmd.extend(["-iname", pat])
    cmd.extend([")"])
    try:
        out = subprocess.run(cmd, text=True, capture_output=True, timeout=30, check=False)
    except Exception:
        return []
    return [x for x in out.stdout.splitlines()[:max_results] if x]


def _python_for_site_package(path: Path) -> Optional[str]:
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


def _scan_porems_python_packages(search_roots: Sequence[Path]) -> List[Dict[str, Any]]:
    packages: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for root in search_roots:
        for hit in _safe_find(root, ["porems-*.dist-info", "porems"], max_results=500):
            p = Path(hit)
            if p.name == "porems" and not p.is_dir():
                continue
            if "site-packages" not in p.parts:
                continue
            site_idx = p.parts.index("site-packages")
            site = Path(*p.parts[: site_idx + 1])
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
            packages.append(
                {
                    "package_path": str(pkg),
                    "site_packages": str(site),
                    "python_executable": _python_for_site_package(pkg),
                    "version": version,
                }
            )
    return packages


def discover_porems(config: Dict[str, Any]) -> Dict[str, Any]:
    ensure_pore_dirs(config)
    tools = config["tools"]
    hint = Path(tools.get("porems_path_hint", ""))
    roots = [Path(x) for x in tools.get("porems_search_roots", [])]
    candidates: List[str] = []
    if hint.exists():
        candidates.append(str(hint))
    for exe in ["porems", "PoreMS", "porems.py"]:
        found = shutil.which(exe)
        if found:
            candidates.append(found)
    for root in roots:
        candidates.extend(_safe_find(root, ["*porems*", "*PoreMS*"]))
    unique = sorted(dict.fromkeys(candidates))
    discovered_packages = _scan_porems_python_packages(roots)
    package_found = False
    package_error = None
    try:
        __import__("porems")
        package_found = True
    except Exception as exc:
        package_error = str(exc)
    examples = [p for p in unique if "example" in p.lower()]
    executable = next((p for p in unique if os.access(p, os.X_OK) and Path(p).is_file()), None)
    version = None
    if executable:
        try:
            out = subprocess.run([executable, "--version"], text=True, capture_output=True, timeout=15, check=False)
            version = (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr).strip() else None
        except Exception as exc:
            version = f"version_probe_failed: {exc}"
    external_python = next((p["python_executable"] for p in discovered_packages if p.get("python_executable")), None)
    external_version = next((p["version"] for p in discovered_packages if p.get("version")), None)
    return {
        "available": bool(executable or package_found or discovered_packages),
        "source": "installed_porems" if (executable or package_found or discovered_packages) else "not_available",
        "executable": executable,
        "python_package_found": package_found,
        "python_package_error": package_error,
        "version": version or external_version,
        "python_executable": sys.executable if package_found else external_python,
        "discovered_python_packages": discovered_packages,
        "candidates": unique,
        "examples": examples,
        "templates": [p for p in unique if "template" in p.lower()],
        "manual_user_input_allowed": True,
        "manual_user_input_note": "Place extxyz/pdb pore models under data/porems_models/manual_* and rerun builders.",
    }


def write_porems_discovery(config: Dict[str, Any]) -> Tuple[Path, Path]:
    report = discover_porems(config)
    log_dir = pore_root(config) / config["paths"]["logs_dir"]
    txt = log_dir / "porems_discovery_report.txt"
    js = log_dir / "porems_discovery.json"
    txt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    js.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return txt, js


def _read_xyz_like(path: str | Path) -> Tuple[List[str], np.ndarray, Tuple[float, float, float]]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = int(lines[0].strip())
    comment = lines[1] if len(lines) > 1 else ""
    box = (30.0, 30.0, 30.0)
    if "Lattice=" in comment:
        part = comment.split('Lattice="', 1)[1].split('"', 1)[0].split()
        if len(part) >= 9:
            box = (float(part[0]), float(part[4]), float(part[8]))
    elems, coords = [], []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        elems.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elems, np.array(coords, dtype=float), box


def _atoms_from_elements(elems: Sequence[str], coords: np.ndarray, chain_id: int = 0) -> List[Atom]:
    atoms = []
    for i, (el, xyz) in enumerate(zip(elems, coords), start=1):
        atoms.append(
            Atom(
                i,
                el,
                el,
                "silica" if el in PORE_ELEMENTS else "polymer",
                chain_id,
                "silica" if el in PORE_ELEMENTS else "polymer",
                0,
                False,
                False,
                False,
                el == "H",
                0,
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
                chain_id,
                0.0,
            )
        )
    return atoms


def validate_pore_elements(elems: Sequence[str]) -> Dict[str, Any]:
    counts = {e: elems.count(e) for e in sorted(set(elems))}
    return {
        "elements": sorted(set(elems)),
        "element_counts": counts,
        "only_si_o_h": set(elems).issubset(PORE_ELEMENTS),
        "explicit_surface_h_present": counts.get("H", 0) > 0,
        "obvious_dangling_check": "not_evaluated_without_porems_bond_order",
    }


def _map_porems_element(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if s.startswith("SI") or s in {"SIL", "S"}:
        return "Si"
    if s.startswith("H"):
        return "H"
    if s.startswith("O") or s in {"OM", "OX", "OH", "OS", "SLX", "SL", "SLG"}:
        return "O"
    return str(symbol).strip()


def _run_porems_external(
    python_executable: str,
    model_dir: Path,
    *,
    pore_diameter_nm: float,
    pore_length_nm: float,
    hydroxylation_mode: str,
) -> Tuple[List[str], np.ndarray, Tuple[float, float, float], str]:
    raw_dir = model_dir / "porems_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / "porems_atoms.json"
    script_path = raw_dir / "generate_porems_model.py"
    box_xy = float(pore_diameter_nm) + 2.0
    hydro = [0, 0]
    if hydroxylation_mode == "silanol_rich":
        hydro = [0, 0]
    elif hydroxylation_mode == "siloxane_rich":
        hydro = [0, 0]
    script_path.write_text(
        f"""
import json
import porems as pms

pore = pms.PoreCylinder([{box_xy!r}, {box_xy!r}, {float(pore_length_nm)!r}], {float(pore_diameter_nm)!r}, 0, hydro={hydro!r})
pore.finalize()
mol_dict = pore._pore.get_mol_dict()
atoms = []
for key, mols in mol_dict.items():
    for mol in mols:
        for atom in mol.get_atom_list():
            atoms.append({{"type": atom.get_atom_type(), "name": getattr(atom, "get_name", lambda: "")(), "pos_nm": atom.get_pos()}})
data = {{
    "box_nm": [float(pore._box[0]), float(pore._box[1]), float(pore._box[2])],
    "centroid_nm": [float(x) for x in getattr(pore, "_centroid", [0, 0, 0])],
    "diameter_nm": float(pore.diameter()[0]) if pore.diameter() else {float(pore_diameter_nm)!r},
    "atoms": atoms,
}}
with open({str(json_path)!r}, "w", encoding="utf-8") as f:
    json.dump(data, f)
""".lstrip(),
        encoding="utf-8",
    )
    log_path = model_dir / "porems_build.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run([python_executable, str(script_path)], text=True, stdout=log, stderr=subprocess.STDOUT, cwd=raw_dir, timeout=300, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"PoreMS external builder failed with exit code {proc.returncode}; see {log_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    elems = [_map_porems_element(a["type"]) for a in data["atoms"]]
    coords = np.array([[10.0 * float(x) for x in a["pos_nm"]] for a in data["atoms"]], dtype=float)
    box = tuple(10.0 * float(x) for x in data["box_nm"])
    return elems, coords, box, str(log_path)


def _manual_pore_models(config: Dict[str, Any]) -> List[Path]:
    base = pore_root(config) / config["paths"]["porems_models_dir"]
    paths: List[Path] = []
    for folder in sorted(base.glob("manual_*")):
        for name in ["pore_model.extxyz", "pore_model.xyz"]:
            p = folder / name
            if p.exists():
                paths.append(p)
                break
    return paths


def build_porems_pores(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_pore_dirs(config)
    discovery = discover_porems(config)
    outdir = pore_root(config) / config["paths"]["porems_models_dir"]
    rows: List[Dict[str, Any]] = []
    manual = _manual_pore_models(config)
    if not discovery["available"] and not manual:
        rows.append(
            {
                "pore_model_id": "porems_unavailable",
                "status": "failed_porems_not_available",
                "source": "none",
                "pore_model_extxyz_path": "",
                "pore_model_pdb_path": "",
                "pore_diameter_nm": "",
                "pore_length_nm": "",
                "hydroxylation_mode": "",
            }
        )
    if discovery["available"] and not manual:
        max_models = int(config["porems"].get(f"max_pore_models_{mode}", 1))
        made = 0
        py = discovery.get("python_executable")
        for diameter in config["porems"]["pore_diameter_nm_candidates"]:
            for length in config["porems"]["pore_length_nm_candidates"]:
                for hydro in config["porems"]["hydroxylation_modes"]:
                    if made >= max_models:
                        break
                    pore_model_id = f"porems_D{float(diameter):.1f}nm_L{float(length):.1f}nm_{hydro}".replace(".", "p")
                    model_dir = outdir / pore_model_id
                    model_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        if not py:
                            raise RuntimeError("PoreMS available but no Python executable was discovered")
                        elems, coords, box, log_path = _run_porems_external(
                            py,
                            model_dir,
                            pore_diameter_nm=float(diameter),
                            pore_length_nm=float(length),
                            hydroxylation_mode=str(hydro),
                        )
                        atoms = _atoms_from_elements(elems, coords)
                        write_xyz(model_dir / "pore_model.extxyz", atoms, box, ext=True)
                        write_pdb(model_dir / "pore_model.pdb", atoms, box)
                        validation = validate_pore_elements(elems)
                        status = "available" if validation["only_si_o_h"] and validation["explicit_surface_h_present"] else "failed_validation"
                        meta = {
                            "pore_model_id": pore_model_id,
                            "source": "porems_python_package",
                            "porems_python_executable": py,
                            "porems_version": discovery.get("version"),
                            "status": status,
                            "pore_diameter_nm": float(diameter),
                            "pore_length_nm": float(length),
                            "surface": config["porems"]["surface"],
                            "hydroxylation_mode": hydro,
                            "validation": validation,
                            "porems_build_log": log_path,
                        }
                        (model_dir / "pore_metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
                        rows.append(
                            {
                                "pore_model_id": pore_model_id,
                                "status": status,
                                "source": "porems_python_package",
                                "pore_model_extxyz_path": str(model_dir / "pore_model.extxyz"),
                                "pore_model_pdb_path": str(model_dir / "pore_model.pdb"),
                                "pore_diameter_nm": float(diameter),
                                "pore_length_nm": float(length),
                                "hydroxylation_mode": hydro,
                            }
                        )
                    except Exception as exc:
                        (model_dir / "porems_build.log").write_text(str(exc) + "\n", encoding="utf-8")
                        rows.append(
                            {
                                "pore_model_id": pore_model_id,
                                "status": "failed_porems_build_failed",
                                "source": "porems_python_package",
                                "pore_model_extxyz_path": "",
                                "pore_model_pdb_path": "",
                                "pore_diameter_nm": float(diameter),
                                "pore_length_nm": float(length),
                                "hydroxylation_mode": hydro,
                            }
                        )
                    made += 1
                if made >= max_models:
                    break
            if made >= max_models:
                break
    for source_path in manual:
        model_dir = source_path.parent
        elems, coords, box = _read_xyz_like(source_path)
        atoms = _atoms_from_elements(elems, coords)
        write_xyz(model_dir / "pore_model.extxyz", atoms, box, ext=True)
        write_pdb(model_dir / "pore_model.pdb", atoms, box)
        meta = {
            "pore_model_id": model_dir.name,
            "source": "manual_user_input",
            "status": "available_manual_user_input",
            "validation": validate_pore_elements(elems),
            "pore_shape": config["porems"]["pore_shape"],
            "surface": config["porems"]["surface"],
        }
        (model_dir / "pore_metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        rows.append(
            {
                "pore_model_id": model_dir.name,
                "status": meta["status"],
                "source": "manual_user_input",
                "pore_model_extxyz_path": str(model_dir / "pore_model.extxyz"),
                "pore_model_pdb_path": str(model_dir / "pore_model.pdb"),
                "pore_diameter_nm": "",
                "pore_length_nm": "",
                "hydroxylation_mode": "",
            }
        )
    manifest = outdir / "pore_model_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _available_pore_rows(config: Dict[str, Any]) -> pd.DataFrame:
    path = pore_root(config) / config["paths"]["porems_models_dir"] / "pore_model_manifest.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["status"].astype(str).str.startswith("available")]


def _toy_patch() -> Tuple[List[str], np.ndarray, Tuple[float, float, float]]:
    elems = ["Si", "O", "O", "O", "O", "H", "H", "H", "H"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [-1.6, 0.0, 0.0],
            [0.0, 1.6, 0.0],
            [0.0, -1.6, 0.0],
            [2.2, 0.0, 0.0],
            [-2.2, 0.0, 0.0],
            [0.0, 2.2, 0.0],
            [0.0, -2.2, 0.0],
        ],
        dtype=float,
    )
    return elems, coords, (24.0, 24.0, 24.0)


def _mlff_cell_min(config: Dict[str, Any]) -> float:
    assumptions = config.get("mlff_assumptions", {})
    cutoff = float(assumptions.get("planned_mlff_cutoff_A", 5.0))
    multiple = float(assumptions.get("min_cell_multiple_of_cutoff", 2.0))
    absolute = float(assumptions.get("min_cell_length_A", 12.0))
    return max(absolute, multiple * cutoff)


def _vec3_text(values: Sequence[float]) -> str:
    return " ".join(f"{float(v):.8f}" for v in values)


def _pore_center(box: Tuple[float, float, float]) -> np.ndarray:
    return np.array([box[0] / 2.0, box[1] / 2.0, box[2] / 2.0], dtype=float)


def _anchor_indices(coords: np.ndarray, center: np.ndarray, patch_type: str, radius: float, max_atoms: int) -> np.ndarray:
    xy = coords[:, :2] - center[:2]
    radial = np.linalg.norm(xy, axis=1)
    order = np.argsort(radial)
    anchor = order[-1] if "concave" in patch_type or "wall" in patch_type or len(order) else 0
    if "flat" in patch_type and len(order):
        anchor = order[len(order) // 2]
    distances = np.linalg.norm(coords - coords[anchor], axis=1)
    keep = np.where(distances <= radius)[0]
    if len(keep) < min(max_atoms, len(coords)):
        keep = np.argsort(distances)[: min(max_atoms, len(coords))]
    return np.array(sorted(keep[: min(max_atoms, len(keep))]), dtype=int)


def _normal_metadata(coords: np.ndarray, box: Tuple[float, float, float], keep: np.ndarray) -> Dict[str, Any]:
    center = _pore_center(box)
    patch_center = coords[keep].mean(axis=0) if len(keep) else center
    radial = np.array([patch_center[0] - center[0], patch_center[1] - center[1], 0.0], dtype=float)
    norm = float(np.linalg.norm(radial))
    if norm > 1.0e-8:
        outward = radial / norm
        status = "box_center_radial"
    else:
        outward = np.array([1.0, 0.0, 0.0], dtype=float)
        radial = outward.copy()
        status = "weak_default_box_center"
    inward = -outward
    return {
        "pore_axis": "z",
        "pore_center_xyz": _vec3_text(center),
        "radial_vector_xyz": _vec3_text(radial),
        "inward_normal_xyz": _vec3_text(inward),
        "normal_definition_status": status,
        "patch_center_xyz": _vec3_text(patch_center),
        "patch_center_array": patch_center,
        "inward_normal_array": inward,
    }


def _surface_classification(elems: Sequence[str], coords: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    o_idx = [i for i, el in enumerate(elems) if el == "O"]
    h_idx = [i for i, el in enumerate(elems) if el == "H"]
    si_idx = [i for i, el in enumerate(elems) if el == "Si"]
    n_silanol = 0
    n_siloxane = 0
    for oi in o_idx:
        oh = any(float(np.linalg.norm(coords[oi] - coords[hi])) <= 1.25 for hi in h_idx)
        si_neighbors = sum(1 for si in si_idx if float(np.linalg.norm(coords[oi] - coords[si])) <= 2.1)
        if oh:
            n_silanol += 1
        if not oh and si_neighbors >= 2:
            n_siloxane += 1
    n_o = len(o_idx)
    min_oh = int(config.get("silica_patch", {}).get("min_surface_oh_count_for_silanol_rich", 2))
    oh_ratio = n_silanol / n_o if n_o else 0.0
    silox_ratio = n_siloxane / n_o if n_o else 0.0
    if n_silanol >= min_oh and n_siloxane >= min_oh:
        surface_class = "mixed_oh"
    elif n_silanol >= min_oh and oh_ratio >= silox_ratio:
        surface_class = "silanol_rich"
    elif n_siloxane > n_silanol:
        surface_class = "siloxane_rich"
    else:
        surface_class = "mixed_oh"
    confidence = "high" if n_o >= 12 and (n_silanol or n_siloxane) else "medium" if n_o >= 4 else "low"
    return {
        "surface_class": surface_class,
        "surface_classification_method": "distance_based_heuristic",
        "classification_confidence": confidence,
        "n_silanol_oh": n_silanol,
        "n_siloxane_o": n_siloxane,
        "n_surface_si": len(si_idx),
        "n_surface_o": n_o,
        "local_oh_count": n_silanol,
    }


def _rebuild_local_cell(
    coords: np.ndarray,
    box: Tuple[float, float, float],
    config: Dict[str, Any],
) -> Tuple[np.ndarray, Tuple[float, float, float], Dict[str, Any]]:
    patch_cfg = config.get("silica_patch", {})
    min_cell = _mlff_cell_min(config)
    vacuum = float(patch_cfg.get("vacuum_buffer_A", 5.0))
    if len(coords):
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        extent = maxs - mins
        local_box = tuple(float(max(min_cell, extent[i] + 2.0 * vacuum)) for i in range(3))
        centroid = coords.mean(axis=0)
        target = np.array([local_box[0] / 2.0, local_box[1] / 2.0, local_box[2] / 2.0])
        shift = target - centroid
        shifted = coords + shift
        status = "centered_on_local_cell_center"
    else:
        local_box = (min_cell, min_cell, min_cell)
        shift = np.zeros(3)
        shifted = coords
        status = "no_atoms_to_center"
    meta = {
        "cell_source": "rebuilt_local_patch_cell" if patch_cfg.get("rebuild_local_cell", True) else "original_pore_cell",
        "original_pore_cell_lx_A": float(box[0]),
        "original_pore_cell_ly_A": float(box[1]),
        "original_pore_cell_lz_A": float(box[2]),
        "coordinate_centering_status": status,
        "local_cell_origin_shift_xyz": _vec3_text(shift),
    }
    return shifted, local_box, meta


def _cell_failure_reason(box: Tuple[float, float, float], config: Dict[str, Any]) -> str:
    min_cell = _mlff_cell_min(config)
    return "" if all(float(x) >= min_cell for x in box) else "cell_smaller_than_2x_mlff_cutoff"


def _heavy_min_distance(elems: Sequence[str], coords: np.ndarray, left_count: int) -> float:
    left = [i for i in range(left_count) if elems[i] != "H"]
    right = [i for i in range(left_count, len(elems)) if elems[i] != "H"]
    if not left or not right:
        return float("inf")
    return min(float(np.linalg.norm(coords[i] - coords[j])) for i in left for j in right)


def _normal_from_row(row: Any) -> np.ndarray:
    text = str(row.get("inward_normal_xyz", "-1 0 0"))
    try:
        values = np.array([float(x) for x in text.split()[:3]], dtype=float)
    except Exception:
        values = np.array([-1.0, 0.0, 0.0], dtype=float)
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 1.0e-8 else np.array([-1.0, 0.0, 0.0], dtype=float)


def _estimate_pore_radius_A(pore_row: Any, box: Tuple[float, float, float], coords: np.ndarray) -> float:
    try:
        diameter_nm = float(pore_row.get("pore_diameter_nm", ""))
        if diameter_nm > 0:
            return diameter_nm * 10.0 / 2.0
    except Exception:
        pass
    center = _pore_center(box)
    radial = np.linalg.norm(coords[:, :2] - center[:2], axis=1) if len(coords) else np.array([min(box[:2]) / 2.0])
    return max(3.0, float(np.percentile(radial, 80)))


def _inside_pore_fraction(coords: np.ndarray, box: Tuple[float, float, float], radius: float, wall_buffer: float, end_buffer: float) -> float:
    if len(coords) == 0:
        return 0.0
    center = _pore_center(box)
    radial = np.linalg.norm(coords[:, :2] - center[:2], axis=1)
    inside = (radial < max(radius - wall_buffer, 0.1)) & (coords[:, 2] > end_buffer) & (coords[:, 2] < box[2] - end_buffer)
    return float(np.count_nonzero(inside) / len(coords))


def _min_cross_distance(elems: Sequence[str], coords: np.ndarray, split: int) -> float:
    left = [i for i in range(split) if elems[i] != "H"]
    right = [i for i in range(split, len(elems)) if elems[i] != "H"]
    if not left or not right:
        return float("inf")
    return min(float(np.linalg.norm(coords[i] - coords[j])) for i in left for j in right)


def crop_silica_patches(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_pore_dirs(config)
    rows: List[Dict[str, Any]] = []
    pores = _available_pore_rows(config)
    outbase = pore_root(config) / config["paths"]["silica_patches_dir"]
    if pores.empty:
        pd.DataFrame(
            [
                {
                    "patch_id": "no_available_pore_model",
                    "status": "skipped_no_available_pore_model",
                    "source_pore_model_id": "",
                    "patch_extxyz_path": "",
                }
            ]
        ).to_csv(outbase / "silica_patch_manifest.csv", index=False)
        return outbase / "silica_patch_manifest.csv"
    patch_cfg = config["silica_patch"]
    max_atoms = int(patch_cfg["max_atoms_silica_patch"])
    min_silica_atoms = int(patch_cfg.get("min_silica_atoms_for_valid_patch", 40))
    radius = float(patch_cfg.get("target_patch_radius_A", 8.0))
    for _, row in pores.iterrows():
        elems, coords, box = _read_xyz_like(Path(row["pore_model_extxyz_path"]))
        for patch_idx, patch_type in enumerate(patch_cfg.get("patch_types", ["concave_pore_wall_patch"]), start=1):
            keep = _anchor_indices(coords, _pore_center(box), str(patch_type), radius, max_atoms)
            patch_elems = [elems[i] for i in keep]
            patch_coords = coords[keep]
            normal_meta = _normal_metadata(coords, box, keep)
            local_coords, local_box, cell_meta = _rebuild_local_cell(patch_coords, box, config)
            classification = _surface_classification(patch_elems, local_coords, config)
            cell_failure = _cell_failure_reason(local_box, config)
            silica_atoms = sum(1 for el in patch_elems if el in PORE_ELEMENTS)
            coverage_failure = silica_atoms < min_silica_atoms
            usable = not cell_failure and not coverage_failure
            failure_reason = cell_failure or ("patch_too_small_or_poor_surface_coverage" if coverage_failure else "")
            patch_id = f"patch_{row['pore_model_id']}_{patch_idx:03d}_{patch_type}"
            pdir = outbase / patch_id
            pdir.mkdir(parents=True, exist_ok=True)
            atoms = _atoms_from_elements(patch_elems, local_coords)
            write_xyz(pdir / "patch.extxyz", atoms, local_box, ext=True)
            write_pdb(pdir / "patch.pdb", atoms, local_box)
            meta = {
                "patch_id": patch_id,
                "patch_type": patch_type,
                "source_pore_model_id": row["pore_model_id"],
                "status": "available",
                "source": row["source"],
                "usable_for_cp2k_aimd": bool(usable),
                "failure_reason": failure_reason,
                "geometry_status": "available_geometry" if usable else "available_geometry_not_usable_for_cp2k_aimd",
                "n_atoms": len(patch_elems),
                "selection_radius_A": radius,
                **{k: v for k, v in normal_meta.items() if not k.endswith("_array")},
                **cell_meta,
                **classification,
            }
            (pdir / "patch_metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
            rows.append(
                {
                    "patch_id": patch_id,
                    "patch_type": patch_type,
                    "status": "available",
                    "usable_for_cp2k_aimd": bool(usable),
                    "failure_reason": failure_reason,
                    "source_pore_model_id": row["pore_model_id"],
                    "patch_extxyz_path": str(pdir / "patch.extxyz"),
                    "n_atoms": len(patch_elems),
                    "selection_radius_A": radius,
                    **{k: v for k, v in normal_meta.items() if not k.endswith("_array")},
                    **cell_meta,
                    **classification,
                }
            )
    manifest = outbase / "silica_patch_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _patch_rows(config: Dict[str, Any]) -> pd.DataFrame:
    path = pore_root(config) / config["paths"]["silica_patches_dir"] / "silica_patch_manifest.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "usable_for_cp2k_aimd" in df.columns:
        usable = df["usable_for_cp2k_aimd"].astype(str).str.lower().isin({"true", "1", "yes"})
        return df[(df["status"].astype(str) == "available") & usable]
    return df[df["status"].astype(str) == "available"]


def build_aimd_local_structures(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_pore_dirs(config)
    patches = _patch_rows(config)
    outbase = pore_root(config) / config["paths"]["aimd_local_structures_dir"]
    maxn = int(config["aimd_local_matrix"][f"{mode}_max_structures"])
    rows: List[Dict[str, Any]] = []
    if patches.empty:
        rows.append({"aimd_structure_id": "no_available_silica_patch", "status": "skipped_no_available_silica_patch", "source_patch_id": "", "extxyz_path": ""})
    else:
        families = config["aimd_local_matrix"]["families"]
        seeds = config["aimd_local_matrix"]["seeds"]
        made = 0
        for _, patch in patches.iterrows():
            elems, coords, box = _read_xyz_like(Path(patch["patch_extxyz_path"]))
            for family in families:
                for seed in seeds:
                    if made >= maxn:
                        break
                    sid = f"aimd_{made+1:04d}_{family}_seed{seed}"
                    sdir = outbase / sid
                    sdir.mkdir(parents=True, exist_ok=True)
                    pe = "pe" in family
                    pp = "pp" in family
                    all_elems = list(elems)
                    all_coords = [x for x in coords]
                    rng = np.random.default_rng(seed)
                    placement_status = "silica_patch_only"
                    min_polymer_silica_distance = float("inf")
                    if pe or pp:
                        row = {
                            "system_id": "fragment",
                            "seed": seed,
                            "chain_length_backbone": 8,
                            "n_pe_chains": 1 if pe else 0,
                            "n_pp_chains": 1 if pp else 0,
                            "initial_packing_density_g_cm3": 0.5,
                        }
                        topo = build_python_topology(row)
                        normal = _normal_from_row(patch)
                        patch_center = np.array([box[0] / 2.0, box[1] / 2.0, box[2] / 2.0])
                        distances = [float(x) for x in config["aimd_local_matrix"].get("polymer_wall_distances_A", [3.5])]
                        distance = distances[0] if "compressed" in family else distances[min(seed - 1, len(distances) - 1)]
                        polymer_coords = np.array([[a.x, a.y, a.z] for a in topo.atoms], dtype=float)
                        polymer_coords -= polymer_coords.mean(axis=0)
                        base = patch_center + normal * distance
                        placed = polymer_coords + base + rng.normal(0, 0.03, polymer_coords.shape)
                        min_allowed = float(config.get("packing", {}).get("min_heavy_atom_distance_A", 1.6))
                        for attempt in range(40):
                            trial_coords = np.vstack([np.array(all_coords, dtype=float), placed])
                            trial_elems = all_elems + [a.element for a in topo.atoms]
                            min_polymer_silica_distance = _heavy_min_distance(trial_elems, trial_coords, len(all_elems))
                            if min_polymer_silica_distance >= min_allowed:
                                placement_status = "placed_along_inward_normal"
                                break
                            placed = placed + normal * 0.25
                        if placement_status != "placed_along_inward_normal":
                            rows.append(
                                {
                                    "aimd_structure_id": sid,
                                    "status": "skipped_overlap_unresolved",
                                    "source_patch_id": patch["patch_id"],
                                    "family": family,
                                    "extxyz_path": "",
                                    "min_polymer_silica_distance_A": min_polymer_silica_distance,
                                }
                            )
                            made += 1
                            continue
                        for atom, xyz in zip(topo.atoms, placed):
                            all_elems.append(atom.element)
                            all_coords.append(xyz)
                    atoms = _atoms_from_elements(all_elems, np.array(all_coords), 0)
                    write_xyz(sdir / "structure.extxyz", atoms, box, ext=True)
                    write_pdb(sdir / "structure.pdb", atoms, box)
                    (sdir / "metadata.yaml").write_text(
                        yaml.safe_dump(
                            {
                                "aimd_structure_id": sid,
                                "family": family,
                                "source_patch_id": patch["patch_id"],
                                "patch_type": patch.get("patch_type", ""),
                                "status": "available",
                                "placement_status": placement_status,
                                "placement_direction": "inward_normal",
                                "inward_normal_xyz": patch.get("inward_normal_xyz", ""),
                                "min_polymer_silica_distance_A": min_polymer_silica_distance if math.isfinite(min_polymer_silica_distance) else None,
                                "purpose": "aimd_local_training_structure_only_no_cp2k_run",
                            },
                            sort_keys=False,
                        ),
                        encoding="utf-8",
                    )
                    rows.append(
                        {
                            "aimd_structure_id": sid,
                            "status": "available",
                            "source_patch_id": patch["patch_id"],
                            "patch_id": patch["patch_id"],
                            "patch_type": patch.get("patch_type", ""),
                            "family": family,
                            "extxyz_path": str(sdir / "structure.extxyz"),
                            "placement_status": placement_status,
                            "inward_normal_xyz": patch.get("inward_normal_xyz", ""),
                            "min_polymer_silica_distance_A": min_polymer_silica_distance if math.isfinite(min_polymer_silica_distance) else "",
                        }
                    )
                    made += 1
                if made >= maxn:
                    break
    manifest = outbase / "aimd_local_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def build_full_pore_seed_structures(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_pore_dirs(config)
    pores = _available_pore_rows(config)
    outbase = pore_root(config) / config["paths"]["full_pore_seed_structures_dir"]
    maxn = int(config["full_pore_seed_matrix"][f"{mode}_max_systems"])
    rows: List[Dict[str, Any]] = []
    if pores.empty:
        rows.append({"full_pore_seed_id": "no_available_pore_model", "status": "skipped_no_available_pore_model", "source_pore_model_id": "", "extxyz_path": ""})
    else:
        made = 0
        for _, pore in pores.iterrows():
            elems, coords, box = _read_xyz_like(Path(pore["pore_model_extxyz_path"]))
            seed_cfg = config.get("full_pore_seed", {})
            wall_buffer = float(seed_cfg.get("wall_buffer_A", 3.0))
            end_buffer = float(seed_cfg.get("end_buffer_A", 3.0))
            min_inside = float(seed_cfg.get("min_polymer_inside_pore_fraction", 0.95))
            min_silica = float(seed_cfg.get("min_polymer_silica_distance_A", 1.6))
            pore_radius = _estimate_pore_radius_A(pore, box, coords)
            for pe, pp in config["full_pore_seed_matrix"]["pe_pp_compositions"]:
                if made >= maxn:
                    break
                seed = int(config["full_pore_seed_matrix"]["seeds"][0])
                sid = f"full_pore_seed_{made+1:04d}_PE{int(pe*100):02d}_PP{int(pp*100):02d}_seed{seed}"
                sdir = outbase / sid
                sdir.mkdir(parents=True, exist_ok=True)
                row = {
                    "system_id": sid,
                    "seed": seed,
                    "chain_length_backbone": 80,
                    "n_pe_chains": 1 if pe > 0 else 0,
                    "n_pp_chains": 1 if pp > 0 else 0,
                    "initial_packing_density_g_cm3": 0.5,
                }
                topo = build_python_topology(row)
                all_elems = list(elems)
                all_coords = [x for x in coords]
                polymer_raw = np.array([[atom.x, atom.y, atom.z] for atom in topo.atoms], dtype=float)
                polymer_raw -= polymer_raw.mean(axis=0)
                radial = np.linalg.norm(polymer_raw[:, :2], axis=1)
                allowed_radius = max(pore_radius - wall_buffer, 0.5)
                max_radial = max(float(radial.max()), 1.0e-8)
                if max_radial > allowed_radius:
                    polymer_raw[:, :2] *= allowed_radius / max_radial * 0.95
                z_span = float(polymer_raw[:, 2].max() - polymer_raw[:, 2].min()) if len(polymer_raw) else 0.0
                allowed_z = max(box[2] - 2.0 * end_buffer, 1.0)
                if z_span > allowed_z:
                    polymer_raw[:, 2] *= allowed_z / z_span * 0.95
                center = _pore_center(box)
                placed_polymer = polymer_raw + center
                for atom in topo.atoms:
                    all_elems.append(atom.element)
                for xyz in placed_polymer:
                    all_coords.append(xyz)
                combined_coords = np.array(all_coords, dtype=float)
                inside_fraction = _inside_pore_fraction(placed_polymer, box, pore_radius, wall_buffer, end_buffer)
                min_distance = _min_cross_distance(all_elems, combined_coords, len(elems))
                usable = inside_fraction >= min_inside and min_distance >= min_silica
                packing_status = "packed_inside_pore" if usable else "polymer_not_inside_pore_or_overlap"
                atoms = _atoms_from_elements(all_elems, np.array(all_coords), 0)
                write_xyz(sdir / "seed.extxyz", atoms, box, ext=True)
                write_pdb(sdir / "seed.pdb", atoms, box)
                relaxation = {
                    "lammps_relax_performed": False,
                    "relax_is_training_data": False,
                    "relax_is_production_md": False,
                    "mlff_start_structure_kind": "raw_full_pore_seed",
                    "mlff_start_extxyz_path": str(sdir / "seed.extxyz"),
                }
                (sdir / "metadata.yaml").write_text(
                    yaml.safe_dump(
                        {
                            "full_pore_seed_id": sid,
                            "status": "available",
                            "source_pore_model_id": pore["pore_model_id"],
                            "purpose": "mlff_exploration_seed_only_no_mlff_run",
                            "packing_method": "deterministic_cylindrical_sampler",
                            "packing_status": packing_status,
                            "usable_for_mlff_start": bool(usable),
                            "failure_reason": "" if usable else "polymer_not_inside_pore_or_overlap",
                            "polymer_inside_pore_fraction": inside_fraction,
                            "min_polymer_silica_distance_A": min_distance,
                            "relaxation": relaxation,
                        },
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "full_pore_seed_id": sid,
                        "status": "available",
                        "source_pore_model_id": pore["pore_model_id"],
                        "extxyz_path": str(sdir / "seed.extxyz"),
                        "polymer_inside_pore_fraction": inside_fraction,
                        "min_polymer_silica_distance_A": min_distance,
                        "packing_method": "deterministic_cylindrical_sampler",
                        "packing_status": packing_status,
                        "usable_for_mlff_start": bool(usable),
                        "failure_reason": "" if usable else "polymer_not_inside_pore_or_overlap",
                        "mlff_start_structure_kind": "raw_full_pore_seed",
                    }
                )
                made += 1
    manifest = outbase / "full_pore_seed_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def write_cp2k_structure_inputs(config: Dict[str, Any]) -> Path:
    ensure_pore_dirs(config)
    outbase = pore_root(config) / config["paths"]["aimd_exports_dir"]
    rows: List[Dict[str, Any]] = []
    manifest = pore_root(config) / config["paths"]["aimd_local_structures_dir"] / "aimd_local_manifest.csv"
    if manifest.exists():
        df = pd.read_csv(manifest)
        for _, row in df[df["status"].astype(str) == "available"].iterrows():
            src = Path(row["extxyz_path"])
            d = outbase / row["aimd_structure_id"]
            d.mkdir(parents=True, exist_ok=True)
            dst = d / "structure.xyz"
            elems, coords, box = _read_xyz_like(src)
            atoms = _atoms_from_elements(elems, coords)
            write_xyz(dst, atoms, box, ext=False)
            (d / "cp2k_structure_metadata.yaml").write_text(
                yaml.safe_dump({"cp2k_run_performed": False, "aimd_trajectory_generated": False, "source_extxyz": str(src)}, sort_keys=False),
                encoding="utf-8",
            )
            rows.append({"aimd_structure_id": row["aimd_structure_id"], "status": "structure_input_written_no_cp2k_run", "cp2k_xyz_path": str(dst)})
    if not rows:
        rows.append({"aimd_structure_id": "no_available_aimd_structure", "status": "skipped_no_available_aimd_structure", "cp2k_xyz_path": ""})
    out = outbase / "cp2k_structure_input_manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def validate_extxyz(path: Path, allowed: set[str], require_h: bool = False) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "valid": False, "reason": "missing"}
    elems, coords, box = _read_xyz_like(path)
    return {
        "exists": True,
        "valid": set(elems).issubset(allowed) and (not require_h or "H" in elems) and all(x > 0 for x in box),
        "n_atoms": len(elems),
        "elements": ";".join(sorted(set(elems))),
        "has_explicit_h": "H" in elems,
        "box_lx_A": box[0],
        "box_ly_A": box[1],
        "box_lz_A": box[2],
        "pbc": True,
    }


def validate_from_manifest(config: Dict[str, Any], manifest_rel: str, id_col: str, path_col: str, out_name: str, allowed: set[str], require_h: bool = False) -> Path:
    ensure_pore_dirs(config)
    manifest = pore_root(config) / manifest_rel
    rows: List[Dict[str, Any]] = []
    if manifest.exists():
        df = pd.read_csv(manifest)
        for _, row in df.iterrows():
            rec = {id_col: row.get(id_col, ""), "status": row.get("status", "")}
            path = str(row.get(path_col, "") or "")
            rec.update(validate_extxyz(Path(path), allowed, require_h) if path else {"exists": False, "valid": False, "reason": "no_structure_path"})
            rows.append(rec)
    if not rows:
        rows.append({id_col: "no_manifest_rows", "status": "skipped", "exists": False, "valid": False})
    out = pore_root(config) / config["paths"]["logs_dir"] / out_name
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def validate_pore_structures(config: Dict[str, Any]) -> Path:
    manifest = f"{config['paths']['porems_models_dir']}/pore_model_manifest.csv"
    return validate_from_manifest(config, manifest, "pore_model_id", "pore_model_extxyz_path", "pore_structure_validation.csv", PORE_ELEMENTS, True)


def validate_aimd_local_structures(config: Dict[str, Any]) -> Path:
    manifest = f"{config['paths']['aimd_local_structures_dir']}/aimd_local_manifest.csv"
    return validate_from_manifest(config, manifest, "aimd_structure_id", "extxyz_path", "aimd_local_validation.csv", AIMD_ELEMENTS, True)


def validate_full_pore_seed_structures(config: Dict[str, Any]) -> Path:
    manifest = f"{config['paths']['full_pore_seed_structures_dir']}/full_pore_seed_manifest.csv"
    return validate_from_manifest(config, manifest, "full_pore_seed_id", "extxyz_path", "full_pore_seed_validation.csv", AIMD_ELEMENTS, True)


def export_pore_aimd_manifests(config: Dict[str, Any]) -> Tuple[Path, Path]:
    ensure_pore_dirs(config)
    root = pore_root(config)
    outdir = root / config["paths"]["aimd_exports_dir"]
    pieces = []
    for rel, kind in [
        (f"{config['paths']['porems_models_dir']}/pore_model_manifest.csv", "pore_model"),
        (f"{config['paths']['silica_patches_dir']}/silica_patch_manifest.csv", "silica_patch"),
        (f"{config['paths']['aimd_local_structures_dir']}/aimd_local_manifest.csv", "aimd_local"),
        (f"{config['paths']['full_pore_seed_structures_dir']}/full_pore_seed_manifest.csv", "full_pore_seed"),
        (f"{config['paths']['aimd_exports_dir']}/cp2k_structure_input_manifest.csv", "cp2k_structure_input"),
    ]:
        p = root / rel
        if p.exists():
            df = pd.read_csv(p)
            df.insert(0, "manifest_kind", kind)
            pieces.append(df)
    if pieces:
        combined = pd.concat(pieces, ignore_index=True, sort=False)
    else:
        combined = pd.DataFrame([{"manifest_kind": "none", "status": "no_pore_aimd_outputs_available"}])
    csv_path = outdir / "pore_aimd_master_manifest.csv"
    json_path = outdir / "pore_aimd_master_manifest.json"
    combined.to_csv(csv_path, index=False)
    json_path.write_text(combined.to_json(orient="records", indent=2), encoding="utf-8")
    return csv_path, json_path


def assert_no_fake_cp2k_or_mlff(root: Path) -> List[str]:
    forbidden = []
    names = ["aimd.traj", "cp2k.out", "mlff_production", "production.lammpstrj", "prod.lammpstrj"]
    for name in names:
        forbidden.extend(str(p) for p in root.rglob(name))
    return forbidden
