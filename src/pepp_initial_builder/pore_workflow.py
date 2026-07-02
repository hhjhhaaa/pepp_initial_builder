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
    max_atoms = int(config["silica_patch"]["max_atoms_silica_patch"])
    for _, row in pores.iterrows():
        elems, coords, box = _read_xyz_like(Path(row["pore_model_extxyz_path"]))
        by_elem = {el: [i for i, e in enumerate(elems) if e == el] for el in sorted(set(elems))}
        keep_list: List[int] = []
        for el, quota in [("H", max(1, max_atoms // 6)), ("O", max(1, max_atoms // 2)), ("Si", max_atoms)]:
            for idx in by_elem.get(el, []):
                if len(keep_list) >= max_atoms:
                    break
                if el == "Si" and len(keep_list) < max_atoms:
                    keep_list.append(idx)
                elif el != "Si" and sum(1 for k in keep_list if elems[k] == el) < quota:
                    keep_list.append(idx)
            if len(keep_list) >= max_atoms:
                break
        if len(keep_list) < max_atoms:
            for idx in range(len(elems)):
                if idx not in keep_list:
                    keep_list.append(idx)
                if len(keep_list) >= max_atoms:
                    break
        keep = np.array(sorted(keep_list[: min(max_atoms, len(keep_list))]))
        patch_id = f"patch_{row['pore_model_id']}_001"
        pdir = outbase / patch_id
        pdir.mkdir(parents=True, exist_ok=True)
        atoms = _atoms_from_elements([elems[i] for i in keep], coords[keep])
        write_xyz(pdir / "patch.extxyz", atoms, box, ext=True)
        write_pdb(pdir / "patch.pdb", atoms, box)
        meta = {"patch_id": patch_id, "source_pore_model_id": row["pore_model_id"], "status": "available", "source": row["source"]}
        (pdir / "patch_metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        rows.append({"patch_id": patch_id, "status": "available", "source_pore_model_id": row["pore_model_id"], "patch_extxyz_path": str(pdir / "patch.extxyz")})
    manifest = outbase / "silica_patch_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _patch_rows(config: Dict[str, Any]) -> pd.DataFrame:
    path = pore_root(config) / config["paths"]["silica_patches_dir"] / "silica_patch_manifest.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
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
                        shift = np.array([0.0, 0.0, 4.0 + 0.2 * seed])
                        for atom in topo.atoms:
                            all_elems.append(atom.element)
                            all_coords.append(np.array([atom.x, atom.y, atom.z]) + shift + rng.normal(0, 0.05, 3))
                    atoms = _atoms_from_elements(all_elems, np.array(all_coords), 0)
                    write_xyz(sdir / "structure.extxyz", atoms, box, ext=True)
                    write_pdb(sdir / "structure.pdb", atoms, box)
                    (sdir / "metadata.yaml").write_text(
                        yaml.safe_dump(
                            {
                                "aimd_structure_id": sid,
                                "family": family,
                                "source_patch_id": patch["patch_id"],
                                "status": "available",
                                "purpose": "aimd_local_training_structure_only_no_cp2k_run",
                            },
                            sort_keys=False,
                        ),
                        encoding="utf-8",
                    )
                    rows.append({"aimd_structure_id": sid, "status": "available", "source_patch_id": patch["patch_id"], "family": family, "extxyz_path": str(sdir / "structure.extxyz")})
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
                for atom in topo.atoms:
                    all_elems.append(atom.element)
                    all_coords.append(np.array([atom.x, atom.y, atom.z]) + np.array([box[0] / 2, box[1] / 2, box[2] / 2]))
                atoms = _atoms_from_elements(all_elems, np.array(all_coords), 0)
                write_xyz(sdir / "seed.extxyz", atoms, box, ext=True)
                write_pdb(sdir / "seed.pdb", atoms, box)
                (sdir / "metadata.yaml").write_text(
                    yaml.safe_dump(
                        {"full_pore_seed_id": sid, "status": "available", "source_pore_model_id": pore["pore_model_id"], "purpose": "mlff_exploration_seed_only_no_mlff_run"},
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                rows.append({"full_pore_seed_id": sid, "status": "available", "source_pore_model_id": pore["pore_model_id"], "extxyz_path": str(sdir / "seed.extxyz")})
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
    return validate_from_manifest(config, "data/porems_models/pore_model_manifest.csv", "pore_model_id", "pore_model_extxyz_path", "pore_structure_validation.csv", PORE_ELEMENTS, True)


def validate_aimd_local_structures(config: Dict[str, Any]) -> Path:
    return validate_from_manifest(config, "data/aimd_local_structures/aimd_local_manifest.csv", "aimd_structure_id", "extxyz_path", "aimd_local_validation.csv", AIMD_ELEMENTS, True)


def validate_full_pore_seed_structures(config: Dict[str, Any]) -> Path:
    return validate_from_manifest(config, "data/full_pore_seed_structures/full_pore_seed_manifest.csv", "full_pore_seed_id", "extxyz_path", "full_pore_seed_validation.csv", AIMD_ELEMENTS, True)


def export_pore_aimd_manifests(config: Dict[str, Any]) -> Tuple[Path, Path]:
    ensure_pore_dirs(config)
    root = pore_root(config)
    outdir = root / config["paths"]["aimd_exports_dir"]
    pieces = []
    for rel, kind in [
        ("data/porems_models/pore_model_manifest.csv", "pore_model"),
        ("data/silica_patches/silica_patch_manifest.csv", "silica_patch"),
        ("data/aimd_local_structures/aimd_local_manifest.csv", "aimd_local"),
        ("data/full_pore_seed_structures/full_pore_seed_manifest.csv", "full_pore_seed"),
        ("data/aimd_exports/cp2k_structure_input_manifest.csv", "cp2k_structure_input"),
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
