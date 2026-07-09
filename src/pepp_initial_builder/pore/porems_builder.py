from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from pepp_initial_builder.common.io import write_pdb, write_xyz
from pepp_initial_builder.common.tables import Atom
from pepp_initial_builder.pore.config import ensure_pore_dirs, pore_root
from pepp_initial_builder.pore.porems_discovery import discover_porems

PORE_ELEMENTS = {"Si", "O", "H"}
AIMD_ELEMENTS = {"C", "H", "O", "Si"}


def read_xyz_like(path: str | Path) -> Tuple[List[str], np.ndarray, Tuple[float, float, float]]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n_atoms = int(lines[0].strip())
    comment = lines[1] if len(lines) > 1 else ""
    box = (30.0, 30.0, 30.0)
    if "Lattice=" in comment:
        part = comment.split('Lattice="', 1)[1].split('"', 1)[0].split()
        if len(part) >= 9:
            box = (float(part[0]), float(part[4]), float(part[8]))
    elems: List[str] = []
    coords: List[List[float]] = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        elems.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elems, np.array(coords, dtype=float), box


def atoms_from_elements(elems: Sequence[str], coords: np.ndarray, chain_id: int = 0) -> List[Atom]:
    atoms = []
    for i, (element, xyz) in enumerate(zip(elems, coords), start=1):
        atoms.append(Atom(i, element, element, "silica" if element in PORE_ELEMENTS else "polymer", chain_id, "silica" if element in PORE_ELEMENTS else "polymer", 0, False, False, False, element == "H", 0, float(xyz[0]), float(xyz[1]), float(xyz[2]), chain_id, 0.0))
    return atoms


def validate_pore_elements(elems: Sequence[str]) -> Dict[str, Any]:
    counts = {element: elems.count(element) for element in sorted(set(elems))}
    return {"elements": sorted(set(elems)), "element_counts": counts, "only_si_o_h": set(elems).issubset(PORE_ELEMENTS), "explicit_surface_h_present": counts.get("H", 0) > 0, "obvious_dangling_check": "not_evaluated_without_porems_bond_order"}


def map_porems_element(symbol: str) -> str:
    text = str(symbol).strip().upper()
    if text.startswith("SI") or text in {"SIL", "S"}:
        return "Si"
    if text.startswith("H"):
        return "H"
    if text.startswith("O") or text in {"OM", "OX", "OH", "OS", "SLX", "SL", "SLG"}:
        return "O"
    return str(symbol).strip()


def run_porems_external(python_executable: str, model_dir: Path, *, pore_diameter_nm: float, pore_length_nm: float, hydroxylation_mode: str):
    raw_dir = model_dir / "porems_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_path = raw_dir / "porems_atoms.json"
    script_path = raw_dir / "generate_porems_model.py"
    box_xy = float(pore_diameter_nm) + 2.0
    hydro = [0, 0]
    script = f"""
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
""".lstrip()
    script_path.write_text(script, encoding="utf-8")
    log_path = model_dir / "porems_build.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run([python_executable, str(script_path)], text=True, stdout=log, stderr=subprocess.STDOUT, cwd=raw_dir, timeout=300, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"PoreMS external builder failed with exit code {proc.returncode}; see {log_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    elems = [map_porems_element(atom["type"]) for atom in data["atoms"]]
    coords = np.array([[10.0 * float(x) for x in atom["pos_nm"]] for atom in data["atoms"]], dtype=float)
    box = tuple(10.0 * float(x) for x in data["box_nm"])
    return elems, coords, box, str(log_path)


def manual_pore_models(config: Dict[str, Any]) -> List[Path]:
    base = pore_root(config) / config["paths"]["porems_models_dir"]
    paths: List[Path] = []
    for folder in sorted(base.glob("manual_*")):
        for name in ["pore_model.extxyz", "pore_model.xyz"]:
            path = folder / name
            if path.exists():
                paths.append(path)
                break
    return paths


def build_porems_pores(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_pore_dirs(config)
    discovery = discover_porems(config)
    outdir = pore_root(config) / config["paths"]["porems_models_dir"]
    rows: List[Dict[str, Any]] = []
    manual = manual_pore_models(config)
    if not discovery["available"] and not manual:
        rows.append({"pore_model_id": "porems_unavailable", "status": "failed_porems_not_available", "source": "none", "pore_model_extxyz_path": "", "pore_model_pdb_path": "", "pore_diameter_nm": "", "pore_length_nm": "", "hydroxylation_mode": ""})
    if discovery["available"] and not manual:
        max_models = int(config["porems"].get(f"max_pore_models_{mode}", 1))
        made = 0
        python_exe = discovery.get("python_executable")
        for diameter in config["porems"]["pore_diameter_nm_candidates"]:
            for length in config["porems"]["pore_length_nm_candidates"]:
                for hydro in config["porems"]["hydroxylation_modes"]:
                    if made >= max_models:
                        break
                    pore_model_id = f"porems_D{float(diameter):.1f}nm_L{float(length):.1f}nm_{hydro}".replace(".", "p")
                    model_dir = outdir / pore_model_id
                    model_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        if not python_exe:
                            raise RuntimeError("PoreMS available but no Python executable was discovered")
                        elems, coords, box, log_path = run_porems_external(python_exe, model_dir, pore_diameter_nm=float(diameter), pore_length_nm=float(length), hydroxylation_mode=str(hydro))
                        atoms = atoms_from_elements(elems, coords)
                        write_xyz(model_dir / "pore_model.extxyz", atoms, box, ext=True)
                        write_pdb(model_dir / "pore_model.pdb", atoms, box)
                        validation = validate_pore_elements(elems)
                        status = "available" if validation["only_si_o_h"] else "failed_validation"
                        meta = {"pore_model_id": pore_model_id, "source": "porems_python_package", "porems_python_executable": python_exe, "porems_version": discovery.get("version"), "status": status, "pore_diameter_nm": float(diameter), "pore_length_nm": float(length), "surface": config["porems"]["surface"], "hydroxylation_mode": hydro, "validation": validation, "porems_build_log": log_path}
                        (model_dir / "pore_metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
                        rows.append({"pore_model_id": pore_model_id, "status": status, "source": "porems_python_package", "pore_model_extxyz_path": str(model_dir / "pore_model.extxyz"), "pore_model_pdb_path": str(model_dir / "pore_model.pdb"), "pore_diameter_nm": float(diameter), "pore_length_nm": float(length), "hydroxylation_mode": hydro})
                    except Exception as exc:
                        (model_dir / "porems_build.log").write_text(str(exc) + "\n", encoding="utf-8")
                        rows.append({"pore_model_id": pore_model_id, "status": "failed_porems_build_failed", "source": "porems_python_package", "pore_model_extxyz_path": "", "pore_model_pdb_path": "", "pore_diameter_nm": float(diameter), "pore_length_nm": float(length), "hydroxylation_mode": hydro})
                    made += 1
                if made >= max_models:
                    break
            if made >= max_models:
                break
    for source_path in manual:
        model_dir = source_path.parent
        elems, coords, box = read_xyz_like(source_path)
        atoms = atoms_from_elements(elems, coords)
        write_xyz(model_dir / "pore_model.extxyz", atoms, box, ext=True)
        write_pdb(model_dir / "pore_model.pdb", atoms, box)
        meta = {"pore_model_id": model_dir.name, "source": "manual_user_input", "status": "available_manual_user_input", "validation": validate_pore_elements(elems), "pore_shape": config["porems"]["pore_shape"], "surface": config["porems"]["surface"]}
        (model_dir / "pore_metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
        rows.append({"pore_model_id": model_dir.name, "status": meta["status"], "source": "manual_user_input", "pore_model_extxyz_path": str(model_dir / "pore_model.extxyz"), "pore_model_pdb_path": str(model_dir / "pore_model.pdb"), "pore_diameter_nm": "", "pore_length_nm": "", "hydroxylation_mode": ""})
    manifest = outdir / "pore_model_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def available_pore_rows(config: Dict[str, Any]) -> pd.DataFrame:
    path = pore_root(config) / config["paths"]["porems_models_dir"] / "pore_model_manifest.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["status"].astype(str).str.startswith("available")]
