from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from openbabel import pybel

from pepp_initial_builder.common.openbabel import read_xyz_elements_coords
from pepp_initial_builder.cp2k_aimd.config import ensure_dirs, p, write_rows
from pepp_initial_builder.pore.config import ensure_pore_dirs, pore_root
from pepp_initial_builder.pore.porems_builder import available_pore_rows, build_porems_pores, read_xyz_like
from pepp_initial_builder.pore.surface_classifier import anchor_indices, pore_center, rebuild_local_cell
from pepp_initial_builder.polymer.emc_builder import write_emc_chain_template

AtomSet = Tuple[List[str], np.ndarray]


def _unit(v: np.ndarray, fallback: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 1.0e-8:
        return v / n
    return np.array(fallback, dtype=float)


def _rotation_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a = _unit(source)
    b = _unit(target)
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if np.linalg.norm(cross) < 1.0e-8:
        return np.eye(3) if dot > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / max(np.linalg.norm(cross) ** 2, 1.0e-12))


def _read_xyz(path: str | Path) -> AtomSet:
    elems, coords = read_xyz_elements_coords(path)
    return elems, np.array(coords, dtype=float)


def _write_extxyz(path: Path, elems: Sequence[str], coords: np.ndarray, box: Sequence[float], meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        f'Lattice="{float(box[0]):.8f} 0 0 0 {float(box[1]):.8f} 0 0 0 {float(box[2]):.8f}"',
        "Properties=species:S:1:pos:R:3",
        'pbc="T T T"',
    ]
    for key, value in meta.items():
        text = str(value)
        fields.append(f'{key}="{text}"' if any(ch.isspace() for ch in text) else f"{key}={text}")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(elems)}\n")
        handle.write(" ".join(fields) + "\n")
        for elem, xyz in zip(elems, coords):
            handle.write(f"{elem} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")


def _bond_counts(elems: Sequence[str], coords: np.ndarray) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    si_to_o: Dict[int, List[int]] = {}
    o_to_si: Dict[int, List[int]] = {}
    si_idx = [i for i, e in enumerate(elems) if e == "Si"]
    o_idx = [i for i, e in enumerate(elems) if e == "O"]
    for si in si_idx:
        for oi in o_idx:
            d = float(np.linalg.norm(coords[si] - coords[oi]))
            if d <= 2.05:
                si_to_o.setdefault(si, []).append(oi)
                o_to_si.setdefault(oi, []).append(si)
    return si_to_o, o_to_si


def _oh_counts(elems: Sequence[str], coords: np.ndarray) -> Dict[int, List[int]]:
    o_to_h: Dict[int, List[int]] = {}
    o_idx = [i for i, e in enumerate(elems) if e == "O"]
    h_idx = [i for i, e in enumerate(elems) if e == "H"]
    for oi in o_idx:
        for hi in h_idx:
            if float(np.linalg.norm(coords[oi] - coords[hi])) <= 1.25:
                o_to_h.setdefault(oi, []).append(hi)
    return o_to_h


def _cap_silica_once(elems_in: Sequence[str], coords_in: np.ndarray) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    elems = list(elems_in)
    coords = np.array(coords_in, dtype=float).copy()
    si_to_o, o_to_si = _bond_counts(elems, coords)
    o_to_h = _oh_counts(elems, coords)
    keep = []
    for idx, elem in enumerate(elems):
        if elem == "O" and not o_to_si.get(idx):
            continue
        if elem == "H" and not any(idx in hs for hs in o_to_h.values()):
            continue
        keep.append(idx)
    elems = [elems[i] for i in keep]
    coords = coords[keep]
    centroid = coords.mean(axis=0)
    si_to_o, o_to_si = _bond_counts(elems, coords)
    o_to_h = _oh_counts(elems, coords)
    added_h = 0
    added_oh = 0

    new_elems: List[str] = []
    new_coords: List[np.ndarray] = []
    for oi, elem in enumerate(elems):
        if elem != "O":
            continue
        si_neighbors = o_to_si.get(oi, [])
        if len(si_neighbors) == 1 and not o_to_h.get(oi):
            direction = _unit(coords[oi] - coords[si_neighbors[0]], coords[oi] - centroid)
            new_elems.append("H")
            new_coords.append(coords[oi] + 0.96 * direction)
            added_h += 1

    if new_coords:
        elems.extend(new_elems)
        coords = np.vstack([coords, np.array(new_coords)])

    si_to_o, _o_to_si = _bond_counts(elems, coords)
    new_elems = []
    new_coords = []
    tetra = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, -1.0, -1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([-1.0, -1.0, 1.0]),
    ]
    for si, elem in enumerate(elems):
        if elem != "Si":
            continue
        current = si_to_o.get(si, [])
        missing = max(0, 4 - len(current))
        if missing == 0:
            continue
        existing_dirs = [_unit(coords[oi] - coords[si]) for oi in current]
        base = _unit(-(np.sum(existing_dirs, axis=0) if existing_dirs else coords[si] - centroid), coords[si] - centroid)
        prefer_tetra_ring = len(existing_dirs) == 1 and missing >= 2
        candidates = [] if prefer_tetra_ring else [base]
        if existing_dirs:
            primary = existing_dirs[0]
            helper = np.array([1.0, 0.0, 0.0]) if abs(primary[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
            u = _unit(np.cross(primary, helper))
            v = _unit(np.cross(primary, u))
            ring_scale = math.sqrt(8.0 / 9.0)
            for angle in (0.0, 120.0, 240.0):
                a = math.radians(angle)
                candidates.append(_unit((-1.0 / 3.0) * primary + ring_scale * (math.cos(a) * u + math.sin(a) * v)))
        if not prefer_tetra_ring:
            candidates += [_unit(base + 0.45 * t) for t in tetra]
            candidates += [_unit(t) for t in tetra]
            for z in np.linspace(-0.9, 0.9, 7):
                radius = math.sqrt(max(0.0, 1.0 - float(z) ** 2))
                for angle in np.linspace(0.0, 300.0, 6):
                    a = math.radians(float(angle))
                    candidates.append(_unit(np.array([radius * math.cos(a), radius * math.sin(a), float(z)])))
        used = 0
        pending_heavy: List[np.ndarray] = []
        chosen_dirs: List[np.ndarray] = []
        while used < missing:
            best: Tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
            for cand in candidates:
                if any(float(np.dot(cand, prev)) > 0.85 for prev in chosen_dirs):
                    continue
                o_pos = coords[si] + 1.62 * cand
                other_oxygen = [
                    coords[i]
                    for i, other in enumerate(elems)
                    if other == "O"
                ] + pending_heavy
                min_oo = float("inf")
                if other_oxygen:
                    min_oo = float(np.min(np.linalg.norm(np.array(other_oxygen) - o_pos, axis=1)))
                    if min_oo < 2.05:
                        continue
                other_heavy = [
                    coords[i]
                    for i, other in enumerate(elems)
                    if other != "H" and i != si
                ] + pending_heavy
                min_heavy = float("inf")
                if other_heavy:
                    min_heavy = float(np.min(np.linalg.norm(np.array(other_heavy) - o_pos, axis=1)))
                    if min_heavy < 1.35:
                        continue
                score = min(min_oo, 3.0) + 0.25 * min(min_heavy, 3.0)
                h_pos = o_pos + 0.96 * cand
                if best is None or score > best[0]:
                    best = (score, cand, o_pos, h_pos)
            if best is None:
                break
            _score, cand, o_pos, h_pos = best
            new_elems.extend(["O", "H"])
            new_coords.extend([o_pos, h_pos])
            pending_heavy.append(o_pos)
            chosen_dirs.append(cand)
            added_oh += 1
            used += 1

    if new_coords:
        elems.extend(new_elems)
        coords = np.vstack([coords, np.array(new_coords)])

    si_to_o, o_to_si = _bond_counts(elems, coords)
    o_to_h = _oh_counts(elems, coords)
    bad_si_indices = [i for i, e in enumerate(elems) if e == "Si" and len(si_to_o.get(i, [])) < 4]
    bad_si = len(bad_si_indices)
    bad_o = sum(1 for i, e in enumerate(elems) if e == "O" and not (len(o_to_si.get(i, [])) >= 2 or (len(o_to_si.get(i, [])) == 1 and len(o_to_h.get(i, [])) >= 1)))
    meta = {"added_boundary_H": added_h, "added_boundary_OH": added_oh, "undercoordinated_Si_after_capping": bad_si, "uncapped_O_after_capping": bad_o, "_undercoordinated_si_indices": ",".join(str(i) for i in bad_si_indices)}
    return elems, coords, meta


def _cap_silica(elems_in: Sequence[str], coords_in: np.ndarray) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    elems = list(elems_in)
    coords = np.array(coords_in, dtype=float).copy()
    last_meta: Dict[str, Any] = {}
    for _pass in range(8):
        capped_elems, capped_coords, meta = _cap_silica_once(elems, coords)
        bad_si_text = str(meta.pop("_undercoordinated_si_indices", ""))
        if meta["undercoordinated_Si_after_capping"] == 0:
            return capped_elems, capped_coords, meta
        bad_si = {int(item) for item in bad_si_text.split(",") if item}
        if not bad_si:
            break
        keep = [i for i in range(len(capped_elems)) if i not in bad_si]
        elems = [capped_elems[i] for i in keep]
        coords = capped_coords[keep]
        last_meta = meta
    raise RuntimeError(f"Unable to produce a chemically closed silica patch; last capping state: {last_meta}")


def _center_fragment(elems: Sequence[str], coords: np.ndarray) -> AtomSet:
    c = coords.mean(axis=0)
    centered = coords - c
    if len(centered) >= 3:
        _values, vectors = np.linalg.eigh(np.cov(centered.T))
        order = np.argsort(_values)[::-1]
        basis = vectors[:, order]
        if np.linalg.det(basis) < 0:
            basis[:, -1] *= -1.0
        centered = centered @ basis
        centered[:, 2] -= centered[:, 2].mean()
    return list(elems), centered


def _make_ps_fragment(kind: str) -> AtomSet:
    smiles = {
        "PS_dimer": "CC(c1ccccc1)CC(c2ccccc2)C",
        "PS_trimer": "CC(c1ccccc1)CC(c2ccccc2)CC(c3ccccc3)C",
    }[kind]
    mol = pybel.readstring("smi", smiles)
    mol.addh()
    mol.make3D(forcefield="mmff94", steps=300)
    elems = [atom.type.rstrip("0123456789") or atom.atomicnum for atom in mol.atoms]
    elems = [str(e).capitalize().replace("Cl", "C") if not isinstance(e, int) else pybel.ob.GetSymbol(e) for e in elems]
    coords = np.array([[atom.coords[0], atom.coords[1], atom.coords[2]] for atom in mol.atoms], dtype=float)
    elems = ["C" if e.startswith("C") else "H" if e.startswith("H") else "O" if e.startswith("O") else e for e in elems]
    return _center_fragment(elems, coords)


def _make_emc_fragment(config: Dict[str, Any], chain_type: str, chain_length: int, seed: int, outdir: Path) -> AtomSet:
    conda_bin = Path(os.environ.get("CONDA_PREFIX", "/public/home/jinhao.hu/.conda/envs/peppmixure")) / "bin"
    os.environ["PATH"] = f"{conda_bin}:{os.environ.get('PATH', '')}"
    template = write_emc_chain_template(config, chain_type, chain_length, seed, outdir)
    return _center_fragment(*_read_xyz(template["xyz"]))


def _packmol_executable(config: Dict[str, Any]) -> str:
    tools = config.get("tools", {})
    executable = tools.get("packmol_executable")
    if not executable:
        raise RuntimeError("tools.packmol_executable is required")
    candidate = Path(str(executable)).expanduser()
    if not candidate.exists():
        raise RuntimeError(f"Configured Packmol executable does not exist: {candidate}")
    return str(candidate)


def _write_packmol_xyz(path: Path, elems: Sequence[str], coords: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(elems)}\n")
        handle.write("generated_for_packmol\n")
        for elem, xyz in zip(elems, coords):
            handle.write(f"{elem} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")


def _run_packmol(input_path: Path, executable: str, timeout_seconds: int) -> None:
    log_path = input_path.parent / "packmol.log"
    with input_path.open("rb") as stdin, log_path.open("wb") as stdout:
        proc = subprocess.run([executable], stdin=stdin, stdout=stdout, stderr=subprocess.STDOUT, cwd=input_path.parent, timeout=timeout_seconds, check=False)
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    if proc.returncode != 0 or "Success!" not in text:
        raise RuntimeError(f"Packmol failed for {input_path}; see {log_path}")


def _packmol_success(log_path: str) -> str:
    path = Path(log_path)
    if not path.exists():
        return "not_applicable"
    return "success" if "Success!" in path.read_text(encoding="utf-8", errors="ignore") else "failed"


def _write_manual_structure_review(config: Dict[str, Any], review_rows: List[Dict[str, Any]]) -> None:
    logs_dir = p(config, "logs_dir")
    csv_path = logs_dir / "manual_structure_review.csv"
    md_path = logs_dir / "manual_structure_review.md"
    write_rows(csv_path, review_rows)

    lines = [
        "# Manual structure review",
        "",
        "Status: pending user review before CP2K input generation.",
        "",
        "Generation flow:",
        "1. Build one PoreMS-derived silica patch and align the local slab frame.",
        "2. Cap silica boundaries with H/OH; unclosable boundary Si atoms are removed and capping is rerun.",
        "3. Generate PE/PP fragments through EMC and PS fragments as phenyl-side-chain styrene oligomers.",
        "4. Pack fixed H/OH-capped silica plus polymer fragments with Packmol inside the configured surface box.",
        "5. Treat silica closure and atom-overlap checks as hard gates; polymer-silica distance is reported for manual review, not used as a rejection gate.",
        "",
        "| index | structure | atoms | status | review | min polymer-silica A | Si undercoord | uncapped O | min heavy-heavy A | packmol | extxyz |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for idx, row in enumerate(review_rows):
        lines.append(
            "| {idx} | {sid} | {natoms} | {status} | {review} | {dist} | {bad_si} | {bad_o} | {heavy} | {packmol} | {extxyz} |".format(
                idx=idx,
                sid=row["aimd_structure_id"],
                natoms=row["n_atoms"],
                status=row["status"],
                review=row["manual_review_status"],
                dist=row["min_polymer_silica_distance_A"],
                bad_si=row["undercoordinated_Si_after_capping"],
                bad_o=row["uncapped_O_after_capping"],
                heavy=row["min_heavy_heavy_distance_A"],
                packmol=row["packmol_status"],
                extxyz=row["extxyz_path"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_surface_packmol_input(packmol_dir: Path, silica: AtomSet, fragments: Sequence[Tuple[str, AtomSet]], settings: Dict[str, Any], seed: int) -> Path:
    silica_elems, silica_coords = silica
    template_dir = packmol_dir / "templates"
    silica_xyz = template_dir / "fixed_h_capped_silica.xyz"
    _write_packmol_xyz(silica_xyz, silica_elems, silica_coords)

    coords = np.array(silica_coords, dtype=float)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    top_z = float(maxs[2])
    padding = float(settings.get("packmol_lateral_padding_A", 7.0))
    gap = float(settings.get("packmol_surface_gap_A", 2.1))
    layer = float(settings.get("packmol_surface_layer_thickness_A", 7.5))
    xlo, xhi = float(mins[0] - padding), float(maxs[0] + padding)
    ylo, yhi = float(mins[1] - padding), float(maxs[1] + padding)
    zlo, zhi = top_z + gap, top_z + gap + layer

    lines = [
        f"tolerance {float(settings.get('packmol_tolerance_A', 1.85)):.6f}",
        "filetype xyz",
        f"output {packmol_dir / 'packed_surface.xyz'}",
        f"seed {int(seed)}",
        f"maxit {int(settings.get('packmol_maxit', 5000))}",
        "",
        f"structure {silica_xyz}",
        "  number 1",
        "  fixed 0.0 0.0 0.0 0.0 0.0 0.0",
        "end structure",
        "",
    ]
    for idx, (name, (frag_elems, frag_coords)) in enumerate(fragments, start=1):
        template = template_dir / f"fragment_{idx:02d}_{name}.xyz"
        _write_packmol_xyz(template, frag_elems, frag_coords)
        lines.extend(
            [
                f"structure {template}",
                "  number 1",
                f"  inside box {xlo:.6f} {ylo:.6f} {zlo:.6f} {xhi:.6f} {yhi:.6f} {zhi:.6f}",
                "end structure",
                "",
            ]
        )
    packmol_dir.mkdir(parents=True, exist_ok=True)
    input_path = packmol_dir / "packmol.inp"
    input_path.write_text("\n".join(lines), encoding="utf-8")
    return input_path


def _place_fragments(
    silica_elems: Sequence[str],
    silica_coords: np.ndarray,
    fragments: Sequence[Tuple[str, AtomSet]],
    box: Sequence[float],
    config: Dict[str, Any],
    structure_dir: Path,
    seed: int,
) -> Tuple[List[str], np.ndarray, Tuple[float, float, float], Dict[str, Any]]:
    polymer_start = len(silica_elems)
    if fragments:
        packmol_dir = structure_dir / "packmol"
        settings = config.get("closed_small_anchors", {})
        input_path = _write_surface_packmol_input(packmol_dir, (list(silica_elems), np.array(silica_coords, dtype=float)), fragments, settings, seed)
        output_path = packmol_dir / "packed_surface.xyz"
        _run_packmol(input_path, _packmol_executable(config), int(settings.get("packmol_timeout_seconds", 300)))
        if not output_path.exists():
            raise RuntimeError(f"Packmol reported success but did not write {output_path}")
        elems, coords = _read_xyz(output_path)
        packing_meta: Dict[str, Any] = {
            "packing_method": "packmol_surface_box",
            "packmol_input_path": str(input_path),
            "packmol_log_path": str(packmol_dir / "packmol.log"),
            "packmol_output_path": str(output_path),
        }
    else:
        elems = list(silica_elems)
        coords = np.array(silica_coords, dtype=float).copy()
        packing_meta = {"packing_method": "silica_only_no_packmol"}
    min_poly_silica = min_cross_distance(elems, coords, polymer_start) if fragments else float("inf")
    min_all = _min_all_distance(elems, coords)
    min_all_atoms = _min_all_atom_distance(elems, coords)
    min_heavy = _min_heavy_heavy_distance(elems, coords)
    min_oo = _min_oxygen_oxygen_distance(elems, coords)
    mins = coords.min(axis=0)
    coords = coords - mins + 7.0
    extent = coords.max(axis=0) + 7.0
    box = tuple(float(max(float(box[i]), extent[i])) for i in range(3))
    return elems, coords, box, {"min_polymer_silica_distance_A": f"{min_poly_silica:.3f}", "min_all_pair_distance_A": f"{min_all:.3f}", "min_all_atom_distance_A": f"{min_all_atoms:.3f}", "min_heavy_heavy_distance_A": f"{min_heavy:.3f}", "min_oxygen_oxygen_distance_A": f"{min_oo:.3f}", **packing_meta}


def min_cross_distance(elems: Sequence[str], coords: np.ndarray, split: int) -> float:
    left = [i for i in range(split) if elems[i] != "H"]
    right = [i for i in range(split, len(elems)) if elems[i] != "H"]
    if not left or not right:
        return float("inf")
    return min(float(np.linalg.norm(coords[i] - coords[j])) for i in left for j in right)


def _min_between(left_elems: Sequence[str], left_coords: np.ndarray, right_elems: Sequence[str], right_coords: np.ndarray) -> float:
    return _min_pair_between(left_elems, left_coords, right_elems, right_coords)[0]


def _min_all_cross_distance(left_elems: Sequence[str], left_coords: np.ndarray, right_elems: Sequence[str], right_coords: np.ndarray) -> float:
    if not left_elems or not right_elems:
        return float("inf")
    return min(float(np.linalg.norm(left_coords[i] - right_coords[j])) for i in range(len(left_elems)) for j in range(len(right_elems)))


def _min_non_hh_between(left_elems: Sequence[str], left_coords: np.ndarray, right_elems: Sequence[str], right_coords: np.ndarray) -> float:
    value = float("inf")
    for i, le in enumerate(left_elems):
        for j, re in enumerate(right_elems):
            if le == "H" and re == "H":
                continue
            value = min(value, float(np.linalg.norm(left_coords[i] - right_coords[j])))
    return value


def _min_pair_between(left_elems: Sequence[str], left_coords: np.ndarray, right_elems: Sequence[str], right_coords: np.ndarray) -> Tuple[float, int, int]:
    left = [i for i, elem in enumerate(left_elems) if elem != "H"]
    right = [i for i, elem in enumerate(right_elems) if elem != "H"]
    if not left or not right:
        return float("inf"), 0, 0
    best = (float("inf"), left[0], right[0])
    for i in left:
        for j in right:
            d = float(np.linalg.norm(left_coords[i] - right_coords[j]))
            if d < best[0]:
                best = (d, i, j)
    return best


def _min_all_distance(elems: Sequence[str], coords: np.ndarray) -> float:
    value = float("inf")
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            if elems[i] == "H" and elems[j] == "H":
                continue
            value = min(value, float(np.linalg.norm(coords[i] - coords[j])))
    return value


def _min_all_atom_distance(elems: Sequence[str], coords: np.ndarray) -> float:
    value = float("inf")
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            value = min(value, float(np.linalg.norm(coords[i] - coords[j])))
    return value


def _min_heavy_heavy_distance(elems: Sequence[str], coords: np.ndarray) -> float:
    heavy = [i for i, elem in enumerate(elems) if elem != "H"]
    value = float("inf")
    for left, i in enumerate(heavy):
        for j in heavy[left + 1 :]:
            value = min(value, float(np.linalg.norm(coords[i] - coords[j])))
    return value


def _min_oxygen_oxygen_distance(elems: Sequence[str], coords: np.ndarray) -> float:
    oxygens = [i for i, elem in enumerate(elems) if elem == "O"]
    value = float("inf")
    for left, i in enumerate(oxygens):
        for j in oxygens[left + 1 :]:
            value = min(value, float(np.linalg.norm(coords[i] - coords[j])))
    return value


def _build_base_silica(config: Dict[str, Any], mode: str) -> Tuple[List[str], np.ndarray, Tuple[float, float, float], Dict[str, Any]]:
    ensure_pore_dirs(config)
    build_porems_pores(config, mode)
    rows = available_pore_rows(config)
    if rows.empty:
        raise RuntimeError("No available PoreMS SiO2 pore model for closed small anchors")
    row = rows.iloc[0]
    elems, coords, box = read_xyz_like(Path(row["pore_model_extxyz_path"]))
    center = pore_center(box)
    keep = anchor_indices(coords, center, "concave_pore_wall_patch", float(config["closed_small_anchors"].get("patch_radius_A", 7.5)), int(config["closed_small_anchors"].get("max_silica_atoms_before_capping", 90)))
    local_coords, local_box, cell_meta = rebuild_local_cell(coords[keep], box, config)
    capped_elems, capped_coords, cap_meta = _cap_silica([elems[i] for i in keep], local_coords)
    heavy = np.array([xyz for elem, xyz in zip(capped_elems, capped_coords) if elem in {"Si", "O"}], dtype=float)
    if len(heavy) >= 3:
        centered = capped_coords - capped_coords.mean(axis=0)
        heavy_centered = heavy - heavy.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov(heavy_centered.T))
        normal = vectors[:, int(np.argmin(values))]
        rotated_heavy = heavy_centered @ _rotation_from_to(normal, np.array([0.0, 0.0, 1.0])).T
        if np.percentile(rotated_heavy[:, 2], 90) < abs(np.percentile(rotated_heavy[:, 2], 10)):
            normal = -normal
        rot = _rotation_from_to(normal, np.array([0.0, 0.0, 1.0]))
        capped_coords = centered @ rot.T
        cell_meta["surface_orientation_status"] = "pca_tangent_plane_aligned_to_xy"
    else:
        cell_meta["surface_orientation_status"] = "not_aligned_too_few_heavy_atoms"
    margin = float(config["closed_small_anchors"].get("cell_margin_A", 7.0))
    mins = capped_coords.min(axis=0)
    capped_coords = capped_coords - mins + margin
    extent = capped_coords.max(axis=0) + margin
    local_box = tuple(float(max(extent[i], float(config.get("mlff_assumptions", {}).get("min_cell_length_A", 22.0)))) for i in range(3))
    return capped_elems, capped_coords, local_box, {"source_pore_model_id": row["pore_model_id"], "n_silica_atoms_before_capping": len(keep), **cell_meta, **cap_meta}


def build_closed_small_anchors(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_dirs(config)
    ensure_pore_dirs(config)
    outbase = p(config, "aimd_local_structures_dir")
    outbase.mkdir(parents=True, exist_ok=True)
    template_base = p(config, "exports_dir") / "closed_small_anchor_polymer_templates"
    chain_len = int(config.get("closed_small_anchors", {}).get("emc_chain_length_backbone", 32))
    silica_elems, silica_coords, box, silica_meta = _build_base_silica(config, mode)
    fragments = {
        "PE": _make_emc_fragment(config, "PE", chain_len, 101, template_base / "pe"),
        "PP": _make_emc_fragment(config, "PP", chain_len, 201, template_base / "pp"),
        "PS": _make_ps_fragment("PS_dimer"),
        "PS_trimer": _make_ps_fragment("PS_trimer"),
    }
    specs = [
        ("closed_anchor_0001_silica_only_h_capped", "silica_only_h_capped", []),
        ("closed_anchor_0002_PE_emc_h_capped_silica_contact", "PE_emc_silica_contact", ["PE"]),
        ("closed_anchor_0003_PP_emc_h_capped_silica_contact", "PP_emc_silica_contact", ["PP"]),
        ("closed_anchor_0004_PS_phenyl_h_capped_silica_contact", "PS_phenyl_silica_contact", ["PS"]),
        ("closed_anchor_0005_PS_trimer_h_capped_silica_contact", "PS_trimer_silica_contact", ["PS_trimer"]),
        ("closed_anchor_0006_PE_PP_emc_mixed_h_capped_silica_contact", "PE_PP_mixed_silica_contact", ["PE", "PP"]),
        ("closed_anchor_0007_PE_PS_mixed_h_capped_silica_contact", "PE_PS_mixed_silica_contact", ["PE", "PS"]),
        ("closed_anchor_0008_PP_PS_mixed_h_capped_silica_contact", "PP_PS_mixed_silica_contact", ["PP", "PS"]),
        ("closed_anchor_0009_PE_PP_PS_mixed_h_capped_silica_contact", "PE_PP_PS_mixed_silica_contact", ["PE", "PP", "PS"]),
    ]
    rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []
    for anchor_id, family, names in specs[: int(config.get("closed_small_anchors", {}).get(f"{mode}_max_structures", len(specs)))]:
        frag_list = [(name, fragments[name]) for name in names]
        structure_dir = outbase / anchor_id
        elems, coords, local_box, geom_meta = _place_fragments(silica_elems, silica_coords, frag_list, box, config, structure_dir, seed=9000 + len(rows))
        atom_counts = {element: elems.count(element) for element in sorted(set(elems))}
        metadata = {
            "source_stage": "closed_hydrogen_capped_small_anchor",
            "family": family,
            "crop_family": family,
            "polymer_architecture": "_".join(names) if names else "none",
            "silica_boundary_treatment": "H_and_OH_capped",
            **silica_meta,
            **geom_meta,
        }
        extxyz_path = structure_dir / "structure.extxyz"
        _write_extxyz(extxyz_path, elems, coords, local_box, metadata)
        hard_geometry_ok = (
            silica_meta["undercoordinated_Si_after_capping"] == 0
            and silica_meta["uncapped_O_after_capping"] == 0
            and float(geom_meta["min_all_pair_distance_A"]) >= 0.65
            and float(geom_meta["min_heavy_heavy_distance_A"]) >= 1.35
            and float(geom_meta["min_oxygen_oxygen_distance_A"]) >= 2.05
        )
        status = "available" if hard_geometry_ok else "failed_geometry_or_capping_gate"
        row = {
            "aimd_structure_id": anchor_id,
            "status": status,
            "manual_review_status": "pending_user_review",
            "structure_review_required": True,
            "family": family,
            "crop_family": family,
            "crop_source": "porems_cropped_h_capped_silica_cluster",
            "source_stage": "closed_hydrogen_capped_small_anchor",
            "source_full_pore_id": "",
            "source_snapshot_path": silica_meta["source_pore_model_id"],
            "source_frame_index": "",
            "selection_reason": "small chemically closed AIMD anchor with PoreMS-derived SiO2 and capped boundary",
            "what_local_environment_it_teaches": family,
            "nearest_wall_distance_A": geom_meta["min_polymer_silica_distance_A"],
            "nearest_site_type": "h_capped_silica_boundary_or_silanol",
            "n_silanol_OH_within_5A": "",
            "n_siloxane_O_within_5A": "",
            "surface_class": "h_capped_porems_silica_cluster",
            "polymer_architecture": metadata["polymer_architecture"],
            "boundary_treatment": "hydrogen_oh_capped_cluster_in_periodic_box",
            "cap_atom_count": silica_meta["added_boundary_H"] + 2 * silica_meta["added_boundary_OH"],
            "n_atoms": len(elems),
            "usable_for_cp2k_aimd": status == "available",
            "failure_reason": "" if status == "available" else "capping_or_hard_distance_gate_failed",
            "extxyz_path": str(extxyz_path),
            "local_cell_lx_A": local_box[0],
            "local_cell_ly_A": local_box[1],
            "local_cell_lz_A": local_box[2],
            "coordinate_centering_status": "silica_cluster_centered_polymers_shifted_above_patch",
            "local_cell_origin_shift_xyz": "",
        }
        rows.append(row)
        report_rows.append({**row, **{f"count_{k}": v for k, v in atom_counts.items()}, **geom_meta, **silica_meta})
        review_rows.append(
            {
                "aimd_structure_id": anchor_id,
                "family": family,
                "status": status,
                "manual_review_status": row["manual_review_status"],
                "n_atoms": len(elems),
                "atom_counts": ";".join(f"{element}:{count}" for element, count in sorted(atom_counts.items())),
                "undercoordinated_Si_after_capping": silica_meta["undercoordinated_Si_after_capping"],
                "uncapped_O_after_capping": silica_meta["uncapped_O_after_capping"],
                "min_polymer_silica_distance_A": geom_meta["min_polymer_silica_distance_A"],
                "min_all_pair_distance_A": geom_meta["min_all_pair_distance_A"],
                "min_heavy_heavy_distance_A": geom_meta["min_heavy_heavy_distance_A"],
                "min_oxygen_oxygen_distance_A": geom_meta["min_oxygen_oxygen_distance_A"],
                "packing_method": geom_meta["packing_method"],
                "packmol_status": _packmol_success(geom_meta.get("packmol_log_path", "")),
                "packmol_log_path": geom_meta.get("packmol_log_path", ""),
                "extxyz_path": str(extxyz_path),
                "review_instruction": "inspect_extxyz_before_cp2k_input_generation",
            }
        )
    manifest = write_rows(outbase / "aimd_local_manifest.csv", rows)
    write_rows(p(config, "aimd_structure_manifest"), [{**row, "manifest_kind": "aimd_local"} for row in rows])
    report = p(config, "logs_dir") / "closed_small_anchor_structure_report.csv"
    write_rows(report, report_rows)
    _write_manual_structure_review(config, review_rows)
    (p(config, "logs_dir") / "closed_small_anchor_structure_report.md").write_text(
        "# Closed small-anchor structure report\n\n"
        "Source stage: closed_hydrogen_capped_small_anchor.\n"
        "SiO2 source: PoreMS pore crop; boundary treatment: H/OH capping.\n"
        "PE/PP source: EMC class-II chain templates. PS source: capped styrene oligomer OpenBabel 3D fragment with phenyl side groups, pending EMC topology validation.\n",
        encoding="utf-8",
    )
    return manifest
