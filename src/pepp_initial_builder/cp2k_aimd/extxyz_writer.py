from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HARTREE_TO_EV = 27.211386245988
BOHR_TO_A = 0.529177210903
FORCE_AU_TO_EV_A = HARTREE_TO_EV / BOHR_TO_A
AIMD_ELEMENTS = {"C", "H", "O", "Si"}


def read_xyz_like(path: str | Path) -> Tuple[List[str], List[List[float]], Tuple[float, float, float]]:
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
    return elems, coords, box


def read_xyz_frames(path: Path) -> List[Tuple[List[str], List[List[float]]]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    frames: List[Tuple[List[str], List[List[float]]]] = []
    cursor = 0
    while cursor < len(lines):
        try:
            n_atoms = int(lines[cursor].strip())
        except Exception:
            break
        end = cursor + 2 + n_atoms
        if end > len(lines):
            break
        elems: List[str] = []
        values: List[List[float]] = []
        ok = True
        for line in lines[cursor + 2 : end]:
            parts = line.split()
            if len(parts) < 4:
                ok = False
                break
            try:
                elems.append(parts[0])
                values.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
            except Exception:
                ok = False
                break
        if not ok:
            break
        frames.append((elems, values))
        cursor = end
    return frames


def comment_metadata(row: Dict[str, str], frame_index: int, energy_eV: float, box: Tuple[float, float, float], normal_end: bool) -> str:
    meta = {
        "energy": f"{energy_eV:.12f}",
        "energy_unit": "eV",
        "forces_unit": "eV/Angstrom",
        "cp2k_energy_raw_unit": "Hartree",
        "cp2k_force_raw_unit": "Hartree/Bohr",
        "source": "real_cp2k_output",
        "source_stage": row.get("source_stage", ""),
        "crop_family": row.get("crop_family", row.get("family", "")),
        "polymer_architecture": row.get("polymer_architecture", ""),
        "aimd_seed_id": row.get("aimd_structure_id", ""),
        "family": row.get("family", ""),
        "patch_id": row.get("patch_id", ""),
        "label_mode": row.get("label_mode", ""),
        "cp2k_project": row.get("cp2k_project", f"{row.get('aimd_structure_id', '')}_{row.get('label_mode', '')}"),
        "cp2k_run_type": row.get("cp2k_run_type", "ENERGY_FORCE" if row.get("label_mode") == "sp_force" else "MD"),
        "source_job_dir": row.get("job_dir", ""),
        "source_frame_index": str(frame_index),
        "cp2k_normal_end": str(bool(normal_end)).lower(),
    }
    fields = [f'Lattice="{box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]}"', "Properties=species:S:1:pos:R:3:forces:R:3", 'pbc="T T T"']
    fields.extend(f'{key}="{value}"' if "/" in value or " " in value else f"{key}={value}" for key, value in meta.items())
    return " ".join(fields)


def write_extxyz_frames(path: Path, row: Dict[str, str], elems: Sequence[str], positions, forces, energies_eV, box: Tuple[float, float, float], normal_end: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for frame_index, (coords, force_frame, energy_eV) in enumerate(zip(positions, forces, energies_eV)):
            handle.write(f"{len(elems)}\n")
            handle.write(comment_metadata(row, frame_index, energy_eV, box, normal_end) + "\n")
            for element, xyz, force in zip(elems, coords, force_frame):
                handle.write(f"{element} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f} {force[0]:.12f} {force[1]:.12f} {force[2]:.12f}\n")
