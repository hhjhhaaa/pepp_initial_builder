from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from pepp_initial_builder.cp2k_aimd.config import ensure_dirs, p, write_rows

Atom = Tuple[str, float, float, float]


def _shift(atoms: Sequence[Atom], dx: float, dy: float, dz: float) -> List[Atom]:
    return [(element, x + dx, y + dy, z + dz) for element, x, y, z in atoms]


def _rotate_z(atoms: Sequence[Atom], angle_deg: float) -> List[Atom]:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return [(element, c * x - s * y, s * x + c * y, z) for element, x, y, z in atoms]


def _center_xy(atoms: Sequence[Atom], target_x: float, target_y: float) -> List[Atom]:
    xs = [atom[1] for atom in atoms]
    ys = [atom[2] for atom in atoms]
    return _shift(atoms, target_x - (min(xs) + max(xs)) / 2.0, target_y - (min(ys) + max(ys)) / 2.0, 0.0)


def _center_in_cell(atoms: Sequence[Atom], box: Tuple[float, float, float]) -> Tuple[List[Atom], Tuple[float, float, float]]:
    xs = [atom[1] for atom in atoms]
    ys = [atom[2] for atom in atoms]
    zs = [atom[3] for atom in atoms]
    shift = (
        box[0] / 2.0 - (min(xs) + max(xs)) / 2.0,
        box[1] / 2.0 - (min(ys) + max(ys)) / 2.0,
        box[2] / 2.0 - (min(zs) + max(zs)) / 2.0,
    )
    return _shift(atoms, *shift), shift


def _surface_atoms(surface_type: str) -> Tuple[List[Atom], Dict[str, Any]]:
    spacing = 3.10
    si_xy = [(0.0, 0.0), (spacing, 0.0), (0.0, spacing), (spacing, spacing)]
    atoms: List[Atom] = []
    for x, y in si_xy:
        atoms.append(("Si", x, y, 6.00))
    for x, y in [(spacing / 2, 0.0), (spacing / 2, spacing), (0.0, spacing / 2), (spacing, spacing / 2)]:
        atoms.append(("O", x, y, 6.00))
    for x, y in si_xy:
        atoms.append(("O", x, y, 7.58))
    hydroxyl_sites = si_xy if surface_type != "siloxane_rich" else si_xy[:2]
    for x, y in hydroxyl_sites:
        atoms.append(("H", x, y, 8.55))
    for x, y in si_xy:
        atoms.append(("O", x, y, 4.42))
        atoms.append(("H", x, y, 3.45))
    meta = {
        "surface_type": surface_type,
        "surface_class": "silanol_rich" if surface_type != "siloxane_rich" else "siloxane_rich",
        "n_silanol_OH_within_5A": len(hydroxyl_sites),
        "n_siloxane_O_within_5A": 4,
        "nearest_OH_distance_A": "",
        "nearest_siloxane_O_distance_A": "",
    }
    return atoms, meta


def _pe_fragment(n_backbone: int = 6) -> List[Atom]:
    atoms: List[Atom] = []
    for i in range(n_backbone):
        x = 1.32 * i
        z = 0.35 if i % 2 else -0.35
        atoms.append(("C", x, 0.0, z))
        atoms.append(("H", x, 0.95, z + 0.35))
        atoms.append(("H", x, -0.95, z + 0.35))
        if i == 0:
            atoms.append(("H", x - 0.85, 0.0, z - 0.55))
        if i == n_backbone - 1:
            atoms.append(("H", x + 0.85, 0.0, z - 0.55))
    return atoms


def _pp_fragment(n_repeat: int = 3) -> List[Atom]:
    atoms: List[Atom] = []
    n_backbone = n_repeat * 2
    for i in range(n_backbone):
        x = 1.32 * i
        z = 0.35 if i % 2 else -0.35
        atoms.append(("C", x, 0.0, z))
        atoms.append(("H", x, -0.95, z + 0.35))
        if i % 2 == 0:
            atoms.append(("H", x, 0.95, z + 0.35))
        else:
            atoms.append(("C", x, 1.52, z + 0.10))
            atoms.extend([("H", x - 0.72, 2.02, z + 0.45), ("H", x + 0.72, 2.02, z + 0.45), ("H", x, 1.52, z - 0.95)])
        if i == 0:
            atoms.append(("H", x - 0.85, 0.0, z - 0.55))
        if i == n_backbone - 1:
            atoms.append(("H", x + 0.85, 0.0, z - 0.55))
    return atoms


def _min_distance(a: Sequence[Atom], b: Sequence[Atom]) -> float:
    value = 10**9
    for ea, xa, ya, za in a:
        for eb, xb, yb, zb in b:
            if ea == "H" and eb == "H":
                continue
            dist = math.dist((xa, ya, za), (xb, yb, zb))
            value = min(value, dist)
    return value


def _write_extxyz(path: Path, atoms: Sequence[Atom], box: Tuple[float, float, float], metadata: Dict[str, Any]) -> None:
    fields = [
        f'Lattice="{box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]}"',
        "Properties=species:S:1:pos:R:3",
        'pbc="T T T"',
    ]
    for key, value in metadata.items():
        text = str(value)
        fields.append(f'{key}="{text}"' if " " in text or "/" in text else f"{key}={text}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(atoms)}\n")
        handle.write(" ".join(fields) + "\n")
        for element, x, y, z in atoms:
            handle.write(f"{element} {x:.10f} {y:.10f} {z:.10f}\n")


def _place_fragment(fragment: Sequence[Atom], surface: Sequence[Atom], z_gap: float, angle_deg: float = 0.0, y_shift: float = 0.0) -> List[Atom]:
    placed = _rotate_z(fragment, angle_deg)
    placed = _center_xy(placed, 4.8, 4.8 + y_shift)
    min_z = min(atom[3] for atom in placed)
    top_z = max(atom[3] for atom in surface)
    return _shift(placed, 0.0, 0.0, top_z + z_gap - min_z)


def _anchor_specs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs = config.get("small_anchor_structures", {}).get("structures")
    if specs:
        return list(specs)
    return [
        {"family": "silica_only_silanol_face", "surface_type": "silanol_rich", "polymer": "none", "z_gap_A": 4.0},
        {"family": "PE_short_CH2_silanol_contact", "surface_type": "silanol_rich", "polymer": "PE", "z_gap_A": 2.8},
        {"family": "PE_short_CH2_siloxane_contact", "surface_type": "siloxane_rich", "polymer": "PE", "z_gap_A": 3.0, "angle_deg": 25.0},
        {"family": "PP_methyl_silanol_contact", "surface_type": "silanol_rich", "polymer": "PP", "z_gap_A": 2.9, "angle_deg": 10.0},
        {"family": "PP_methyl_siloxane_contact", "surface_type": "siloxane_rich", "polymer": "PP", "z_gap_A": 3.1, "angle_deg": -20.0},
        {"family": "PP_backbone_CH_silanol_contact", "surface_type": "silanol_rich", "polymer": "PP", "z_gap_A": 3.4, "angle_deg": 90.0},
        {"family": "PE_PP_mixed_silanol_contact", "surface_type": "silanol_rich", "polymer": "PE_PP", "z_gap_A": 3.2},
        {"family": "crowded_polymer_wall_contact", "surface_type": "silanol_rich", "polymer": "PE_PP", "z_gap_A": 2.7, "crowded": True},
    ]


def build_small_anchor_structures(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_dirs(config)
    outbase = p(config, "aimd_local_structures_dir")
    outbase.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    max_structures = int(config.get("small_anchor_structures", {}).get(f"{mode}_max_structures", 12))
    min_heavy_distance = float(config.get("small_anchor_structures", {}).get("min_polymer_surface_distance_A", 1.8))
    box = tuple(float(x) for x in config.get("small_anchor_structures", {}).get("cell_abc_A", [18.0, 18.0, 18.0]))
    for idx, spec in enumerate(_anchor_specs(config)[:max_structures]):
        anchor_id = f"small_anchor_{idx + 1:04d}_{spec['family']}"
        surface, surface_meta = _surface_atoms(str(spec.get("surface_type", "silanol_rich")))
        polymer_kind = str(spec.get("polymer", "none"))
        polymer: List[Atom] = []
        if polymer_kind == "PE":
            polymer = _place_fragment(_pe_fragment(), surface, float(spec.get("z_gap_A", 3.0)), float(spec.get("angle_deg", 0.0)))
        elif polymer_kind == "PP":
            polymer = _place_fragment(_pp_fragment(), surface, float(spec.get("z_gap_A", 3.0)), float(spec.get("angle_deg", 0.0)))
        elif polymer_kind == "PE_PP":
            pe = _place_fragment(_pe_fragment(5), surface, float(spec.get("z_gap_A", 3.0)), -15.0, -1.2)
            pp = _place_fragment(_pp_fragment(2), surface, float(spec.get("z_gap_A", 3.0)) + (0.2 if spec.get("crowded") else 0.8), 18.0, 1.2)
            polymer = pe + pp
        atoms = surface + polymer
        min_dist = _min_distance(surface, polymer) if polymer else ""
        atoms, origin_shift = _center_in_cell(atoms, box)
        usable = True
        failure = ""
        if polymer and float(min_dist) < min_heavy_distance:
            usable = False
            failure = "polymer_surface_distance_below_gate"
        metadata = {
            "source_stage": "designed_small_interface_anchor",
            "anchor_design": "small_crystalline_face_short_repeat_unit",
            "family": spec["family"],
            "crop_family": spec["family"],
            "surface_type": surface_meta["surface_type"],
            "surface_class": surface_meta["surface_class"],
            "polymer_architecture": polymer_kind,
            "nearest_wall_distance_A": f"{min_dist:.3f}" if polymer else "",
            "nearest_site_type": "silanol_OH" if surface_meta["surface_class"] == "silanol_rich" else "siloxane_O",
            "n_silanol_OH_within_5A": surface_meta["n_silanol_OH_within_5A"],
            "n_siloxane_O_within_5A": surface_meta["n_siloxane_O_within_5A"],
        }
        structure_dir = outbase / anchor_id
        extxyz_path = structure_dir / "structure.extxyz"
        _write_extxyz(extxyz_path, atoms, box, metadata)
        rows.append(
            {
                "aimd_structure_id": anchor_id,
                "status": "available" if usable else "rejected_geometry_gate",
                "family": spec["family"],
                "crop_family": spec["family"],
                "crop_source": "designed_small_anchor",
                "source_stage": "designed_small_interface_anchor",
                "source_full_pore_id": "",
                "source_snapshot_path": "",
                "source_frame_index": "",
                "selection_reason": spec["family"],
                "what_local_environment_it_teaches": spec.get("teaches", f"Small {polymer_kind}/silica anchor for local C/H/O/Si interaction."),
                "nearest_wall_distance_A": f"{min_dist:.3f}" if polymer else "",
                "nearest_site_type": metadata["nearest_site_type"],
                "n_silanol_OH_within_5A": surface_meta["n_silanol_OH_within_5A"],
                "n_siloxane_O_within_5A": surface_meta["n_siloxane_O_within_5A"],
                "surface_class": surface_meta["surface_class"],
                "polymer_architecture": polymer_kind,
                "boundary_treatment": "small_periodic_vacuum_anchor",
                "cap_atom_count": 0,
                "n_atoms": len(atoms),
                "usable_for_cp2k_aimd": usable,
                "failure_reason": failure,
                "extxyz_path": str(extxyz_path),
                "local_cell_lx_A": box[0],
                "local_cell_ly_A": box[1],
                "local_cell_lz_A": box[2],
                "coordinate_centering_status": "centered_in_local_cell",
                "local_cell_origin_shift_xyz": ",".join(f"{value:.6f}" for value in origin_shift),
            }
        )
    return write_rows(outbase / "aimd_local_manifest.csv", rows)
