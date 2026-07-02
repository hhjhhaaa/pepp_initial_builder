from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from pepp_initial_builder.common.io import write_pdb, write_xyz
from pepp_initial_builder.common.openbabel import pdb_elements_coords_via_obabel
from pepp_initial_builder.pore.config import ensure_pore_dirs, pore_root
from pepp_initial_builder.pore.porems_builder import atoms_from_elements, available_pore_rows, read_xyz_like
from pepp_initial_builder.pore.surface_classifier import estimate_pore_radius_A, inside_pore_fraction, min_cross_distance, pore_center
from pepp_initial_builder.polymer.emc_builder import write_emc_chain_template


def _packmol_executable(config: Dict[str, Any]) -> str | None:
    tools = config.get("tools", {})
    hinted = tools.get("packmol_path_hint") or tools.get("known_packmol_executable")
    if hinted:
        path = Path(str(hinted)).expanduser()
        if path.exists():
            return str(path)
    return shutil.which("packmol")


def _write_packmol_full_pore_input(
    structure_dir: Path,
    pore_atoms,
    polymer_templates: List[Path],
    box,
    pore_radius: float,
    wall_buffer: float,
    end_buffer: float,
    seed: int,
    config: Dict[str, Any],
) -> Path:
    packmol_dir = structure_dir / "packmol"
    template_dir = packmol_dir / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    pore_pdb = template_dir / "fixed_silica_pore.pdb"
    write_pdb(pore_pdb, pore_atoms, box)
    center = pore_center(box)
    allowed_radius = max(pore_radius - wall_buffer, 0.5)
    zmin = end_buffer
    length = max(float(box[2]) - 2.0 * end_buffer, 1.0)
    pack_cfg = config.get("packing", {})
    lines = [
        f"tolerance {float(pack_cfg.get('tolerance_A', pack_cfg.get('min_heavy_atom_distance_A', 2.0))):.6f}",
        "filetype pdb",
        f"output {packmol_dir / 'packed_full_pore.pdb'}",
        f"seed {int(seed)}",
        f"maxit {int(pack_cfg.get('maxit', 2000))}",
        "",
        f"structure {pore_pdb}",
        "  number 1",
        "  fixed 0.0 0.0 0.0 0.0 0.0 0.0",
        "end structure",
        "",
    ]
    for idx, template in enumerate(polymer_templates, start=1):
        lines += [
            f"structure {template}",
            "  number 1",
            f"  inside cylinder {center[0]:.6f} {center[1]:.6f} {zmin:.6f} 0.0 0.0 1.0 {allowed_radius:.6f} {length:.6f}",
            "end structure",
            "",
        ]
    inp = packmol_dir / "packmol.inp"
    inp.write_text("\n".join(lines), encoding="utf-8")
    return inp


def _run_packmol(inp: Path, exe: str, timeout: int) -> tuple[bool, str]:
    log = inp.parent / "packmol.log"
    try:
        with inp.open("rb") as fin, log.open("wb") as fout:
            proc = subprocess.run([exe], stdin=fin, stdout=fout, stderr=subprocess.STDOUT, cwd=inp.parent, timeout=timeout, check=False)
        text = log.read_text(encoding="utf-8", errors="ignore")
        return proc.returncode == 0 and "Success!" in text and (inp.parent / "packed_full_pore.pdb").exists(), ""
    except Exception as exc:
        prior = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
        log.write_text(prior + f"\nPackmol execution failed: {exc}\n", encoding="utf-8")
        return False, str(exc)


def build_full_pore_seed_structures(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_pore_dirs(config)
    pores = available_pore_rows(config)
    outbase = pore_root(config) / config["paths"]["full_pore_seed_structures_dir"]
    maxn = int(config["full_pore_seed_matrix"][f"{mode}_max_systems"])
    packmol = _packmol_executable(config)
    rows: List[Dict[str, Any]] = []
    if pores.empty:
        rows.append({"full_pore_seed_id": "no_available_pore_model", "status": "skipped_no_available_pore_model", "source_pore_model_id": "", "extxyz_path": ""})
    else:
        made = 0
        for _, pore in pores.iterrows():
            elems, coords, box = read_xyz_like(Path(pore["pore_model_extxyz_path"]))
            seed_cfg = config.get("full_pore_seed", {})
            wall_buffer = float(seed_cfg.get("wall_buffer_A", 3.0))
            end_buffer = float(seed_cfg.get("end_buffer_A", 3.0))
            min_inside = float(seed_cfg.get("min_polymer_inside_pore_fraction", 0.95))
            min_silica = float(seed_cfg.get("min_polymer_silica_distance_A", 1.6))
            pore_radius = estimate_pore_radius_A(pore, box, coords)
            for pe, pp in config["full_pore_seed_matrix"]["pe_pp_compositions"]:
                if made >= maxn:
                    break
                seed = int(config["full_pore_seed_matrix"]["seeds"][0])
                sid = f"full_pore_seed_{made + 1:04d}_PE{int(pe * 100):02d}_PP{int(pp * 100):02d}_seed{seed}"
                structure_dir = outbase / sid
                structure_dir.mkdir(parents=True, exist_ok=True)
                pore_atoms = atoms_from_elements(elems, np.array(coords, dtype=float), 0)
                polymer_templates: List[Path] = []
                polymer_elems: List[str] = []
                for chain_type, present in [("PE", pe > 0), ("PP", pp > 0)]:
                    if not present:
                        continue
                    template_dir = structure_dir / "emc_chain_templates" / chain_type.lower()
                    template = write_emc_chain_template(config, chain_type, int(config["full_pore_seed_matrix"]["chain_lengths_backbone"][0]), seed, template_dir)
                    polymer_templates.append(Path(template["pdb"]))
                    elems_template, _coords_template = pdb_elements_coords_via_obabel(template["pdb"], config.get("tools", {}).get("known_openbabel_executable"))
                    polymer_elems.extend(elems_template)
                inp = _write_packmol_full_pore_input(structure_dir, pore_atoms, polymer_templates, box, pore_radius, wall_buffer, end_buffer, seed, config)
                packmol_log = inp.parent / "packmol.log"
                packmol_ok = False
                packmol_failure = "packmol_not_found"
                if packmol:
                    packmol_ok, packmol_failure = _run_packmol(inp, packmol, int(config.get("packing", {}).get("timeout_seconds", 300)))
                    if packmol_ok:
                        packmol_failure = ""
                if packmol_ok:
                    try:
                        packed_elems, packed_coords = pdb_elements_coords_via_obabel(inp.parent / "packed_full_pore.pdb", config.get("tools", {}).get("known_openbabel_executable"))
                        expected_atoms = len(elems) + len(polymer_elems)
                        if len(packed_coords) != expected_atoms:
                            raise ValueError(f"converted coordinate count {len(packed_coords)} != expected {expected_atoms}")
                    except Exception as exc:
                        packed_coords = []
                        packed_elems = []
                        packmol_ok = False
                        packmol_failure = f"openbabel_conversion_failed: {exc}"
                if packmol_ok:
                    n_pore = len(elems)
                    all_elems = packed_elems
                    combined_coords = np.array(packed_coords[: len(all_elems)], dtype=float)
                    placed_polymer = combined_coords[n_pore:]
                    inside_fraction = inside_pore_fraction(placed_polymer, box, pore_radius, wall_buffer, end_buffer)
                    min_distance = min_cross_distance(all_elems, combined_coords, n_pore)
                    usable = inside_fraction >= min_inside and min_distance >= min_silica
                    packing_status = "packed_inside_pore" if usable else "polymer_not_inside_pore_or_overlap"
                    status = "available" if usable else "unavailable_packmol_threshold_failed"
                    atoms = atoms_from_elements(all_elems, combined_coords, 0)
                    extxyz_path = str(structure_dir / "seed.extxyz")
                    write_xyz(extxyz_path, atoms, box, ext=True)
                    write_pdb(structure_dir / "seed.pdb", atoms, box)
                else:
                    inside_fraction = 0.0
                    min_distance = 0.0
                    usable = False
                    status = "unavailable_packmol_failed"
                    packing_status = packmol_failure
                    extxyz_path = ""
                relaxation = {"lammps_relax_performed": False, "relax_is_training_data": False, "relax_is_production_md": False, "mlff_start_structure_kind": "raw_full_pore_seed", "mlff_start_extxyz_path": str(structure_dir / "seed.extxyz")}
                failure_reason = "" if usable else ("polymer_not_inside_pore_or_overlap" if packmol_ok else "packmol_failed_or_missing")
                metadata = {
                    "full_pore_seed_id": sid,
                    "status": status,
                    "source_pore_model_id": pore["pore_model_id"],
                    "purpose": "mlff_exploration_seed_only_no_mlff_run",
                    "packing_method": "packmol_cylindrical_constraint",
                    "packing_status": packing_status,
                    "usable_for_mlff_start": bool(usable),
                    "failure_reason": failure_reason,
                    "polymer_inside_pore_fraction": inside_fraction,
                    "min_polymer_silica_distance_A": min_distance,
                    "packmol_input_path": str(inp),
                    "packmol_log_path": str(packmol_log),
                    "relaxation": relaxation,
                }
                (structure_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
                rows.append(
                    {
                        "full_pore_seed_id": sid,
                        "status": status,
                        "source_pore_model_id": pore["pore_model_id"],
                        "extxyz_path": extxyz_path,
                        "polymer_inside_pore_fraction": inside_fraction,
                        "min_polymer_silica_distance_A": min_distance,
                        "packing_method": "packmol_cylindrical_constraint",
                        "packing_status": packing_status,
                        "usable_for_mlff_start": bool(usable),
                        "failure_reason": failure_reason,
                        "packmol_input_path": str(inp),
                        "packmol_log_path": str(packmol_log),
                        "mlff_start_structure_kind": "raw_full_pore_seed",
                    }
                )
                made += 1
    manifest = outbase / "full_pore_seed_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest

