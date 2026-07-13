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
from pepp_initial_builder.common.run import run_id
from pepp_initial_builder.pore.config import ensure_pore_dirs, pore_root
from pepp_initial_builder.pore.porems_builder import atoms_from_elements, available_pore_rows, read_xyz_like
from pepp_initial_builder.pore.surface_classifier import estimate_pore_radius_A, inside_pore_fraction, min_cross_distance, pore_center
from pepp_initial_builder.polymer.emc_builder import write_emc_chain_template, write_emc_mixed_template


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


def _composition_label(pe: float, pp: float) -> str:
    return f"PE_HDPE{int(round(pe * 100)):02d}_PP{int(round(pp * 100)):02d}"


def _composition_specs(matrix: Dict[str, Any]) -> List[List[Dict[str, float]]]:
    if "polymer_compositions" in matrix:
        parsed = []
        for item in matrix.get("polymer_compositions", []):
            if isinstance(item, dict):
                specs = [{"component": str(component).upper(), "fraction": float(fraction)} for component, fraction in item.items()]
            else:
                values = list(item)
                if len(values) != 3:
                    raise ValueError("polymer_compositions list entries must be dicts or [PE, PP, PS] fractions")
                specs = [
                    {"component": "PE", "fraction": float(values[0])},
                    {"component": "PP", "fraction": float(values[1])},
                    {"component": "PS", "fraction": float(values[2])},
                ]
            parsed.append([spec for spec in specs if float(spec["fraction"]) > 0.0])
        return parsed
    return [
        [{"component": "PE", "fraction": float(pe)}, {"component": "PP", "fraction": float(pp)}]
        for pe, pp in matrix.get("pe_pp_compositions", [])
    ]


def _composition_label_from_specs(specs: List[Dict[str, float]]) -> str:
    fractions = {spec["component"]: float(spec["fraction"]) for spec in specs}
    pe = fractions.get("PE", 0.0)
    pp = fractions.get("PP", 0.0)
    ps = fractions.get("PS", 0.0)
    if ps <= 0.0:
        return _composition_label(pe, pp)
    return f"PE{int(round(pe * 100)):02d}_PP{int(round(pp * 100)):02d}_PS{int(round(ps * 100)):02d}"



def _component_chain_counts(config: Dict[str, Any], specs: List[Dict[str, float]], loading: str) -> Dict[str, int]:
    total = _loading_multiplier(config, loading)
    fractions = {spec["component"]: max(float(spec["fraction"]), 0.0) for spec in specs}
    norm = sum(fractions.values()) or 1.0
    scaled = {component: value / norm * total for component, value in fractions.items() if value > 0.0}
    counts = {component: max(1, int(np.floor(value))) for component, value in scaled.items()}
    while sum(counts.values()) < total:
        component = max(scaled, key=lambda name: scaled[name] - counts.get(name, 0))
        counts[component] = counts.get(component, 0) + 1
    while sum(counts.values()) > total:
        candidates = [name for name, count in counts.items() if count > 1]
        component = min(candidates or list(counts), key=lambda name: scaled.get(name, 0.0) - counts.get(name, 0))
        counts[component] -= 1
        if counts[component] <= 0:
            counts.pop(component, None)
    return counts


def _loading_multiplier(config: Dict[str, Any], loading: str) -> int:
    values = config.get("full_pore_seed", {}).get("loading_chain_multiplier", {})
    return max(1, int(values.get(str(loading), 1)))


def _infer_template_roles(chain_type: str, elems: List[str], coords: List[List[float]], template_index: int) -> List[Dict[str, Any]]:
    arr = np.array(coords, dtype=float)
    roles: List[Dict[str, Any]] = []
    carbon = [i for i, elem in enumerate(elems) if elem == "C"]
    side_methyl: set[int] = set()
    parent_for: Dict[int, int] = {}
    if chain_type == "PP":
        for idx in carbon:
            c_neighbors = [j for j in carbon if j != idx and float(np.linalg.norm(arr[idx] - arr[j])) <= 1.85]
            h_neighbors = [j for j, elem in enumerate(elems) if elem == "H" and float(np.linalg.norm(arr[idx] - arr[j])) <= 1.25]
            if len(c_neighbors) == 1 and len(h_neighbors) >= 2:
                side_methyl.add(idx)
                parent_for[idx] = c_neighbors[0]
    for local_idx, elem in enumerate(elems):
        attached_methyl = next((c for c in side_methyl if elem == "H" and float(np.linalg.norm(arr[local_idx] - arr[c])) <= 1.25), None)
        if chain_type == "PP" and local_idx in side_methyl:
            atom_role = "PP_side_methyl_C"
            is_side = True
            parent = parent_for.get(local_idx)
        elif chain_type == "PP" and attached_methyl is not None:
            atom_role = "PP_side_methyl_H"
            is_side = True
            parent = parent_for.get(attached_methyl)
        elif chain_type == "PS" and elem == "C":
            atom_role = "PS_C"
            is_side = False
            parent = None
        elif chain_type == "PS" and elem == "H":
            atom_role = "PS_H"
            is_side = False
            parent = None
        elif elem == "C":
            atom_role = f"{chain_type}_backbone_C"
            is_side = False
            parent = None
        elif elem == "H":
            atom_role = f"{chain_type}_backbone_H"
            is_side = False
            parent = None
        else:
            atom_role = f"{chain_type}_{elem}"
            is_side = False
            parent = None
        roles.append(
            {
                "template_index": template_index,
                "template_atom_index": local_idx + 1,
                "element": elem,
                "polymer_type": chain_type,
                "pe_variant": "PE_HDPE_linear" if chain_type == "PE" else "",
                "pp_variant": "PP_atactic_like_v0" if chain_type == "PP" else "",
                "ps_variant": "PS_atactic_phenyl_v0" if chain_type == "PS" else "",
                "atom_role": atom_role,
                "is_side_group": bool(is_side),
                "parent_template_atom_id": "" if parent is None else parent + 1,
                "parent_backbone_atom_id": "",
                "methyl_orientation_class": "emc_generated_atactic_like" if atom_role == "PP_side_methyl_C" else "",
            }
        )
    return roles


def _write_atom_roles(path: Path, n_pore: int, role_templates: List[Dict[str, Any]]) -> None:
    rows = []
    for idx, role in enumerate(role_templates, start=1):
        source_parent = role.get("parent_template_atom_id")
        parent_global = "" if source_parent in {"", None} else n_pore + idx - int(role["template_atom_index"]) + int(source_parent)
        rows.append({**role, "atom_id": n_pore + idx, "parent_backbone_atom_id": parent_global})
    pd.DataFrame(rows).to_csv(path, index=False)


def write_branched_pe_capability_probe(config: Dict[str, Any]) -> Path:
    root = pore_root(config)
    rid = run_id(config)
    out = root / config["paths"].get("aimd_exports_dir", config["paths"].get("exports_dir", "data/exports")) / "branched_pe_capability_manifest.csv"
    log = root / config["paths"].get("logs_dir", "outputs/logs") / "branched_pe_emc_capability_probe.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    branch_cfg = config.get("branched_pe", {})
    for branch_type in branch_cfg.get("branch_types", ["ethyl", "butyl"]):
        rows.append(
            {
                "run_id": rid,
                "pe_variant": branch_cfg.get("variant", "PE_branched_LDPE_like_v1"),
                "branch_type": branch_type,
                "branch_density_per_1000C": "15-25",
                "emc_topology_generated": False,
                "pcff_class2_terms_complete": False,
                "status": "pending_not_generated",
                "failure_reason": "branched_pe_emc_recipe_not_implemented_no_density_substitution_allowed",
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False)
    log.write_text(
        "PE_branched_LDPE_like_v1 probe result: pending_not_generated.\n"
        "Reason: no verified EMC recipe currently generates complete PCFF/Class2 branched PE topology; density lowering is forbidden as LDPE substitute.\n",
        encoding="utf-8",
    )
    return out


def build_full_pore_seed_structures(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_pore_dirs(config)
    if mode == "pilot":
        write_branched_pe_capability_probe(config)
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
            matrix = config["full_pore_seed_matrix"]
            for specs in _composition_specs(matrix):
                fractions = {spec["component"]: float(spec["fraction"]) for spec in specs}
                for loading in matrix.get("polymer_loading_modes", ["low"]):
                    for seed_value in matrix.get("seeds", [1]):
                        if made >= maxn:
                            break
                        seed = int(seed_value)
                        composition = _composition_label_from_specs(specs)
                        sid = f"full_pore_seed_{made + 1:04d}_{composition}_{loading}_seed{seed}"
                        structure_dir = outbase / sid
                        structure_dir.mkdir(parents=True, exist_ok=True)
                        pore_atoms = atoms_from_elements(elems, np.array(coords, dtype=float), 0)
                        polymer_templates: List[Path] = []
                        polymer_elems: List[str] = []
                        role_templates: List[Dict[str, Any]] = []
                        template_index = 0
                        chain_counts = _component_chain_counts(config, specs, str(loading))
                        if len(chain_counts) > 1:
                            mixed_dir = structure_dir / "emc_mixed_params"
                            write_emc_mixed_template(config, chain_counts, int(matrix["chain_lengths_backbone"][0]), seed, mixed_dir)
                        for chain_type, count in chain_counts.items():
                            if chain_type not in {"PE", "PP", "PS"}:
                                raise RuntimeError(f"Unsupported full-pore polymer component: {chain_type}")
                            for copy_idx in range(int(count)):
                                template_index += 1
                                template_dir = structure_dir / "emc_chain_templates" / f"{chain_type.lower()}_{copy_idx + 1:02d}"
                                template = write_emc_chain_template(config, chain_type, int(matrix["chain_lengths_backbone"][0]), seed + template_index, template_dir)
                                polymer_templates.append(Path(template["pdb"]))
                                elems_template, coords_template = pdb_elements_coords_via_obabel(template["pdb"], config.get("tools", {}).get("known_openbabel_executable"))
                                polymer_elems.extend(elems_template)
                                role_templates.extend(_infer_template_roles(chain_type, elems_template, coords_template, template_index))
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
                            _write_atom_roles(structure_dir / "atom_roles.csv", n_pore, role_templates)
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
                            "purpose": "pilot_production_full_pore_relaxed_cp2k_crop_source",
                            "polymer_architecture": composition,
                            "pe_variant": "PE_HDPE_linear" if fractions.get("PE", 0.0) > 0 else "",
                            "pp_variant": "PP_atactic_like_v0" if fractions.get("PP", 0.0) > 0 else "",
                            "ps_variant": "PS_atactic_phenyl_v0" if fractions.get("PS", 0.0) > 0 else "",
                            "composition": composition,
                            "component_chain_counts": chain_counts,
                            "loading_mode": loading,
                            "seed": seed,
                            "packing_method": "packmol_cylindrical_constraint",
                            "packing_status": packing_status,
                            "usable_for_mlff_start": bool(usable),
                            "failure_reason": failure_reason,
                            "polymer_inside_pore_fraction": inside_fraction,
                            "min_polymer_silica_distance_A": min_distance,
                            "packmol_input_path": str(inp),
                            "packmol_log_path": str(packmol_log),
                            "atom_roles_path": str(structure_dir / "atom_roles.csv"),
                            "relaxation": relaxation,
                        }
                        (structure_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
                        rows.append(
                            {
                                "full_pore_seed_id": sid,
                                "status": status,
                                "source_pore_model_id": pore["pore_model_id"],
                                "extxyz_path": extxyz_path,
                                "polymer_architecture": composition,
                                "pe_variant": metadata["pe_variant"],
                                "pp_variant": metadata["pp_variant"],
                                "ps_variant": metadata["ps_variant"],
                                "composition": composition,
                                "component_chain_counts": chain_counts,
                                "loading_mode": loading,
                                "seed": seed,
                                "polymer_inside_pore_fraction": inside_fraction,
                                "min_polymer_silica_distance_A": min_distance,
                                "packing_method": "packmol_cylindrical_constraint",
                                "packing_status": packing_status,
                                "usable_for_mlff_start": bool(usable),
                                "failure_reason": failure_reason,
                                "packmol_input_path": str(inp),
                                "packmol_log_path": str(packmol_log),
                                "atom_roles_path": str(structure_dir / "atom_roles.csv"),
                                "mlff_start_structure_kind": "raw_full_pore_seed",
                            }
                        )
                        made += 1
                    if made >= maxn:
                        break
                if made >= maxn:
                    break
    manifest = outbase / "full_pore_seed_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest
