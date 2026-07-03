from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from pepp_initial_builder.cp2k_aimd.config import ensure_dirs, p
from pepp_initial_builder.cp2k_aimd.crop_builder import build_crop
from pepp_initial_builder.cp2k_aimd.full_pore_snapshot_reader import read_full_pore_snapshot_sources, write_source_audit
from pepp_initial_builder.cp2k_aimd.local_environment_selector import select_local_environments
from pepp_initial_builder.pore.porems_builder import read_xyz_like


Candidate = Tuple[Dict[str, str], Any, List[str], Any, tuple[float, float, float]]


def select_balanced_candidates(candidates: List[Candidate], config: Dict[str, Any], max_structures: int) -> List[Candidate]:
    quotas = config.get("cp2k_crop", {}).get("balanced_family_min_counts", {})
    selected: List[Candidate] = []
    used: set[int] = set()

    def take(candidate_idx: int) -> None:
        selected.append(candidates[candidate_idx])
        used.add(candidate_idx)

    for family, min_count in quotas.items():
        count = 0
        for idx, (_source, env, _elems, _coords, _box) in enumerate(candidates):
            if idx in used or env.selection_reason != family:
                continue
            take(idx)
            count += 1
            if len(selected) >= max_structures or count >= int(min_count):
                break
        if len(selected) >= max_structures:
            return selected
    for idx in range(len(candidates)):
        if idx not in used:
            take(idx)
            if len(selected) >= max_structures:
                break
    return selected


def build_aimd_local_structures(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_dirs(config)
    outbase = p(config, "aimd_local_structures_dir")
    outbase.mkdir(parents=True, exist_ok=True)
    sources = read_full_pore_snapshot_sources(config)
    write_source_audit(config, sources)
    max_structures = int(config.get("aimd_local_matrix", {}).get(f"{mode}_max_structures", 12))
    rows: List[Dict[str, Any]] = []
    if not sources:
        rows.append(
            {
                "aimd_structure_id": "no_full_pore_snapshot_source",
                "status": "skipped_no_full_pore_snapshot_source",
                "crop_source": "full_pore_snapshot",
                "source_stage": "",
                "source_full_pore_id": "",
                "source_snapshot_path": "",
                "extxyz_path": "",
                "usable_for_cp2k_aimd": False,
                "failure_reason": "no_full_pore_seed_or_snapshot_manifest",
            }
        )
    candidates: List[Candidate] = []
    for source in sources:
        source_path = Path(source["source_snapshot_path"])
        if not source_path.exists():
            rows.append(
                {
                    "aimd_structure_id": f"missing_source_{len(rows) + 1:04d}",
                    "status": "skipped_missing_full_pore_snapshot",
                    "crop_source": "full_pore_snapshot",
                    "source_stage": source["source_stage"],
                    "source_full_pore_id": source["source_full_pore_id"],
                    "source_snapshot_path": str(source_path),
                    "extxyz_path": "",
                    "usable_for_cp2k_aimd": False,
                    "failure_reason": "source_snapshot_path_missing",
                }
            )
            continue
        elems, coords, box = read_xyz_like(source_path)
        environments = select_local_environments(elems, coords, source, config, max_structures)
        families_done = {env.selection_reason for env in environments}
        requested = set(config.get("aimd_local_matrix", {}).get("families", []))
        if source.get("pe_variant") != "PE_branched_LDPE_like_v1" and "PE_branched_side_chain_silanol_contact" in requested:
            rows.append(
                {
                    "aimd_structure_id": f"skipped_branched_pe_{len(rows) + 1:04d}",
                    "status": "skipped_no_branched_pe_source",
                    "family": "PE_branched_side_chain_silanol_contact",
                    "crop_source": "full_pore_snapshot",
                    "source_stage": source["source_stage"],
                    "source_full_pore_id": source["source_full_pore_id"],
                    "source_snapshot_path": source["source_snapshot_path"],
                    "extxyz_path": "",
                    "usable_for_cp2k_aimd": False,
                    "failure_reason": "PE_branched_LDPE_like_v1_pending_not_generated",
                }
            )
        atom_roles = str(source.get("atom_roles_path", ""))
        if any(f.startswith("PP_methyl") for f in requested) and "PP_methyl_silanol_contact" not in families_done and "PP_methyl_siloxane_contact" not in families_done:
            rows.append(
                {
                    "aimd_structure_id": f"skipped_pp_methyl_{len(rows) + 1:04d}",
                    "status": "skipped_missing_atom_role_metadata" if not atom_roles else "skipped_no_pp_methyl_wall_candidate",
                    "family": "PP_methyl_*",
                    "crop_source": "full_pore_snapshot",
                    "source_stage": source["source_stage"],
                    "source_full_pore_id": source["source_full_pore_id"],
                    "source_snapshot_path": source["source_snapshot_path"],
                    "extxyz_path": "",
                    "usable_for_cp2k_aimd": False,
                    "failure_reason": "PP methyl selection requires atom_role=PP_side_methyl_C,is_side_group=true,parent_backbone_atom_id",
                }
            )
        for env in environments:
            candidates.append((source, env, elems, coords, box))
    made = 0
    for source, env, elems, coords, box in select_balanced_candidates(candidates, config, max_structures):
        aimd_seed_id = f"aimd_{made + 1:04d}_{env.selection_reason}"
        structure_dir = outbase / aimd_seed_id
        meta = build_crop(source, env, elems, coords, box, structure_dir, aimd_seed_id, config, mode)
        rows.append(
            {
                "aimd_structure_id": aimd_seed_id,
                "status": meta["status"],
                "family": meta["selection_reason"],
                "crop_source": meta["crop_source"],
                "source_stage": meta["source_stage"],
                "source_full_pore_id": meta["source_full_pore_id"],
                "source_snapshot_path": meta["source_snapshot_path"],
                "source_frame_index": meta["source_frame_index"],
                "selection_reason": meta["selection_reason"],
                "what_local_environment_it_teaches": meta["what_local_environment_it_teaches"],
                "center_atom_id": meta["center_atom_id"],
                "center_atom_element": meta["center_atom_element"],
                "nearest_wall_distance_A": meta["nearest_wall_distance_A"],
                "nearest_site_type": meta.get("nearest_site_type", ""),
                "n_silanol_OH_within_5A": meta.get("n_silanol_OH_within_5A", ""),
                "n_siloxane_O_within_5A": meta.get("n_siloxane_O_within_5A", ""),
                "nearest_OH_distance_A": meta.get("nearest_OH_distance_A", ""),
                "nearest_siloxane_O_distance_A": meta.get("nearest_siloxane_O_distance_A", ""),
                "local_polymer_density": meta["local_polymer_density"],
                "local_PE_fraction": meta["local_PE_fraction"],
                "local_PP_fraction": meta["local_PP_fraction"],
                "surface_class": meta["surface_class"],
                "crop_family": meta.get("crop_family", meta["selection_reason"]),
                "polymer_architecture": meta.get("polymer_architecture", ""),
                "pe_variant": meta.get("pe_variant", ""),
                "pp_variant": meta.get("pp_variant", ""),
                "center_atom_role": meta.get("center_atom_role", ""),
                "parent_backbone_atom_id": meta.get("parent_backbone_atom_id", ""),
                "nearest_silica_atom_id": meta.get("nearest_silica_atom_id", ""),
                "nearest_silica_element": meta.get("nearest_silica_element", ""),
                "methyl_wall_alignment": meta.get("methyl_wall_alignment", ""),
                "methyl_orientation_class": meta.get("methyl_orientation_class", ""),
                "n_PP_side_methyl_within_5A": meta.get("n_PP_side_methyl_within_5A", ""),
                "n_PE_backbone_C_within_5A": meta.get("n_PE_backbone_C_within_5A", ""),
                "boundary_treatment": meta["boundary_treatment"],
                "cap_atom_count": meta["cap_atom_count"],
                "n_atoms": meta["n_atoms"],
                "usable_for_cp2k_aimd": meta["usable_for_cp2k_aimd"],
                "failure_reason": meta["failure_reason"],
                "extxyz_path": meta["input_extxyz_path"],
                "pdb_path": meta["input_pdb_path"],
                "local_cell_lx_A": meta["local_cell_lx_A"],
                "local_cell_ly_A": meta["local_cell_ly_A"],
                "local_cell_lz_A": meta["local_cell_lz_A"],
                "coordinate_centering_status": meta["coordinate_centering_status"],
                "local_cell_origin_shift_xyz": meta["local_cell_origin_shift_xyz"],
            }
        )
        made += 1
    manifest = outbase / "aimd_local_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


build_seed_structures = build_aimd_local_structures
