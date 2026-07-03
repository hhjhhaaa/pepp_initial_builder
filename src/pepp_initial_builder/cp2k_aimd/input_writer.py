from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml

from pepp_initial_builder.cp2k_aimd.config import ensure_dirs, p, read_rows, write_rows
from pepp_initial_builder.cp2k_aimd.extxyz_writer import AIMD_ELEMENTS, read_xyz_like
from pepp_initial_builder.cp2k_aimd.structure_writer import write_cp2k_structure_inputs


def available_aimd_rows(config: Dict[str, Any]) -> List[Dict[str, str]]:
    selected_short = p(config, "exports_dir") / "selected_short_aimd_manifest.csv"
    selected_rows = [row for row in read_rows(selected_short) if row.get("status") == "selected_for_short_aimd" and row.get("source_sp_status") == "parsed_real_cp2k_output" and row.get("extxyz_path")]
    if selected_rows:
        return [{**row, "requested_label_mode": "short_aimd"} for row in selected_rows]
    master = p(config, "aimd_structure_manifest")
    rows = read_rows(master)
    selected = [row for row in rows if row.get("manifest_kind") == "aimd_local" and row.get("status") == "available" and row.get("extxyz_path")]
    if selected:
        return selected
    local = p(config, "aimd_local_manifest")
    rows = read_rows(local)
    selected = [row for row in rows if row.get("status") == "available" and row.get("extxyz_path")]
    if selected:
        return selected
    structure = p(config, "cp2k_structure_input_manifest")
    rows = read_rows(structure)
    return [{**row, "extxyz_path": row.get("cp2k_xyz_path", "")} for row in rows if "written" in row.get("status", "") and row.get("cp2k_xyz_path")]


def mode_names(row: Dict[str, str]) -> List[str]:
    requested = row.get("requested_label_mode")
    if requested:
        return [requested]
    return ["sp_force"]


def select_short_aimd_from_successful_sp(config: Dict[str, Any]) -> Path:
    ensure_dirs(config)
    parsed = [
        row
        for row in read_rows(p(config, "cp2k_parsed_dir") / "cp2k_parsed_manifest.csv")
        if row.get("status") == "parsed_real_cp2k_output" and row.get("label_mode") == "sp_force"
    ]
    local_by_id = {row.get("aimd_structure_id", ""): row for row in read_rows(p(config, "aimd_local_manifest"))}
    priority = [
        "PP_methyl_silanol_contact",
        "PP_methyl_siloxane_contact",
        "PE_branched_side_chain_silanol_contact",
        "PE_HDPE_CH2_silanol_contact",
        "PE_PP_mixed_near_wall",
        "silica_only_wall_baseline",
    ]
    rows: List[Dict[str, Any]] = []
    for family in priority:
        chosen = [row for row in parsed if row.get("family") == family][:2]
        for row in chosen:
            local = local_by_id.get(row.get("aimd_structure_id", ""), {})
            rows.append(
                {
                    **local,
                    "aimd_structure_id": row.get("aimd_structure_id", ""),
                    "family": row.get("family", ""),
                    "crop_family": local.get("crop_family", row.get("family", "")),
                    "status": "selected_for_short_aimd",
                    "source_sp_status": row.get("status", ""),
                    "source_sp_frames_extxyz_path": row.get("frames_extxyz_path", ""),
                    "source_sp_job_dir": row.get("source_job_dir", row.get("job_dir", "")),
                    "requested_label_mode": "short_aimd",
                    "extxyz_path": local.get("extxyz_path", ""),
                }
            )
    if not rows:
        rows.append({"status": "skipped_no_successful_sp_patch", "source_sp_status": "", "requested_label_mode": "short_aimd", "extxyz_path": ""})
    return write_rows(p(config, "exports_dir") / "selected_short_aimd_manifest.csv", rows)


def cell_inc(box: Tuple[float, float, float]) -> str:
    return f"&CELL\n  ABC {box[0]:.10f} {box[1]:.10f} {box[2]:.10f}\n  PERIODIC XYZ\n&END CELL\n"


def coords_xyz(path: Path, elems: Sequence[str], coords: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(elems)}\nPE/PP-silica CP2K coordinates\n")
        for element, xyz in zip(elems, coords):
            handle.write(f"{element} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")


def coords_inc(path: Path, elems: Sequence[str], coords: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("&COORD\n")
        for element, xyz in zip(elems, coords):
            handle.write(f"  {element} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")
        handle.write("&END COORD\n")


def kind_block(config: Dict[str, Any]) -> str:
    cp2k = config["cp2k"]
    lines = []
    for element in ["C", "H", "O", "Si"]:
        lines += [f"    &KIND {element}", f"      BASIS_SET {cp2k['basis_set']}", f"      POTENTIAL {cp2k['potential']}", "    &END KIND"]
    return "\n".join(lines)


def input_text(aimd_id: str, label_mode: str, box: Tuple[float, float, float], config: Dict[str, Any], mode: str) -> str:
    cp2k = config["cp2k"]
    run_type = "ENERGY_FORCE" if label_mode == "sp_force" else "MD"
    if str(cp2k.get("scf_solver", "DIAGONALIZATION")).upper() == "OT":
        scf_solver_block = f"""      &OT
        PRECONDITIONER {cp2k.get('ot_preconditioner', 'FULL_SINGLE_INVERSE')}
        MINIMIZER {cp2k.get('ot_minimizer', 'DIIS')}
        ENERGY_GAP {float(cp2k.get('ot_energy_gap', 0.001)):.6f}
      &END OT
      &OUTER_SCF
        MAX_SCF {int(cp2k.get('outer_scf_max', 20))}
        EPS_SCF {float(cp2k['eps_scf']):.3e}
      &END OUTER_SCF"""
    else:
        scf_solver_block = f"""      ADDED_MOS {int(cp2k.get('added_mos', 0))}
      &SMEAR
        METHOD FERMI_DIRAC
        ELECTRONIC_TEMPERATURE {float(cp2k.get('electronic_smearing_K', 300))}
      &END SMEAR
      &MIXING
        METHOD {cp2k.get('scf_mixing', 'BROYDEN_MIXING')}
      &END MIXING"""
    motion = """&MOTION
  &PRINT
    &FORCES ON
    &END FORCES
    &STRESS ON
    &END STRESS
  &END PRINT
&END MOTION"""
    if label_mode == "short_aimd":
        md = config["label_modes"]["short_nvt_aimd"]
        steps = int(md.get(f"steps_{mode}", md.get("steps_main", 2000)))
        motion = f"""&MOTION
  &MD
    ENSEMBLE {md.get("ensemble", "NVT")}
    STEPS {steps}
    TIMESTEP {float(md.get("timestep_fs", 0.5))}
    TEMPERATURE {float(cp2k.get("temperatures_K", [523.0])[0])}
  &END MD
  &PRINT
    &TRAJECTORY
      &EACH
        MD {int(md.get("frame_stride", 20))}
      &END EACH
    &END TRAJECTORY
    &FORCES ON
    &END FORCES
    &STRESS ON
    &END STRESS
  &END PRINT
&END MOTION"""
    return f"""&GLOBAL
  PROJECT {aimd_id}_{label_mode}
  RUN_TYPE {run_type}
&END GLOBAL

&FORCE_EVAL
  METHOD QS
  STRESS_TENSOR ANALYTICAL
  &DFT
    BASIS_SET_FILE_NAME {cp2k['basis_set_file']}
    POTENTIAL_FILE_NAME {cp2k['potential_file']}
    CHARGE {int(cp2k.get('charge', 0))}
    MULTIPLICITY {int(cp2k.get('multiplicity', 1))}
    &MGRID
      CUTOFF {float(cp2k['cutoff_Ry']):.1f}
      REL_CUTOFF {float(cp2k['rel_cutoff_Ry']):.1f}
    &END MGRID
    &SCF
      MAX_SCF {int(cp2k['max_scf'])}
      EPS_SCF {float(cp2k['eps_scf']):.3e}
{scf_solver_block}
    &END SCF
    &XC
      &XC_FUNCTIONAL {cp2k['xc_functional']}
      &END XC_FUNCTIONAL
      &VDW_POTENTIAL
        POTENTIAL_TYPE PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE DFTD3(BJ)
          PARAMETER_FILE_NAME {cp2k['dftd3_file']}
          REFERENCE_FUNCTIONAL {cp2k['xc_functional']}
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
    &END XC
  &END DFT
  &SUBSYS
    {cell_inc(box).replace(chr(10), chr(10) + "    ").rstrip()}
    @INCLUDE coords.inc
{kind_block(config)}
  &END SUBSYS
&END FORCE_EVAL

{motion}
"""


def write_cp2k_label_inputs(config: Dict[str, Any], mode: str = "main") -> Path:
    ensure_dirs(config)
    limit = int(config["dataset"].get(f"{mode}_max_structures", config["dataset"].get("main_max_structures", 10**9)))
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(available_aimd_rows(config)[:limit]):
        aimd_id = str(row.get("aimd_structure_id") or row.get("id") or idx)
        family = str(row.get("family", "unknown"))
        src = Path(str(row["extxyz_path"]))
        if not src.exists():
            rows.append({"aimd_structure_id": aimd_id, "family": family, "label_mode": "none", "status": "skipped_missing_structure"})
            continue
        elems, coords, box = read_xyz_like(src)
        if not set(elems).issubset(AIMD_ELEMENTS):
            rows.append({"aimd_structure_id": aimd_id, "family": family, "label_mode": "none", "status": "skipped_forbidden_elements"})
            continue
        for label_mode in mode_names(row):
            job_dir = p(config, "cp2k_jobs_dir") / aimd_id / label_mode
            job_dir.mkdir(parents=True, exist_ok=True)
            project = f"{aimd_id}_{label_mode}"
            coords_xyz(job_dir / "coords.xyz", elems, coords)
            coords_inc(job_dir / "coords.inc", elems, coords)
            (job_dir / "cell.inc").write_text(cell_inc(box), encoding="utf-8")
            (job_dir / "input.inp").write_text(input_text(aimd_id, label_mode, box, config, mode), encoding="utf-8")
            metadata = {
                "aimd_structure_id": aimd_id,
                "family": family,
                "crop_family": row.get("crop_family", family),
                "patch_id": row.get("patch_id") or row.get("source_patch_id", ""),
                "label_mode": label_mode,
                "cp2k_project": project,
                "cp2k_run_type": "ENERGY_FORCE" if label_mode == "sp_force" else "MD",
                "source_extxyz_path": str(src),
                "source_stage": row.get("source_stage", ""),
                "polymer_architecture": row.get("polymer_architecture", ""),
                "pe_variant": row.get("pe_variant", ""),
                "pp_variant": row.get("pp_variant", ""),
                "cp2k_run_performed": False,
            }
            (job_dir / "job_metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
            (job_dir / "run_status.json").write_text(json.dumps({"status": "input_written_not_submitted", "cp2k_out": str(job_dir / "cp2k.out")}, indent=2) + "\n", encoding="utf-8")
            rows.append({**metadata, "status": "cp2k_input_written_no_cp2k_run", "job_dir": str(job_dir), "input_inp_path": str(job_dir / "input.inp"), "coords_xyz_path": str(job_dir / "coords.xyz"), "coords_inc_path": str(job_dir / "coords.inc"), "cell_inc_path": str(job_dir / "cell.inc")})
    if not rows:
        rows.append({"aimd_structure_id": "none", "family": "", "label_mode": "none", "status": "skipped_no_available_aimd_structure"})
    return write_rows(p(config, "cp2k_jobs_dir") / "cp2k_label_input_manifest.csv", rows)
