from __future__ import annotations

import json
import shutil
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
    parsed_sp = [
        row
        for row in read_rows(p(config, "cp2k_parsed_dir") / "cp2k_parsed_manifest.csv")
        if row.get("label_mode") == "sp_force"
    ]
    local_by_id = {row.get("aimd_structure_id", ""): row for row in read_rows(p(config, "aimd_local_manifest"))}
    rows: List[Dict[str, Any]] = []
    successful_sp = [
        row
        for row in parsed_sp
        if row.get("status") == "parsed_real_cp2k_output"
        and row.get("frames_extxyz_path")
        and row.get("aimd_structure_id", "") in local_by_id
    ]
    if parsed_sp and len(successful_sp) == len(parsed_sp):
        for row in successful_sp:
            local = local_by_id[row.get("aimd_structure_id", "")]
            detected_out = row.get("detected_cp2k_out", "")
            source_sp_job_dir = row.get("source_job_dir") or row.get("job_dir") or (
                str(Path(detected_out).parent) if detected_out else ""
            )
            rows.append(
                {
                    **local,
                    "aimd_structure_id": row.get("aimd_structure_id", ""),
                    "family": row.get("family", ""),
                    "crop_family": local.get("crop_family", row.get("family", "")),
                    "status": "selected_for_short_aimd",
                    "source_sp_status": row.get("status", ""),
                    "source_sp_frames_extxyz_path": row.get("frames_extxyz_path", ""),
                    "source_sp_job_dir": source_sp_job_dir,
                    "requested_label_mode": "short_aimd",
                    "extxyz_path": local.get("extxyz_path", ""),
                }
            )
    if not rows:
        rows.append(
            {
                "status": "skipped_no_complete_successful_sp_set" if parsed_sp else "skipped_no_successful_sp_patch",
                "source_sp_status": "",
                "requested_label_mode": "short_aimd",
                "extxyz_path": "",
                "parsed_sp_rows": str(len(parsed_sp)),
                "successful_sp_rows": str(len(successful_sp)),
            }
        )
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


def valence_electron_count(elems: Sequence[str]) -> int:
    valence = {"H": 1, "C": 4, "O": 6, "Si": 4}
    return sum(valence.get(elem, 0) for elem in elems)


def md_thermostat_block(md: Dict[str, Any]) -> str:
    ensemble = str(md.get("ensemble", "NVT")).upper()
    thermostat_type = str(md.get("thermostat_type", "CSVR")).upper()
    if ensemble == "NVE" or thermostat_type in {"", "NONE", "FALSE"}:
        return ""
    timecon = float(md.get("thermostat_timecon_fs", 100.0))
    if thermostat_type == "CSVR":
        return f"""    &THERMOSTAT
      TYPE CSVR
      &CSVR
        TIMECON {timecon:.6f}
      &END CSVR
    &END THERMOSTAT"""
    return f"""    &THERMOSTAT
      TYPE {thermostat_type}
    &END THERMOSTAT"""


def input_text(
    aimd_id: str,
    label_mode: str,
    box: Tuple[float, float, float],
    config: Dict[str, Any],
    mode: str,
    elems: Sequence[str] | None = None,
    wfn_restart: bool = False,
) -> str:
    cp2k = config["cp2k"]
    run_type = "ENERGY_FORCE" if label_mode == "sp_force" else "MD"
    electron_count = valence_electron_count(elems or [])
    odd_electrons = bool(elems) and electron_count % 2 == 1
    multiplicity = 2 if odd_electrons else int(cp2k.get("multiplicity", 1))
    uks_line = "    UKS TRUE\n" if odd_electrons else ""
    wfn_restart_line = "    WFN_RESTART_FILE_NAME restart.wfn\n" if wfn_restart else ""
    scf_guess_line = "      SCF_GUESS RESTART\n" if wfn_restart else ""
    scf_eps = float(cp2k["eps_scf"])
    scf_max = int(cp2k["max_scf"])
    outer_scf_max = int(cp2k.get("outer_scf_max", 20))
    if label_mode == "short_aimd":
        md_cfg = config.get("label_modes", {}).get("short_nvt_aimd", {})
        scf_eps = float(md_cfg.get("eps_scf", scf_eps))
        scf_max = int(md_cfg.get("max_scf", scf_max))
        outer_scf_max = int(md_cfg.get("outer_scf_max", outer_scf_max))
    if str(cp2k.get("scf_solver", "DIAGONALIZATION")).upper() == "OT":
        scf_solver_block = f"""      &OT
        PRECONDITIONER {cp2k.get('ot_preconditioner', 'FULL_SINGLE_INVERSE')}
        MINIMIZER {cp2k.get('ot_minimizer', 'DIIS')}
        STEPSIZE {float(cp2k.get('ot_stepsize', 0.08)):.6f}
        ENERGY_GAP {float(cp2k.get('ot_energy_gap', 0.001)):.6f}
      &END OT
      &OUTER_SCF
        MAX_SCF {outer_scf_max}
        EPS_SCF {scf_eps:.3e}
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
    force_eval_print = """  &PRINT
    &FORCES ON
      NDIGITS 12
    &END FORCES
    &STRESS_TENSOR ON
    &END STRESS_TENSOR
  &END PRINT""" if label_mode == "sp_force" else ""
    motion = ""
    if label_mode == "short_aimd":
        md = config["label_modes"]["short_nvt_aimd"]
        steps = int(md.get(f"steps_{mode}", md.get("steps_main", 2000)))
        frame_stride = int(md.get("frame_stride", 20))
        thermostat = md_thermostat_block(md)
        motion = f"""&MOTION
  &MD
    ENSEMBLE {md.get("ensemble", "NVT")}
    STEPS {steps}
    TIMESTEP {float(md.get("timestep_fs", 0.5))}
    TEMPERATURE {float(cp2k.get("temperatures_K", [523.0])[0])}
{thermostat}
  &END MD
  &PRINT
    &TRAJECTORY
      FILENAME pos
      FORMAT XYZ
      &EACH
        MD {frame_stride}
      &END EACH
    &END TRAJECTORY
    &FORCES ON
      FILENAME frc
      FORMAT XYZ
      UNIT hartree*bohr^-1
      &EACH
        MD {frame_stride}
      &END EACH
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
{force_eval_print}
  &DFT
    BASIS_SET_FILE_NAME {cp2k['basis_set_file']}
    POTENTIAL_FILE_NAME {cp2k['potential_file']}
{wfn_restart_line.rstrip()}
    CHARGE {int(cp2k.get('charge', 0))}
    MULTIPLICITY {multiplicity}
{uks_line.rstrip()}
    &MGRID
      CUTOFF {float(cp2k['cutoff_Ry']):.1f}
      REL_CUTOFF {float(cp2k['rel_cutoff_Ry']):.1f}
    &END MGRID
    &SCF
      MAX_SCF {scf_max if label_mode == "short_aimd" else int(cp2k['max_scf'])}
      EPS_SCF {(scf_eps if label_mode == "short_aimd" else float(cp2k['eps_scf'])):.3e}
{scf_guess_line.rstrip()}
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
            wfn_restart = False
            if label_mode == "short_aimd" and row.get("source_sp_job_dir"):
                source_sp_dir = Path(str(row["source_sp_job_dir"]))
                wfn_candidates = sorted(source_sp_dir.glob("*-RESTART.wfn"), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
                if wfn_candidates:
                    shutil.copyfile(wfn_candidates[0], job_dir / "restart.wfn")
                    wfn_restart = True
            coords_xyz(job_dir / "coords.xyz", elems, coords)
            coords_inc(job_dir / "coords.inc", elems, coords)
            (job_dir / "cell.inc").write_text(cell_inc(box), encoding="utf-8")
            (job_dir / "input.inp").write_text(input_text(aimd_id, label_mode, box, config, mode, elems, wfn_restart), encoding="utf-8")
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
                "wfn_restart_from_sp": wfn_restart,
            }
            (job_dir / "job_metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
            (job_dir / "run_status.json").write_text(json.dumps({"status": "input_written_not_submitted", "cp2k_out": str(job_dir / "cp2k.out")}, indent=2) + "\n", encoding="utf-8")
            rows.append({**metadata, "status": "cp2k_input_written_no_cp2k_run", "job_dir": str(job_dir), "input_inp_path": str(job_dir / "input.inp"), "coords_xyz_path": str(job_dir / "coords.xyz"), "coords_inc_path": str(job_dir / "coords.inc"), "cell_inc_path": str(job_dir / "cell.inc")})
    if not rows:
        rows.append({"aimd_structure_id": "none", "family": "", "label_mode": "none", "status": "skipped_no_available_aimd_structure"})
    return write_rows(p(config, "cp2k_jobs_dir") / "cp2k_label_input_manifest.csv", rows)
