from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml

from pepp_initial_builder.cp2k_aimd.config import ensure_dirs, p, read_rows, write_rows
from pepp_initial_builder.cp2k_aimd.extxyz_writer import AIMD_ELEMENTS, read_xyz_like
from pepp_initial_builder.cp2k_aimd.structure_writer import write_cp2k_structure_inputs


def available_aimd_rows(config: Dict[str, Any]) -> List[Dict[str, str]]:
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


def mode_names(family: str) -> List[str]:
    if "silica_patch_only" in family:
        return ["short_aimd"]
    if "compressed_polymer_wall_contact" in family or "pe_pp_crowded_near_wall" in family:
        return ["sp_force"]
    if "distorted_surface_oh_under_polymer" in family:
        return ["sp_force", "short_aimd"]
    return ["sp_force"]


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
      &SMEAR
        METHOD FERMI_DIRAC
        ELECTRONIC_TEMPERATURE {float(cp2k.get('electronic_smearing_K', 300))}
      &END SMEAR
      &MIXING
        METHOD {cp2k.get('scf_mixing', 'BROYDEN')}
      &END MIXING
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
        for label_mode in mode_names(family):
            job_dir = p(config, "cp2k_jobs_dir") / aimd_id / label_mode
            job_dir.mkdir(parents=True, exist_ok=True)
            project = f"{aimd_id}_{label_mode}"
            coords_xyz(job_dir / "coords.xyz", elems, coords)
            coords_inc(job_dir / "coords.inc", elems, coords)
            (job_dir / "cell.inc").write_text(cell_inc(box), encoding="utf-8")
            (job_dir / "input.inp").write_text(input_text(aimd_id, label_mode, box, config, mode), encoding="utf-8")
            (job_dir / "job_metadata.yaml").write_text(yaml.safe_dump({"aimd_structure_id": aimd_id, "family": family, "patch_id": row.get("patch_id") or row.get("source_patch_id", ""), "label_mode": label_mode, "cp2k_project": project, "cp2k_run_type": "ENERGY_FORCE" if label_mode == "sp_force" else "MD", "source_extxyz_path": str(src), "cp2k_run_performed": False}, sort_keys=False), encoding="utf-8")
            (job_dir / "run_status.json").write_text(json.dumps({"status": "input_written_not_submitted", "cp2k_out": str(job_dir / "cp2k.out")}, indent=2) + "\n", encoding="utf-8")
            rows.append({"aimd_structure_id": aimd_id, "family": family, "patch_id": row.get("patch_id") or row.get("source_patch_id", ""), "label_mode": label_mode, "cp2k_project": project, "cp2k_run_type": "ENERGY_FORCE" if label_mode == "sp_force" else "MD", "status": "cp2k_input_written_no_cp2k_run", "job_dir": str(job_dir), "input_inp_path": str(job_dir / "input.inp"), "coords_xyz_path": str(job_dir / "coords.xyz"), "coords_inc_path": str(job_dir / "coords.inc"), "cell_inc_path": str(job_dir / "cell.inc"), "source_extxyz_path": str(src)})
    if not rows:
        rows.append({"aimd_structure_id": "none", "family": "", "label_mode": "none", "status": "skipped_no_available_aimd_structure"})
    return write_rows(p(config, "cp2k_jobs_dir") / "cp2k_label_input_manifest.csv", rows)
