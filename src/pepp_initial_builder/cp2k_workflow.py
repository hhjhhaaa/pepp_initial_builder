from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml

HARTREE_TO_EV = 27.211386245988
BOHR_TO_A = 0.529177210903
FORCE_AU_TO_EV_A = HARTREE_TO_EV / BOHR_TO_A
AIMD_ELEMENTS = {"C", "H", "O", "Si"}


def load_cp2k_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def root(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["root"])


def p(config: Dict[str, Any], key: str) -> Path:
    value = Path(config["paths"][key])
    return value if value.is_absolute() else root(config) / value


def ensure_dirs(config: Dict[str, Any]) -> None:
    for key in ["cp2k_jobs_dir", "cp2k_parsed_dir", "aimd_dataset_dir", "exports_dir", "logs_dir", "jobs_dir"]:
        p(config, key).mkdir(parents=True, exist_ok=True)


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (value.replace("\\", "/") if isinstance(value, str) else value) for key, value in row.items()})
    return path


def _read_xyz_like(path: str | Path) -> Tuple[List[str], List[List[float]], Tuple[float, float, float]]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = int(lines[0].strip())
    comment = lines[1] if len(lines) > 1 else ""
    box = (30.0, 30.0, 30.0)
    if "Lattice=" in comment:
        part = comment.split('Lattice="', 1)[1].split('"', 1)[0].split()
        if len(part) >= 9:
            box = (float(part[0]), float(part[4]), float(part[8]))
    elems: List[str] = []
    coords: List[List[float]] = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        elems.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elems, coords, box


def _available_aimd_rows(config: Dict[str, Any]) -> List[Dict[str, str]]:
    master = p(config, "aimd_structure_manifest")
    rows = _read_rows(master)
    selected = [r for r in rows if r.get("manifest_kind") == "aimd_local" and r.get("status") == "available" and r.get("extxyz_path")]
    if selected:
        return selected
    local = p(config, "aimd_local_manifest")
    rows = _read_rows(local)
    selected = [r for r in rows if r.get("status") == "available" and r.get("extxyz_path")]
    if selected:
        return selected
    structure = p(config, "cp2k_structure_input_manifest")
    rows = _read_rows(structure)
    return [
        {**r, "extxyz_path": r.get("cp2k_xyz_path", "")}
        for r in rows
        if "written" in r.get("status", "") and r.get("cp2k_xyz_path")
    ]


def _mode_names(family: str) -> List[str]:
    if "silica_patch_only" in family:
        return ["short_aimd"]
    if "compressed_polymer_wall_contact" in family or "pe_pp_crowded_near_wall" in family:
        return ["sp_force"]
    if "distorted_surface_oh_under_polymer" in family:
        return ["sp_force", "short_aimd"]
    return ["sp_force"]


def _cell_inc(box: Tuple[float, float, float]) -> str:
    return f"&CELL\n  ABC {box[0]:.10f} {box[1]:.10f} {box[2]:.10f}\n  PERIODIC XYZ\n&END CELL\n"


def _coords_xyz(path: Path, elems: Sequence[str], coords: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(elems)}\nPE/PP-silica CP2K coordinates\n")
        for el, xyz in zip(elems, coords):
            f.write(f"{el} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")


def _coords_inc(path: Path, elems: Sequence[str], coords: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("&COORD\n")
        for el, xyz in zip(elems, coords):
            f.write(f"  {el} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n")
        f.write("&END COORD\n")


def _kind_block(elems: Sequence[str], config: Dict[str, Any]) -> str:
    cp2k = config["cp2k"]
    lines: List[str] = []
    for element in ["C", "H", "O", "Si"]:
        lines += [f"    &KIND {element}", f"      BASIS_SET {cp2k['basis_set']}", f"      POTENTIAL {cp2k['potential']}", "    &END KIND"]
    return "\n".join(lines)


def _input_text(aimd_id: str, label_mode: str, elems: Sequence[str], box: Tuple[float, float, float], config: Dict[str, Any], mode: str) -> str:
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
    {_cell_inc(box).replace(chr(10), chr(10) + "    ").rstrip()}
    @INCLUDE coords.inc
{_kind_block(elems, config)}
  &END SUBSYS
&END FORCE_EVAL

{motion}
"""


def write_cp2k_label_inputs(config: Dict[str, Any], mode: str = "main") -> Path:
    ensure_dirs(config)
    limit = int(config["dataset"].get(f"{mode}_max_structures", config["dataset"].get("main_max_structures", 10**9)))
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(_available_aimd_rows(config)[:limit]):
        aimd_id = str(row.get("aimd_structure_id") or row.get("id") or idx)
        family = str(row.get("family", "unknown"))
        src = Path(str(row["extxyz_path"]))
        if not src.exists():
            rows.append({"aimd_structure_id": aimd_id, "family": family, "label_mode": "none", "status": "skipped_missing_structure"})
            continue
        elems, coords, box = _read_xyz_like(src)
        if not set(elems).issubset(AIMD_ELEMENTS):
            rows.append({"aimd_structure_id": aimd_id, "family": family, "label_mode": "none", "status": "skipped_forbidden_elements"})
            continue
        for label_mode in _mode_names(family):
            job_dir = p(config, "cp2k_jobs_dir") / aimd_id / label_mode
            job_dir.mkdir(parents=True, exist_ok=True)
            project = f"{aimd_id}_{label_mode}"
            _coords_xyz(job_dir / "coords.xyz", elems, coords)
            _coords_inc(job_dir / "coords.inc", elems, coords)
            (job_dir / "cell.inc").write_text(_cell_inc(box), encoding="utf-8")
            (job_dir / "input.inp").write_text(_input_text(aimd_id, label_mode, elems, box, config, mode), encoding="utf-8")
            (job_dir / "job_metadata.yaml").write_text(
                yaml.safe_dump(
                    {
                        "aimd_structure_id": aimd_id,
                        "family": family,
                        "patch_id": row.get("patch_id") or row.get("source_patch_id", ""),
                        "label_mode": label_mode,
                        "cp2k_project": project,
                        "cp2k_run_type": "ENERGY_FORCE" if label_mode == "sp_force" else "MD",
                        "source_extxyz_path": str(src),
                        "cp2k_run_performed": False,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            (job_dir / "run_status.json").write_text(json.dumps({"status": "input_written_not_submitted", "cp2k_out": str(job_dir / "cp2k.out")}, indent=2) + "\n", encoding="utf-8")
            rows.append({"aimd_structure_id": aimd_id, "family": family, "patch_id": row.get("patch_id") or row.get("source_patch_id", ""), "label_mode": label_mode, "cp2k_project": project, "cp2k_run_type": "ENERGY_FORCE" if label_mode == "sp_force" else "MD", "status": "cp2k_input_written_no_cp2k_run", "job_dir": str(job_dir), "input_inp_path": str(job_dir / "input.inp"), "coords_xyz_path": str(job_dir / "coords.xyz"), "coords_inc_path": str(job_dir / "coords.inc"), "cell_inc_path": str(job_dir / "cell.inc"), "source_extxyz_path": str(src)})
    if not rows:
        rows.append({"aimd_structure_id": "none", "family": "", "label_mode": "none", "status": "skipped_no_available_aimd_structure"})
    return _write_rows(p(config, "cp2k_jobs_dir") / "cp2k_label_input_manifest.csv", rows)


def _array_script(label_mode: str, rows: List[Dict[str, str]], config: Dict[str, Any]) -> str:
    hpc = config["hpc"]
    mode_rows = [r for r in rows if r.get("label_mode") == label_mode]
    job_dirs = " ".join(f'"{r["job_dir"]}"' for r in mode_rows)
    time_key = "time_sp" if label_mode == "sp_force" else "time_aimd"
    return f"""#!/bin/bash
#SBATCH --job-name=pepp_cp2k_{label_mode}
#SBATCH --nodes={int(hpc['nodes'])}
#SBATCH --ntasks-per-node={int(hpc['ntasks_per_node'])}
#SBATCH --cpus-per-task={int(hpc['cpus_per_task'])}
#SBATCH --time={hpc[time_key]}
#SBATCH --array=0-{max(len(mode_rows) - 1, 0)}

module purge
module load {hpc['cp2k_module_placeholder']}
# export CP2K_DATA_DIR="__SET_CP2K_DATA_DIR_ON_HPC__"
CP2K_CMD=${{CP2K_CMD:-{hpc['cp2k_command_default']}}}

JOB_DIRS=({job_dirs})
JOB_DIR="${{JOB_DIRS[$SLURM_ARRAY_TASK_ID]}}"
cd "$JOB_DIR"
"$CP2K_CMD" -i input.inp -o cp2k.out
"""


def make_hpc_cp2k_jobs(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_dirs(config)
    rows = [r for r in _read_rows(p(config, "cp2k_jobs_dir") / "cp2k_label_input_manifest.csv") if r.get("status") == "cp2k_input_written_no_cp2k_run"]
    out_rows = []
    for label_mode, filename in [("sp_force", "run_cp2k_sp_array.sbatch"), ("short_aimd", "run_cp2k_short_aimd_array.sbatch")]:
        script = p(config, "jobs_dir") / filename
        script.write_text(_array_script(label_mode, rows, config), encoding="utf-8")
        script.chmod(0o755)
        out_rows.append({"label_mode": label_mode, "script_path": str(script), "job_count": sum(1 for r in rows if r.get("label_mode") == label_mode)})
    for name in ["tiny", "pilot"]:
        sp_submit = p(config, "jobs_dir") / f"submit_cp2k_sp_{name}.sh"
        sp_submit.write_text("#!/bin/bash\nset -euo pipefail\nsbatch run_cp2k_sp_array.sbatch\n", encoding="utf-8")
        sp_submit.chmod(0o755)
        aimd_submit = p(config, "jobs_dir") / f"submit_cp2k_short_aimd_{name}.sh"
        aimd_submit.write_text("#!/bin/bash\nset -euo pipefail\nsbatch run_cp2k_short_aimd_array.sbatch\n", encoding="utf-8")
        aimd_submit.chmod(0o755)
        submit = p(config, "jobs_dir") / f"submit_cp2k_seed_{name}.sh"
        submit.write_text("#!/bin/bash\nset -euo pipefail\nsbatch run_cp2k_sp_array.sbatch\nsbatch run_cp2k_short_aimd_array.sbatch\n", encoding="utf-8")
        submit.chmod(0o755)
    return _write_rows(p(config, "jobs_dir") / "cp2k_hpc_job_manifest.csv", out_rows)


def _parse_energy_hartree(text: str) -> float | None:
    for pattern in [r"ENERGY\|\s+Total FORCE_EVAL.*?:\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", r"Total energy:\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"]:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])
    return None


def _cp2k_health(text: str) -> Tuple[bool, str]:
    lower = text.lower()
    if any(token in lower for token in ("abort", "fatal", "segmentation fault", "forrtl", "error termination", "cannot allocate memory", "oom-kill")):
        return False, "cp2k_abort_or_error"
    if any(token in lower for token in ("scf run not converged", "scf not converged", "no scf convergence")):
        return False, "scf_not_converged"
    normal = "program ended at" in lower or "cp2k ended" in lower
    return normal, "" if normal else "cp2k_normal_end_missing"


def _read_xyz_frames(path: Path) -> List[Tuple[List[str], List[List[float]]]]:
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


def _read_energies(path: Path) -> List[float]:
    if not path.exists():
        return []
    energies: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        try:
            nums = [float(x) for x in parts]
        except Exception:
            continue
        if len(nums) >= 5:
            energies.append(nums[4])
        elif len(nums) >= 2:
            energies.append(nums[-1])
    return energies


def _find_first(job_dir: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        candidates = sorted(job_dir.glob(pattern))
        if candidates:
            return candidates[0]
    return None


def _unit_factors(config: Dict[str, Any]) -> Tuple[float | None, float | None]:
    units = config.get("units", {})
    if units.get("output_energy", "eV") != "eV" or units.get("output_forces", "eV_per_A") != "eV_per_A":
        return None, None
    try:
        return float(units.get("cp2k_energy_hartree_to_eV", HARTREE_TO_EV)), float(units.get("cp2k_force_hartree_per_bohr_to_eV_per_A", FORCE_AU_TO_EV_A))
    except Exception:
        return None, None


def _comment_metadata(row: Dict[str, str], frame_index: int, energy_eV: float, box: Tuple[float, float, float], normal_end: bool) -> str:
    meta = {
        "energy": f"{energy_eV:.12f}",
        "energy_unit": "eV",
        "forces_unit": "eV/Angstrom",
        "cp2k_energy_raw_unit": "Hartree",
        "cp2k_force_raw_unit": "Hartree/Bohr",
        "source": "real_cp2k_output",
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
    fields = [
        f'Lattice="{box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]}"',
        "Properties=species:S:1:pos:R:3:forces:R:3",
        'pbc="T T T"',
    ]
    fields.extend(f'{key}="{value}"' if "/" in value or " " in value else f"{key}={value}" for key, value in meta.items())
    return " ".join(fields)


def _write_extxyz_frames(
    path: Path,
    row: Dict[str, str],
    elems: Sequence[str],
    positions: Sequence[Sequence[Sequence[float]]],
    forces: Sequence[Sequence[Sequence[float]]],
    energies_eV: Sequence[float],
    box: Tuple[float, float, float],
    normal_end: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for frame_index, (coords, force_frame, energy_eV) in enumerate(zip(positions, forces, energies_eV)):
            f.write(f"{len(elems)}\n")
            f.write(_comment_metadata(row, frame_index, energy_eV, box, normal_end) + "\n")
            for el, xyz, force in zip(elems, coords, force_frame):
                f.write(f"{el} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f} {force[0]:.12f} {force[1]:.12f} {force[2]:.12f}\n")


def parse_cp2k_outputs(config: Dict[str, Any]) -> Path:
    ensure_dirs(config)
    rows = []
    energy_factor, force_factor = _unit_factors(config)
    for row in _read_rows(p(config, "cp2k_jobs_dir") / "cp2k_label_input_manifest.csv"):
        if row.get("status") != "cp2k_input_written_no_cp2k_run":
            continue
        job_dir = Path(row["job_dir"])
        out_dir = p(config, "cp2k_parsed_dir") / row["aimd_structure_id"] / row["label_mode"]
        out_dir.mkdir(parents=True, exist_ok=True)
        project = row.get("cp2k_project") or f"{row['aimd_structure_id']}_{row['label_mode']}"
        summary = {
            "status": "not_run_no_cp2k_output",
            "usable_frame_count": 0,
            "failure_reason": "cp2k.out missing",
            "detected_cp2k_out": "",
            "detected_position_file": "",
            "detected_force_file": "",
            "detected_energy_file": "",
            "parser_mode": row.get("cp2k_run_type", "ENERGY_FORCE" if row.get("label_mode") == "sp_force" else "MD"),
            "n_position_frames": 0,
            "n_force_frames": 0,
            "n_energy_frames": 0,
            "n_frames_written": 0,
            "frame_count_mismatch": False,
        }
        frames = out_dir / "frames.extxyz"
        cp2k_out = job_dir / "cp2k.out"
        if cp2k_out.exists():
            summary["detected_cp2k_out"] = str(cp2k_out)
            text = cp2k_out.read_text(encoding="utf-8", errors="ignore")
            normal_end, health_reason = _cp2k_health(text)
            if energy_factor is None or force_factor is None:
                summary.update({"status": "failed_units_uncertain", "failure_reason": "unit_conversion_config_missing_or_invalid"})
            elif not normal_end:
                summary.update({"status": "failed_cp2k", "failure_reason": health_reason})
            elif row.get("label_mode") == "short_aimd" or row.get("cp2k_run_type") == "MD":
                pos_file = _find_first(job_dir, [f"{project}-pos-1.xyz", f"{project}-pos-*.xyz", "*pos*.xyz", "*POS*.xyz"])
                frc_file = _find_first(job_dir, [f"{project}-frc-1.xyz", f"{project}-frc-*.xyz", "*frc*.xyz", "*FRC*.xyz", "*forces*.xyz", "*FORCES*.xyz"])
                ene_file = _find_first(job_dir, [f"{project}-1.ener", f"{project}-*.ener", "*.ener", "*.ENER"])
                pos_frames = _read_xyz_frames(pos_file) if pos_file else []
                frc_frames = _read_xyz_frames(frc_file) if frc_file else []
                energies_h = _read_energies(ene_file) if ene_file else []
                matched = min(len(pos_frames), len(frc_frames), len(energies_h))
                summary.update(
                    {
                        "detected_position_file": str(pos_file or ""),
                        "detected_force_file": str(frc_file or ""),
                        "detected_energy_file": str(ene_file or ""),
                        "n_position_frames": len(pos_frames),
                        "n_force_frames": len(frc_frames),
                        "n_energy_frames": len(energies_h),
                        "n_frames_written": matched,
                        "frame_count_mismatch": len({len(pos_frames), len(frc_frames), len(energies_h)}) > 1,
                    }
                )
                if matched > 0:
                    elems = pos_frames[0][0]
                    _, _, box = _read_xyz_like(job_dir / "coords.xyz")
                    positions = [frame[1] for frame in pos_frames[:matched]]
                    forces = [[(x * force_factor, y * force_factor, z * force_factor) for x, y, z in frame[1]] for frame in frc_frames[:matched]]
                    energies = [x * energy_factor for x in energies_h[:matched]]
                    _write_extxyz_frames(frames, row, elems, positions, forces, energies, box, normal_end)
                    _write_rows(out_dir / "thermo.csv", [{"frame": i, "energy_Hartree": energies_h[i], "energy_eV": energies[i]} for i in range(matched)])
                    summary.update({"status": "parsed_real_cp2k_output", "usable_frame_count": matched, "failure_reason": ""})
                else:
                    summary.update({"status": "failed_cp2k", "failure_reason": "missing_matched_md_position_force_energy_frames"})
            else:
                elems, coords, box = _read_xyz_like(job_dir / "coords.xyz")
                energy_h = _parse_energy_hartree(text)
                force_file = _find_first(job_dir, [f"{project}-frc-1.xyz", f"{project}-frc-*.xyz", "forces.xyz", "*frc*.xyz", "*FRC*.xyz", "*forces*.xyz", "*FORCES*.xyz"])
                force_frames = _read_xyz_frames(force_file) if force_file else []
                summary.update(
                    {
                        "detected_position_file": str(job_dir / "coords.xyz"),
                        "detected_force_file": str(force_file or ""),
                        "detected_energy_file": str(cp2k_out),
                        "n_position_frames": 1,
                        "n_force_frames": len(force_frames),
                        "n_energy_frames": 1 if energy_h is not None else 0,
                    }
                )
                if energy_h is not None and force_frames:
                    force_values = force_frames[-1][1]
                    if len(force_values) == len(elems):
                        forces = [[(x * force_factor, y * force_factor, z * force_factor) for x, y, z in force_values]]
                        energy_eV = energy_h * energy_factor
                        _write_extxyz_frames(frames, row, elems, [coords], forces, [energy_eV], box, normal_end)
                        _write_rows(out_dir / "thermo.csv", [{"frame": 0, "energy_Hartree": energy_h, "energy_eV": energy_eV}])
                        summary.update({"status": "parsed_real_cp2k_output", "usable_frame_count": 1, "failure_reason": "", "n_frames_written": 1})
                    else:
                        summary.update({"status": "failed_cp2k", "failure_reason": "force_atom_count_mismatch"})
                else:
                    summary.update({"status": "failed_cp2k", "failure_reason": "missing_parseable_energy_or_forces"})
        (out_dir / "parse_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
        rows.append({"aimd_structure_id": row["aimd_structure_id"], "family": row.get("family", ""), "patch_id": row.get("patch_id", ""), "label_mode": row["label_mode"], **summary, "frames_extxyz_path": str(frames) if frames.exists() else "", "parse_summary_path": str(out_dir / "parse_summary.yaml")})
    if not rows:
        rows.append({"aimd_structure_id": "none", "label_mode": "none", "status": "not_run_no_cp2k_output", "usable_frame_count": 0, "failure_reason": "no cp2k input jobs"})
    return _write_rows(p(config, "cp2k_parsed_dir") / "cp2k_parsed_manifest.csv", rows)


def _frame_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip().isdigit())


def build_aimd_dataset(config: Dict[str, Any]) -> Path:
    ensure_dirs(config)
    parsed = _read_rows(p(config, "cp2k_parsed_dir") / "cp2k_parsed_manifest.csv")
    real = [r for r in parsed if r.get("status") == "parsed_real_cp2k_output"]
    min_frames = int(config["dataset"]["min_frames_for_training"])
    total = sum(int(r.get("usable_frame_count", 0)) for r in real)
    rows = [{"split": split, "extxyz_path": "", "frame_count": 0} for split in ["train", "val", "test"]]
    if total < min_frames:
        summary = {"dataset_status": "insufficient_real_cp2k_frames", "usable_for_mlff_training": False, "failure_reason": f"real frames < {min_frames}"}
    else:
        block_keys = list(dict.fromkeys("|".join([str(r.get("family", "")), str(r.get("patch_id", "")), str(r.get("aimd_structure_id", ""))]) for r in real))
        split_for = {}
        for i, key in enumerate(block_keys):
            frac = i / max(len(block_keys), 1)
            split_for[key] = "train" if frac < float(config["dataset"]["train_fraction"]) else "val" if frac < float(config["dataset"]["train_fraction"]) + float(config["dataset"]["val_fraction"]) else "test"
        rows = []
        for split in ["train", "val", "test"]:
            part = [r for r in real if split_for.get("|".join([str(r.get("family", "")), str(r.get("patch_id", "")), str(r.get("aimd_structure_id", ""))])) == split]
            out = p(config, "aimd_dataset_dir") / f"{split}.extxyz"
            if part:
                out.write_text("".join(Path(r["frames_extxyz_path"]).read_text(encoding="utf-8") for r in part), encoding="utf-8")
            rows.append({"split": split, "extxyz_path": str(out) if part else "", "frame_count": sum(int(r.get("usable_frame_count", 0)) for r in part)})
        summary = {"dataset_status": "ready", "usable_for_mlff_training": True, "failure_reason": ""}
    (p(config, "aimd_dataset_dir") / "dataset_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    return _write_rows(p(config, "aimd_dataset_dir") / "dataset_manifest.csv", rows)


def validate_aimd_dataset(config: Dict[str, Any]) -> Path:
    ensure_dirs(config)
    summary_path = p(config, "aimd_dataset_dir") / "dataset_summary.yaml"
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"dataset_status": "insufficient_real_cp2k_frames", "usable_for_mlff_training": False, "failure_reason": "dataset not built"}
    rows = []
    for row in _read_rows(p(config, "aimd_dataset_dir") / "dataset_manifest.csv"):
        path = Path(row.get("extxyz_path", ""))
        rows.append({"split": row["split"], "extxyz_path": str(path) if row.get("extxyz_path") else "", "exists": bool(row.get("extxyz_path")) and path.exists(), "frame_count": _frame_count(path) if row.get("extxyz_path") else 0, "valid": bool(summary.get("usable_for_mlff_training")) and bool(row.get("extxyz_path")) and path.exists()})
    if not rows:
        rows.append({"split": "none", "exists": False, "frame_count": 0, "valid": False})
    (p(config, "logs_dir") / "aimd_dataset_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    return _write_rows(p(config, "logs_dir") / "aimd_dataset_validation.csv", rows)


def export_aimd_dataset_manifest(config: Dict[str, Any]) -> Path:
    ensure_dirs(config)
    summary_path = p(config, "aimd_dataset_dir") / "dataset_summary.yaml"
    summary = yaml.safe_load(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"dataset_status": "insufficient_real_cp2k_frames", "usable_for_mlff_training": False, "failure_reason": "dataset not built"}
    by_split = {r["split"]: r for r in _read_rows(p(config, "aimd_dataset_dir") / "dataset_manifest.csv")}
    row = {
        "dataset_id": config["dataset"].get("dataset_id", "pepp_silica_aimd_cp2k"),
        "train_extxyz_path": by_split.get("train", {}).get("extxyz_path", ""),
        "val_extxyz_path": by_split.get("val", {}).get("extxyz_path", ""),
        "test_extxyz_path": by_split.get("test", {}).get("extxyz_path", ""),
        "dataset_manifest_path": str(p(config, "aimd_dataset_dir") / "dataset_manifest.csv"),
        "num_train_frames": by_split.get("train", {}).get("frame_count", 0),
        "num_val_frames": by_split.get("val", {}).get("frame_count", 0),
        "num_test_frames": by_split.get("test", {}).get("frame_count", 0),
        "elements": ";".join(config["dataset_scope"]["elements"]),
        "cp2k_method": config["cp2k"]["method"],
        "basis_set": config["cp2k"]["basis_set"],
        "potential": config["cp2k"]["potential"],
        "dispersion": config["cp2k"]["dispersion"],
        "aimd_structure_source_manifest": str(p(config, "aimd_structure_manifest")),
        "has_energy": bool(config["dataset"]["require_energy"]),
        "has_forces": bool(config["dataset"]["require_forces"]),
        "has_stress": False,
        "dataset_status": summary["dataset_status"],
        "usable_for_mlff_training": bool(summary["usable_for_mlff_training"]),
        "failure_reason": summary.get("failure_reason", ""),
    }
    out_csv = p(config, "exports_dir") / "aimd_dataset_manifest.csv"
    _write_rows(out_csv, [row])
    row = {key: (value.replace("\\", "/") if isinstance(value, str) else value) for key, value in row.items()}
    (p(config, "exports_dir") / "aimd_dataset_manifest.json").write_text(json.dumps([row], indent=2) + "\n", encoding="utf-8")
    return out_csv
