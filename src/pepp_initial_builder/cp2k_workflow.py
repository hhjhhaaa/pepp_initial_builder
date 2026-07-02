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


def _kind_block(elems: Sequence[str], config: Dict[str, Any]) -> str:
    cp2k = config["cp2k"]
    lines: List[str] = []
    present = set(elems)
    for element in ["C", "H", "O", "Si"]:
        if element in present:
            lines += [f"    &KIND {element}", f"      BASIS_SET {cp2k['basis_set']}", f"      POTENTIAL {cp2k['potential']}", "    &END KIND"]
    return "\n".join(lines)


def _input_text(aimd_id: str, label_mode: str, elems: Sequence[str], config: Dict[str, Any], mode: str) -> str:
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
    @INCLUDE cell.inc
    &TOPOLOGY
      COORD_FILE_NAME coords.xyz
      COORD_FILE_FORMAT XYZ
    &END TOPOLOGY
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
            _coords_xyz(job_dir / "coords.xyz", elems, coords)
            (job_dir / "cell.inc").write_text(_cell_inc(box), encoding="utf-8")
            (job_dir / "input.inp").write_text(_input_text(aimd_id, label_mode, elems, config, mode), encoding="utf-8")
            (job_dir / "job_metadata.yaml").write_text(yaml.safe_dump({"aimd_structure_id": aimd_id, "family": family, "label_mode": label_mode, "source_extxyz_path": str(src), "cp2k_run_performed": False}, sort_keys=False), encoding="utf-8")
            (job_dir / "run_status.json").write_text(json.dumps({"status": "input_written_not_submitted", "cp2k_out": str(job_dir / "cp2k.out")}, indent=2) + "\n", encoding="utf-8")
            rows.append({"aimd_structure_id": aimd_id, "family": family, "label_mode": label_mode, "status": "cp2k_input_written_no_cp2k_run", "job_dir": str(job_dir), "input_inp_path": str(job_dir / "input.inp"), "coords_xyz_path": str(job_dir / "coords.xyz"), "cell_inc_path": str(job_dir / "cell.inc"), "source_extxyz_path": str(src)})
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


def _parse_forces_au(path: Path, n_atoms: int) -> List[Tuple[float, float, float]] | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if lines and lines[0].strip().isdigit():
        rows = []
        for line in lines[2 : 2 + int(lines[0].strip())]:
            parts = line.split()
            if len(parts) >= 4:
                rows.append((float(parts[-3]), float(parts[-2]), float(parts[-1])))
        return rows if len(rows) == n_atoms else None
    return None


def _write_extxyz(path: Path, elems: Sequence[str], coords: Sequence[Sequence[float]], box: Tuple[float, float, float], energy_eV: float, forces: Sequence[Tuple[float, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(elems)}\n")
        f.write(f'Lattice="{box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]}" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy_eV:.12f} source=real_cp2k_output pbc="T T T"\n')
        for el, xyz, force in zip(elems, coords, forces):
            f.write(f"{el} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f} {force[0]:.12f} {force[1]:.12f} {force[2]:.12f}\n")


def parse_cp2k_outputs(config: Dict[str, Any]) -> Path:
    ensure_dirs(config)
    rows = []
    for row in _read_rows(p(config, "cp2k_jobs_dir") / "cp2k_label_input_manifest.csv"):
        if row.get("status") != "cp2k_input_written_no_cp2k_run":
            continue
        job_dir = Path(row["job_dir"])
        out_dir = p(config, "cp2k_parsed_dir") / row["aimd_structure_id"] / row["label_mode"]
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {"status": "not_run_no_cp2k_output", "usable_frame_count": 0, "failure_reason": "cp2k.out missing"}
        frames = out_dir / "frames.extxyz"
        cp2k_out = job_dir / "cp2k.out"
        if cp2k_out.exists():
            elems, coords, box = _read_xyz_like(job_dir / "coords.xyz")
            energy_h = _parse_energy_hartree(cp2k_out.read_text(encoding="utf-8", errors="ignore"))
            forces_au = None
            for cand in [job_dir / "forces.xyz", *sorted(job_dir.glob("*forces*.xyz")), *sorted(job_dir.glob("*FORCES*.xyz"))]:
                forces_au = _parse_forces_au(cand, len(elems))
                if forces_au:
                    break
            if energy_h is not None and forces_au is not None:
                forces = [(x * FORCE_AU_TO_EV_A, y * FORCE_AU_TO_EV_A, z * FORCE_AU_TO_EV_A) for x, y, z in forces_au]
                _write_extxyz(frames, elems, coords, box, energy_h * HARTREE_TO_EV, forces)
                _write_rows(out_dir / "thermo.csv", [{"frame": 0, "energy_Hartree": energy_h, "energy_eV": energy_h * HARTREE_TO_EV}])
                summary = {"status": "parsed_real_cp2k_output", "usable_frame_count": 1, "failure_reason": ""}
            else:
                summary = {"status": "failed_cp2k", "usable_frame_count": 0, "failure_reason": "missing_parseable_energy_or_forces"}
        (out_dir / "parse_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
        rows.append({"aimd_structure_id": row["aimd_structure_id"], "family": row.get("family", ""), "label_mode": row["label_mode"], **summary, "frames_extxyz_path": str(frames) if frames.exists() else "", "parse_summary_path": str(out_dir / "parse_summary.yaml")})
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
        families = list(dict.fromkeys(str(r.get("family", "")) for r in real))
        split_for = {}
        for i, fam in enumerate(families):
            frac = i / max(len(families), 1)
            split_for[fam] = "train" if frac < float(config["dataset"]["train_fraction"]) else "val" if frac < float(config["dataset"]["train_fraction"]) + float(config["dataset"]["val_fraction"]) else "test"
        rows = []
        for split in ["train", "val", "test"]:
            part = [r for r in real if split_for.get(str(r.get("family", ""))) == split]
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
