from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from pepp_initial_builder.common.run import run_id
from pepp_initial_builder.pore.config import ensure_pore_dirs, pore_root
from pepp_initial_builder.pore.porems_builder import read_xyz_like

SILICA_TYPE = {"Si": 3, "O": 4, "H": 5}
SILICA_MASS = {3: ("Si", 28.0855), 4: ("O", 15.9994), 5: ("H_silica", 1.008)}
SILICA_LJ = {
    3: (0.0930, 4.15),
    4: (0.0540, 3.47),
    5: (0.0100, 2.50),
}


def _lammps_executable(config: Dict[str, Any]) -> str | None:
    hinted = config.get("tools", {}).get("known_lammps_executable")
    if hinted and Path(str(hinted)).expanduser().exists():
        return str(Path(str(hinted)).expanduser())
    return shutil.which("lmp") or shutil.which("lammps")


def _section_rows(path: Path, section: str) -> List[List[str]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip().split("#", 1)[0].strip() == section:
            start = idx + 1
            break
    if start is None:
        return []
    rows: List[List[str]] = []
    for line in lines[start:]:
        clean = line.split("#", 1)[0].strip()
        if not clean:
            if rows:
                break
            continue
        first = clean.split()[0]
        if not first.replace("-", "").isdigit():
            if rows:
                break
            continue
        rows.append(clean.split())
    return rows


def _template_topology(data_path: Path) -> Dict[str, Any]:
    atoms = _section_rows(data_path, "Atoms")
    sections = {name: _section_rows(data_path, name) for name in ["Bonds", "Angles", "Dihedrals", "Impropers"]}
    atom_rows = []
    for row in atoms:
        atom_rows.append(
            {
                "old_id": int(row[0]),
                "mol": int(row[1]),
                "type": int(row[2]),
                "charge": float(row[3]),
            }
        )
    return {"atoms": atom_rows, **sections}


def _copy_polymer_params(template_dirs: Sequence[Path], out: Path) -> None:
    params = next((d / "polymer.params" for d in template_dirs if (d / "polymer.params").exists()), None)
    if not params:
        raise RuntimeError("Missing EMC polymer.params for LAMMPS relaxation")
    text = params.read_text(encoding="utf-8", errors="ignore")
    out.write_text(text, encoding="utf-8")


def _write_combined_lammps_data(seed_extxyz: Path, template_dirs: Sequence[Path], out: Path) -> Tuple[int, int]:
    elems, coords, box = read_xyz_like(seed_extxyz)
    template_tops = [_template_topology(d / "polymer.data") for d in template_dirs]
    n_polymer = sum(len(t["atoms"]) for t in template_tops)
    n_silica = len(elems) - n_polymer
    if n_silica <= 0:
        raise RuntimeError("Cannot identify silica/polymer split in full-pore seed")
    if any(elem not in SILICA_TYPE for elem in elems[:n_silica]):
        raise RuntimeError("Full-pore seed atom order is not silica-first followed by polymer templates")
    bond_count = sum(len(t["Bonds"]) for t in template_tops)
    angle_count = sum(len(t["Angles"]) for t in template_tops)
    dihedral_count = sum(len(t["Dihedrals"]) for t in template_tops)
    improper_count = sum(len(t["Impropers"]) for t in template_tops)
    max_bond_type = max([int(row[1]) for t in template_tops for row in t["Bonds"]] or [0])
    max_angle_type = max([int(row[1]) for t in template_tops for row in t["Angles"]] or [0])
    max_dihedral_type = max([int(row[1]) for t in template_tops for row in t["Dihedrals"]] or [0])
    max_improper_type = max([int(row[1]) for t in template_tops for row in t["Impropers"]] or [0])
    with out.open("w", encoding="utf-8") as handle:
        handle.write("PE/PP-silica full-pore LAMMPS relax data; polymer topology from EMC, silica fixed host\n\n")
        handle.write(f"{len(elems)} atoms\n{bond_count} bonds\n{angle_count} angles\n{dihedral_count} dihedrals\n{improper_count} impropers\n\n")
        handle.write("5 atom types\n")
        handle.write(f"{max_bond_type} bond types\n{max_angle_type} angle types\n{max_dihedral_type} dihedral types\n{max_improper_type} improper types\n\n")
        handle.write(f"0.0 {box[0]:.10f} xlo xhi\n0.0 {box[1]:.10f} ylo yhi\n0.0 {box[2]:.10f} zlo zhi\n\n")
        handle.write("Masses\n\n")
        handle.write("1 12.01115 # c\n2 1.00797 # hc\n")
        for type_id, (_name, mass) in SILICA_MASS.items():
            handle.write(f"{type_id} {mass:.6f} # {_name}\n")
        handle.write("\nAtoms\n\n")
        for idx in range(n_silica):
            x, y, z = coords[idx]
            handle.write(f"{idx + 1} 0 {SILICA_TYPE[elems[idx]]} 0.000000 {x:.10f} {y:.10f} {z:.10f}\n")
        old_to_new: Dict[Tuple[int, int], int] = {}
        atom_offset = n_silica
        source_offset = n_silica
        mol_offset = 1
        for tidx, top in enumerate(template_tops):
            for local_idx, atom in enumerate(top["atoms"]):
                new_id = atom_offset + local_idx + 1
                x, y, z = coords[source_offset + local_idx]
                old_to_new[(tidx, atom["old_id"])] = new_id
                handle.write(f"{new_id} {atom['mol'] + mol_offset} {atom['type']} {atom['charge']:.6f} {x:.10f} {y:.10f} {z:.10f}\n")
            atom_offset += len(top["atoms"])
            source_offset += len(top["atoms"])
            mol_offset += max(atom["mol"] for atom in top["atoms"])
        for section, width in [("Bonds", 2), ("Angles", 3), ("Dihedrals", 4), ("Impropers", 4)]:
            rows = []
            for tidx, top in enumerate(template_tops):
                for row in top[section]:
                    atom_ids = [old_to_new[(tidx, int(x))] for x in row[2 : 2 + width]]
                    rows.append([int(row[1]), *atom_ids])
            handle.write(f"\n{section}\n\n")
            for idx, row in enumerate(rows, start=1):
                handle.write(f"{idx} " + " ".join(str(x) for x in row) + "\n")
    return n_silica, n_polymer


def _write_lammps_input(path: Path, warmup_steps: int, high_steps: int, cool_steps: int, target_steps: int, cfg: Dict[str, Any]) -> None:
    initial_temp = float(cfg.get("initial_temperature_K", 300.0))
    temp = float(cfg.get("temperature_K", 523.0))
    high_temp = float(cfg.get("high_temperature_K", 650.0))
    timestep = float(cfg.get("timestep_fs", 0.5))
    thermo = int(cfg.get("thermo_every", 500))
    dump_every = int(cfg.get("dump_every", 1000))
    path.write_text(
        f"""units real
atom_style full
boundary p p p
read_data full_pore_relax.data
include polymer.params

mass 3 28.085500
mass 4 15.999400
mass 5 1.008000
pair_coeff 3 3 {SILICA_LJ[3][0]:.6f} {SILICA_LJ[3][1]:.6f}
pair_coeff 4 4 {SILICA_LJ[4][0]:.6f} {SILICA_LJ[4][1]:.6f}
pair_coeff 5 5 {SILICA_LJ[5][0]:.6f} {SILICA_LJ[5][1]:.6f}

group silica type 3 4 5
group polymer subtract all silica
fix freeze_silica silica setforce 0.0 0.0 0.0
velocity silica set 0.0 0.0 0.0

neighbor 3.0 bin
neigh_modify every 1 delay 0 check yes exclude group silica silica
comm_modify cutoff 20.0
kspace_style pppm/cg 1.0e-4

compute tpoly polymer temp
thermo_modify temp tpoly lost error flush yes
thermo {thermo}
thermo_style custom step c_tpoly pe ke etotal press atoms

# Stage 0: remove bad contacts from constrained Packmol insertion.
min_style fire
minimize {cfg.get("minimize_etol", "1.0e-6")} {cfg.get("minimize_ftol", "1.0e-8")} {int(cfg.get("minimize_maxiter", 5000))} {int(cfg.get("minimize_maxeval", 20000))}

# Stage 1: polymer-only warmup inside the fixed silica pore.
velocity polymer create {initial_temp:.3f} 91023 mom yes rot yes dist gaussian
timestep {timestep:.6f}
fix int polymer nvt temp {initial_temp:.3f} {temp:.3f} 100.0
fix mom polymer momentum 200 linear 1 1 1
dump relax_dump all custom {dump_every} relax.lammpstrj id mol type q x y z
dump_modify relax_dump sort id
run {warmup_steps}
unfix int

# Stage 2: high-temperature NVT anneal of the mobile polymer inside the fixed pore.
fix int polymer nvt temp {high_temp:.3f} {high_temp:.3f} 100.0
run {high_steps}
unfix int

# Stage 3: cool to target temperature without changing the fixed pore cell.
fix int polymer nvt temp {high_temp:.3f} {temp:.3f} 100.0
run {cool_steps}
unfix int

# Stage 4: target-temperature NVT hold. CP2K crops are sampled only from this equilibrated tail.
fix int polymer nvt temp {temp:.3f} {temp:.3f} 100.0
run {target_steps}
write_data relaxed.data
write_dump all custom relaxed_snapshot.dump id mol type q x y z modify sort id
""",
        encoding="utf-8",
    )


def _dump_to_extxyz(dump: Path, out: Path, box: Tuple[float, float, float]) -> None:
    lines = dump.read_text(encoding="utf-8", errors="ignore").splitlines()
    marker = lines.index("ITEM: ATOMS id mol type q x y z")
    rows = []
    for line in lines[marker + 1 :]:
        parts = line.split()
        if len(parts) >= 7:
            rows.append((int(parts[0]), int(parts[2]), float(parts[4]), float(parts[5]), float(parts[6])))
    rows.sort(key=lambda x: x[0])
    elem_by_type = {1: "C", 2: "H", 3: "Si", 4: "O", 5: "H"}
    with out.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(rows)}\n")
        handle.write(f'Lattice="{box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]}" Properties=species:S:1:pos:R:3 pbc="T T T"\n')
        for _atom_id, type_id, x, y, z in rows:
            handle.write(f"{elem_by_type[type_id]} {x:.10f} {y:.10f} {z:.10f}\n")


def _parse_thermo_log(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if not path.exists():
        return rows
    header: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[0] == "Step" and ("c_tpoly" in parts or "Temp" in parts):
            header = parts
            continue
        if not header or len(parts) < len(header):
            continue
        try:
            values = [float(x) for x in parts[: len(header)]]
        except Exception:
            continue
        rows.append(dict(zip(header, values)))
    return rows


def _min_distances_to_silica(elems: Sequence[str], coords: np.ndarray, n_silica: int, atom_ids: Sequence[int]) -> List[float]:
    silica_heavy = [idx for idx in range(min(n_silica, len(elems))) if elems[idx] in {"Si", "O"}]
    if not silica_heavy:
        return [float("inf") for _idx in atom_ids]
    silica_coords = coords[silica_heavy]
    distances = []
    for idx in atom_ids:
        distances.append(float(np.min(np.linalg.norm(silica_coords - coords[idx], axis=1))))
    return distances


def _relax_metrics(
    config: Dict[str, Any],
    row: Dict[str, Any],
    seed_dir: Path,
    relaxed_path: str,
    n_silica: int,
    warmup_steps: int,
    high_steps: int,
    cool_steps: int,
    target_steps: int,
) -> Dict[str, Any]:
    gate = config.get("relax_quality_gate", {})
    relax_cfg = config.get("lammps_full_pore_relax", {})
    timestep_fs = float(relax_cfg.get("timestep_fs", 0.5))
    thermo = _parse_thermo_log(seed_dir / "lammps_relax" / "lammps_relax.log")
    hold_start = float(warmup_steps + high_steps + cool_steps)
    hold_rows = [r for r in thermo if r.get("Step", 0.0) >= hold_start]
    temps = [float(r.get("c_tpoly", r.get("Temp", float("nan")))) for r in hold_rows if np.isfinite(r.get("c_tpoly", r.get("Temp", float("nan"))))]
    target_temp = float(gate.get("target_temperature_K", relax_cfg.get("temperature_K", 523.0)))
    temp_mean = float(np.mean(temps)) if temps else float("nan")
    temp_std = float(np.std(temps)) if temps else float("nan")
    final_thermo = hold_rows[-1] if hold_rows else (thermo[-1] if thermo else {})
    metrics: Dict[str, Any] = {
        "run_id": run_id(config),
        "full_pore_seed_id": row.get("full_pore_seed_id", ""),
        "polymer_architecture": row.get("polymer_architecture", ""),
        "pe_variant": row.get("pe_variant", ""),
        "pp_variant": row.get("pp_variant", ""),
        "composition": row.get("composition", ""),
        "loading_mode": row.get("loading_mode", ""),
        "seed": row.get("seed", ""),
        "time_ps": (warmup_steps + high_steps + cool_steps + target_steps) * timestep_fs / 1000.0,
        "temperature_K": temp_mean,
        "hold_temperature_mean_K": temp_mean,
        "hold_temperature_std_K": temp_std,
        "potential_energy": final_thermo.get("PotEng", ""),
        "total_energy": final_thermo.get("TotEng", ""),
        "polymer_inside_pore_fraction": 0.0,
        "min_polymer_silica_distance_A": 0.0,
        "polymer_silica_contact_count_3p5A": 0,
        "polymer_silica_contact_count_5p0A": 0,
        "mean_wall_distance_A": "",
        "pe_near_wall_fraction": "",
        "pp_near_wall_fraction": "",
        "usable_for_cp2k_crop": False,
        "failure_reason": "",
    }
    try:
        elems, coords_list, box = read_xyz_like(Path(relaxed_path))
        coords = np.array(coords_list, dtype=float)
        polymer_ids = [idx for idx in range(n_silica, len(elems)) if elems[idx] in {"C", "H"}]
        polymer_heavy = [idx for idx in polymer_ids if elems[idx] == "C"]
        center = (box[0] / 2.0, box[1] / 2.0)
        matrix = config.get("full_pore_seed_matrix", {})
        diameters = matrix.get("pore_diameter_nm", [4.0])
        nominal_radius = float(diameters[0]) * 10.0 / 2.0 if diameters else 20.0
        silica_inner_radius = float(np.min(np.linalg.norm(coords[:n_silica, :2] - np.array(center), axis=1)))
        radius = max(nominal_radius, silica_inner_radius)
        if polymer_ids:
            radial = np.linalg.norm(coords[polymer_ids, :2] - np.array(center), axis=1)
            # The relaxed full-pore LAMMPS run is periodic in z. Wrapped chain atoms
            # near z=0/lz are still inside the pore, so use radial containment here;
            # silica overlap/contact gates handle wall penetration separately.
            inside = radial < radius
            metrics["polymer_inside_pore_fraction"] = float(np.mean(inside))
        distances = _min_distances_to_silica(elems, coords, n_silica, polymer_heavy)
        if distances:
            arr = np.array(distances, dtype=float)
            metrics["min_polymer_silica_distance_A"] = float(np.min(arr))
            metrics["polymer_silica_contact_count_3p5A"] = int(np.sum(arr <= 3.5))
            metrics["polymer_silica_contact_count_5p0A"] = int(np.sum(arr <= 5.0))
            metrics["mean_wall_distance_A"] = float(np.mean(arr))
        roles_path = seed_dir / "atom_roles.csv"
        if roles_path.exists() and distances:
            roles = pd.read_csv(roles_path)
            dist_by_atom = {atom_id + 1: dist for atom_id, dist in zip(polymer_heavy, distances)}
            near = {atom for atom, dist in dist_by_atom.items() if dist <= 5.0}
            pe_atoms = set(int(x) for x in roles.loc[roles.get("polymer_type", "") == "PE", "atom_id"].tolist())
            pp_atoms = set(int(x) for x in roles.loc[roles.get("polymer_type", "") == "PP", "atom_id"].tolist())
            metrics["pe_near_wall_fraction"] = float(len(pe_atoms & near) / max(len(pe_atoms), 1))
            metrics["pp_near_wall_fraction"] = float(len(pp_atoms & near) / max(len(pp_atoms), 1))
        max_abs = float(np.max(np.abs(coords))) if len(coords) else float("inf")
        temp_ok = bool(temps) and abs(temp_mean - target_temp) <= float(gate.get("hold_temperature_mean_tolerance_K", 50.0)) and temp_std <= float(gate.get("hold_temperature_std_max_K", 80.0))
        inside_ok = metrics["polymer_inside_pore_fraction"] >= float(gate.get("min_polymer_inside_pore_fraction", 0.95))
        dist_ok = metrics["min_polymer_silica_distance_A"] >= float(gate.get("min_polymer_silica_distance_A", 1.6))
        contact_ok = metrics["polymer_silica_contact_count_3p5A"] >= int(gate.get("min_contact_count_3p5A", 1)) or metrics["polymer_silica_contact_count_5p0A"] >= int(gate.get("min_contact_count_5p0A", 10))
        sane_ok = max_abs <= float(gate.get("max_abs_coordinate_A", 100000.0))
        failures = []
        if not temp_ok:
            failures.append("hold_temperature_not_stable")
        if not inside_ok:
            failures.append("polymer_not_inside_pore")
        if not dist_ok:
            failures.append("polymer_silica_overlap")
        if not contact_ok:
            failures.append("no_polymer_silica_wall_contact")
        if not sane_ok:
            failures.append("exploding_atoms_coordinate_sanity_failed")
        metrics["usable_for_cp2k_crop"] = not failures
        metrics["failure_reason"] = ";".join(failures)
    except Exception as exc:
        metrics["failure_reason"] = f"relax_metrics_failed:{exc}"
    return metrics


def _write_relax_products(config: Dict[str, Any], relax_rows: List[Dict[str, Any]], metric_rows: List[Dict[str, Any]], suffix: str = "") -> None:
    root = pore_root(config)
    logs_dir = root / config["paths"].get("logs_dir", "outputs/logs")
    exports_dir = root / config["paths"].get("aimd_exports_dir", config["paths"].get("exports_dir", "data/exports"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(logs_dir / f"full_pore_relax_metrics{suffix}.csv", index=False)
    metric_by_id = {str(row.get("full_pore_seed_id", "")): row for row in metric_rows}
    snapshot_rows = []
    for row in relax_rows:
        seed_id = str(row.get("full_pore_seed_id", ""))
        metric = metric_by_id.get(seed_id, {})
        relaxed = str(row.get("relaxed_extxyz_path", ""))
        if row.get("status") != "lammps_relaxed_full_pore" or not relaxed:
            continue
        snapshot_rows.append(
            {
                "run_id": run_id(config),
                "full_pore_seed_id": seed_id,
                "source_full_pore_id": seed_id,
                "source_stage": "lammps_relaxed_full_pore",
                "snapshot_kind": "final_relaxed_structure",
                "snapshot_path": relaxed,
                "source_snapshot_path": relaxed,
                "frame_index": 0,
                "source_frame_index": 0,
                "time_ps": row.get("lammps_total_time_ps", ""),
                "temperature_K": metric.get("temperature_K", ""),
                "from_last_fraction": True,
                "usable_for_cp2k_crop": bool(metric.get("usable_for_cp2k_crop", False)),
                "failure_reason": metric.get("failure_reason", ""),
                "polymer_architecture": row.get("polymer_architecture", ""),
                "pe_variant": row.get("pe_variant", ""),
                "pp_variant": row.get("pp_variant", ""),
                "composition": row.get("composition", ""),
                "loading_mode": row.get("loading_mode", ""),
                "seed": row.get("seed", ""),
                "atom_roles_path": row.get("atom_roles_path", ""),
                "n_silica_atoms_fixed": row.get("n_silica_atoms_fixed", ""),
            }
        )
    pd.DataFrame(snapshot_rows).to_csv(exports_dir / f"full_pore_snapshot_manifest{suffix}.csv", index=False)


def collect_lammps_relax_manifests(config: Dict[str, Any]) -> Path:
    root = pore_root(config)
    structures_dir = root / config["paths"]["full_pore_seed_structures_dir"]
    logs_dir = root / config["paths"].get("logs_dir", "outputs/logs")
    exports_dir = root / config["paths"].get("aimd_exports_dir", config["paths"].get("exports_dir", "data/exports"))
    manifest_parts = sorted(structures_dir.glob("lammps_relax_manifest.task*.csv"))
    frames = [pd.read_csv(path) for path in manifest_parts if path.exists()]
    if frames:
        relax_df = pd.concat(frames, ignore_index=True)
        relax_df.to_csv(structures_dir / "lammps_relax_manifest.csv", index=False)
        metric_rows: List[Dict[str, Any]] = []
        for row in relax_df.to_dict("records"):
            if row.get("status") != "lammps_relaxed_full_pore" or not row.get("relaxed_extxyz_path"):
                continue
            try:
                seed_dir = Path(str(row["relaxed_extxyz_path"])).parent
                metric_rows.append(
                    _relax_metrics(
                        config,
                        row,
                        seed_dir,
                        str(row["relaxed_extxyz_path"]),
                        int(row.get("n_silica_atoms_fixed", 0)),
                        int(row.get("lammps_warmup_steps", 0)),
                        int(row.get("lammps_high_temp_steps", 0)),
                        int(row.get("lammps_cool_steps", 0)),
                        int(row.get("lammps_target_temp_steps", 0)),
                    )
                )
            except Exception as exc:
                metric_rows.append({"full_pore_seed_id": row.get("full_pore_seed_id", ""), "usable_for_cp2k_crop": False, "failure_reason": f"collect_metric_recompute_failed:{exc}"})
        _write_relax_products(config, relax_df.to_dict("records"), metric_rows)
    return exports_dir / "full_pore_snapshot_manifest.csv"


def write_lammps_relax_array_script(config: Dict[str, Any], mode: str = "pilot") -> Path:
    root = pore_root(config)
    jobs_dir = root / config["paths"].get("jobs_dir", "outputs/jobs")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    seed_manifest = root / config["paths"]["full_pore_seed_structures_dir"] / "full_pore_seed_manifest.csv"
    count = 0
    if seed_manifest.exists():
        rows = pd.read_csv(seed_manifest)
        count = int((rows.get("status", pd.Series(dtype=str)).astype(str).str.startswith("available")).sum())
    array_max = max(count - 1, 0)
    partition = str(config.get("hpc", {}).get("partition", "batch")) if config.get("hpc") else "batch"
    script = jobs_dir / "run_lammps_relax_array.sbatch"
    lmp = config.get("tools", {}).get("known_lammps_executable") or "__SET_LAMMPS_EXECUTABLE_ON_HPC__"
    script.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=pepp_lammps_relax
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00
#SBATCH --array=0-{array_max}

set -euo pipefail
export LAMMPS_EXECUTABLE="{lmp}"
if command -v conda >/dev/null 2>&1; then
  set +u
  eval "$(conda shell.bash hook)" || true
  conda activate peppmixure || true
  set -u
fi
export PYTHONPATH="$PWD/src:${{PYTHONPATH:-}}"
PYTHON_CMD=${{PYTHON_CMD:-/public/home/jinhao.hu/.conda/envs/peppmixure/bin/python}}
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  PYTHON_CMD=python
fi
"$PYTHON_CMD" - <<'PY'
import os
from pepp_initial_builder.pore.config import load_pore_config
from pepp_initial_builder.mlff_seed.lammps_relax import write_lammps_relax_inputs
cfg = load_pore_config("configs/mlff_seed.yaml")
cfg.setdefault("tools", {{}})["known_lammps_executable"] = os.environ["LAMMPS_EXECUTABLE"]
cfg["runtime_filter"] = {{"available_seed_index": int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))}}
write_lammps_relax_inputs(cfg, "{mode}")
PY
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    collect = jobs_dir / "collect_lammps_relax_manifests.sh"
    collect.write_text(
        """#!/bin/bash
set -euo pipefail
if command -v conda >/dev/null 2>&1; then
  set +u
  eval "$(conda shell.bash hook)" || true
  conda activate peppmixure || true
  set -u
fi
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
PYTHON_CMD=${PYTHON_CMD:-/public/home/jinhao.hu/.conda/envs/peppmixure/bin/python}
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  PYTHON_CMD=python
fi
"$PYTHON_CMD" scripts/mlff_seed/collect_lammps_relax.py --config configs/mlff_seed.yaml
""",
        encoding="utf-8",
    )
    collect.chmod(0o755)
    return script


def write_lammps_relax_inputs(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_pore_dirs(config)
    root = pore_root(config)
    runtime_filter = config.get("runtime_filter", {})
    selected_available_index = runtime_filter.get("available_seed_index")
    suffix = f".task{selected_available_index}" if selected_available_index is not None else ""
    out = root / config["paths"]["full_pore_seed_structures_dir"] / f"lammps_relax_manifest{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    seed_manifest = root / config["paths"]["full_pore_seed_structures_dir"] / "full_pore_seed_manifest.csv"
    lmp = _lammps_executable(config)
    relax_cfg = config.get("lammps_full_pore_relax", {})
    high_steps = int(relax_cfg.get(f"{mode}_high_temp_steps", 10000))
    warmup_steps = int(relax_cfg.get(f"{mode}_warmup_steps", 0))
    cool_steps = int(relax_cfg.get(f"{mode}_cool_steps", 0))
    target_steps = int(relax_cfg.get(f"{mode}_target_temp_steps", 10000))
    rows = []
    metric_rows = []
    available_counter = -1
    if not seed_manifest.exists():
        rows.append({"full_pore_seed_id": "no_full_pore_seed_manifest", "status": "skipped_no_full_pore_seed_manifest", "raw_seed_extxyz_path": "", "relaxed_extxyz_path": "", "relax_is_training_data": False, "relax_is_production_md": False})
        pd.DataFrame(rows).to_csv(out, index=False)
        _write_relax_products(config, rows, metric_rows, suffix)
        write_lammps_relax_array_script(config, mode)
        return out
    seed_rows = pd.read_csv(seed_manifest).to_dict("records")
    if selected_available_index is None and mode != "tiny":
        for row in seed_rows:
            seed_id = str(row.get("full_pore_seed_id", ""))
            if not str(row.get("status", "")).startswith("available"):
                continue
            seed_extxyz = Path(str(row.get("extxyz_path", "")))
            rows.append(
                {
                    "full_pore_seed_id": seed_id,
                    "status": "planned_slurm_array_not_run",
                    "failure_reason": "",
                    "raw_seed_extxyz_path": str(seed_extxyz),
                    "relaxed_extxyz_path": "",
                    "source_stage": "",
                    "relax_is_training_data": False,
                    "relax_is_production_md": False,
                    "lammps_relax_dir": str(seed_extxyz.parent / "lammps_relax"),
                    "temperature_K": float(relax_cfg.get("temperature_K", 523.0)),
                    "initial_temperature_K": float(relax_cfg.get("initial_temperature_K", 300.0)),
                    "high_temperature_K": float(relax_cfg.get("high_temperature_K", 650.0)),
                    "timestep_fs": float(relax_cfg.get("timestep_fs", 0.5)),
                    "lammps_warmup_steps": warmup_steps,
                    "lammps_high_temp_steps": high_steps,
                    "lammps_cool_steps": cool_steps,
                    "lammps_target_temp_steps": target_steps,
                    "lammps_total_steps": warmup_steps + high_steps + cool_steps + target_steps,
                    "lammps_total_time_ps": (warmup_steps + high_steps + cool_steps + target_steps) * float(relax_cfg.get("timestep_fs", 0.5)) / 1000.0,
                    "lammps_protocol": relax_cfg.get("protocol", "fixed_silica_pore_polymer_multistage_anneal_v1"),
                    "full_pore_box_compression": bool(relax_cfg.get("full_pore_box_compression", False)),
                    "polymer_architecture": row.get("polymer_architecture", ""),
                    "pe_variant": row.get("pe_variant", ""),
                    "pp_variant": row.get("pp_variant", ""),
                    "composition": row.get("composition", ""),
                    "loading_mode": row.get("loading_mode", ""),
                    "seed": row.get("seed", ""),
                    "atom_roles_path": row.get("atom_roles_path", ""),
                }
            )
        pd.DataFrame(rows).to_csv(out, index=False)
        write_lammps_relax_array_script(config, mode)
        return out
    for row in seed_rows:
        seed_id = str(row.get("full_pore_seed_id", ""))
        if str(row.get("status", "")).startswith("available"):
            available_counter += 1
        if selected_available_index is not None and available_counter != int(selected_available_index):
            continue
        seed_extxyz = Path(str(row.get("extxyz_path", "")))
        seed_dir = seed_extxyz.parent
        relax_dir = seed_dir / "lammps_relax"
        relax_dir.mkdir(parents=True, exist_ok=True)
        template_dirs = sorted((seed_dir / "emc_chain_templates").glob("*"))
        status = "failed"
        reason = ""
        relaxed_path = ""
        try:
            if not str(row.get("status", "")).startswith("available"):
                raise RuntimeError("full_pore_seed_not_available")
            if not lmp:
                raise RuntimeError("lammps_executable_not_found")
            if not seed_extxyz.exists():
                raise RuntimeError("seed_extxyz_missing")
            if not template_dirs:
                raise RuntimeError("emc_chain_templates_missing")
            _copy_polymer_params(template_dirs, relax_dir / "polymer.params")
            n_silica, n_polymer = _write_combined_lammps_data(seed_extxyz, template_dirs, relax_dir / "full_pore_relax.data")
            _write_lammps_input(relax_dir / "in.relax", warmup_steps, high_steps, cool_steps, target_steps, relax_cfg)
            proc = subprocess.run([lmp, "-in", "in.relax"], cwd=relax_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=int(relax_cfg.get("timeout_seconds", 1800)))
            (relax_dir / "lammps_relax.log").write_text(proc.stdout, encoding="utf-8")
            if proc.returncode != 0:
                raise RuntimeError("lammps_relax_failed")
            elems, _coords, box = read_xyz_like(seed_extxyz)
            relaxed_path = str(seed_dir / "relaxed.extxyz")
            _dump_to_extxyz(relax_dir / "relaxed_snapshot.dump", Path(relaxed_path), box)
            status = "lammps_relaxed_full_pore"
            reason = ""
            metadata_path = seed_dir / "metadata.yaml"
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            metadata["relaxation"] = {
                "lammps_relax_performed": True,
                "relax_is_training_data": False,
                "relax_is_production_md": False,
                "mlff_start_structure_kind": "lammps_relaxed_full_pore",
                "mlff_start_extxyz_path": relaxed_path,
                "lammps_relax_dir": str(relax_dir),
                "temperature_K": float(relax_cfg.get("temperature_K", 523.0)),
                "initial_temperature_K": float(relax_cfg.get("initial_temperature_K", 300.0)),
                "high_temperature_K": float(relax_cfg.get("high_temperature_K", 650.0)),
                "timestep_fs": float(relax_cfg.get("timestep_fs", 0.5)),
                "lammps_warmup_steps": warmup_steps,
                "lammps_high_temp_steps": high_steps,
                "lammps_cool_steps": cool_steps,
                "lammps_target_temp_steps": target_steps,
                "lammps_total_steps": warmup_steps + high_steps + cool_steps + target_steps,
                "lammps_total_time_ps": (warmup_steps + high_steps + cool_steps + target_steps) * float(relax_cfg.get("timestep_fs", 0.5)) / 1000.0,
                "lammps_protocol": relax_cfg.get("protocol", "fixed_silica_pore_polymer_multistage_anneal_v1"),
                "full_pore_box_compression": bool(relax_cfg.get("full_pore_box_compression", False)),
                "n_silica_atoms_fixed": n_silica,
                "n_polymer_atoms_mobile": n_polymer,
            }
            metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
            metric = _relax_metrics(config, row, seed_dir, relaxed_path, n_silica, warmup_steps, high_steps, cool_steps, target_steps)
            metric_rows.append(metric)
        except Exception as exc:
            status = "failed_lammps_relax"
            reason = str(exc)
        rows.append(
            {
                "full_pore_seed_id": seed_id,
                "status": status,
                "failure_reason": reason,
                "raw_seed_extxyz_path": str(seed_extxyz),
                "relaxed_extxyz_path": relaxed_path,
                "source_stage": status if status == "lammps_relaxed_full_pore" else "",
                "relax_is_training_data": False,
                "relax_is_production_md": False,
                "lammps_relax_dir": str(relax_dir),
                "temperature_K": float(relax_cfg.get("temperature_K", 523.0)),
                "initial_temperature_K": float(relax_cfg.get("initial_temperature_K", 300.0)),
                "high_temperature_K": float(relax_cfg.get("high_temperature_K", 650.0)),
                "timestep_fs": float(relax_cfg.get("timestep_fs", 0.5)),
                "lammps_warmup_steps": warmup_steps,
                "lammps_high_temp_steps": high_steps,
                "lammps_cool_steps": cool_steps,
                "lammps_target_temp_steps": target_steps,
                "lammps_total_steps": warmup_steps + high_steps + cool_steps + target_steps,
                "lammps_total_time_ps": (warmup_steps + high_steps + cool_steps + target_steps) * float(relax_cfg.get("timestep_fs", 0.5)) / 1000.0,
                "lammps_protocol": relax_cfg.get("protocol", "fixed_silica_pore_polymer_multistage_anneal_v1"),
                "full_pore_box_compression": bool(relax_cfg.get("full_pore_box_compression", False)),
                "polymer_architecture": row.get("polymer_architecture", ""),
                "pe_variant": row.get("pe_variant", ""),
                "pp_variant": row.get("pp_variant", ""),
                "composition": row.get("composition", ""),
                "loading_mode": row.get("loading_mode", ""),
                "seed": row.get("seed", ""),
                "atom_roles_path": row.get("atom_roles_path", ""),
                "n_silica_atoms_fixed": locals().get("n_silica", ""),
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False)
    _write_relax_products(config, rows, metric_rows, suffix)
    if selected_available_index is None:
        collect_lammps_relax_manifests(config)
    write_lammps_relax_array_script(config, mode)
    return out
