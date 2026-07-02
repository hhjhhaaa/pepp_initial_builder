from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

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


def _write_lammps_input(path: Path, high_steps: int, target_steps: int, cfg: Dict[str, Any]) -> None:
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

# Stage 1: high-temperature NVT anneal of the mobile polymer inside the fixed pore.
velocity polymer create {high_temp:.3f} 91023 mom yes rot yes dist gaussian
timestep {timestep:.6f}
fix int polymer nvt temp {high_temp:.3f} {high_temp:.3f} 100.0
fix mom polymer momentum 200 linear 1 1 1
dump relax_dump all custom {dump_every} relax.lammpstrj id mol type q x y z
dump_modify relax_dump sort id
run {high_steps}
unfix int

# Stage 2: cool to target temperature and continue NVT pre-equilibration before CP2K crops.
fix int polymer nvt temp {high_temp:.3f} {temp:.3f} 100.0
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


def write_lammps_relax_inputs(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_pore_dirs(config)
    root = pore_root(config)
    out = root / config["paths"]["full_pore_seed_structures_dir"] / "lammps_relax_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    seed_manifest = root / config["paths"]["full_pore_seed_structures_dir"] / "full_pore_seed_manifest.csv"
    lmp = _lammps_executable(config)
    relax_cfg = config.get("lammps_full_pore_relax", {})
    high_steps = int(relax_cfg.get(f"{mode}_high_temp_steps", 10000))
    target_steps = int(relax_cfg.get(f"{mode}_target_temp_steps", 10000))
    rows = []
    if not seed_manifest.exists():
        rows.append({"full_pore_seed_id": "no_full_pore_seed_manifest", "status": "skipped_no_full_pore_seed_manifest", "raw_seed_extxyz_path": "", "relaxed_extxyz_path": "", "relax_is_training_data": False, "relax_is_production_md": False})
        pd.DataFrame(rows).to_csv(out, index=False)
        return out
    for row in pd.read_csv(seed_manifest).to_dict("records"):
        seed_id = str(row.get("full_pore_seed_id", ""))
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
            _write_lammps_input(relax_dir / "in.relax", high_steps, target_steps, relax_cfg)
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
                "high_temperature_K": float(relax_cfg.get("high_temperature_K", 650.0)),
                "timestep_fs": float(relax_cfg.get("timestep_fs", 0.5)),
                "lammps_high_temp_steps": high_steps,
                "lammps_target_temp_steps": target_steps,
                "lammps_total_steps": high_steps + target_steps,
                "lammps_total_time_ps": (high_steps + target_steps) * float(relax_cfg.get("timestep_fs", 0.5)) / 1000.0,
                "n_silica_atoms_fixed": n_silica,
                "n_polymer_atoms_mobile": n_polymer,
            }
            metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
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
                "high_temperature_K": float(relax_cfg.get("high_temperature_K", 650.0)),
                "timestep_fs": float(relax_cfg.get("timestep_fs", 0.5)),
                "lammps_high_temp_steps": high_steps,
                "lammps_target_temp_steps": target_steps,
                "lammps_total_steps": high_steps + target_steps,
                "lammps_total_time_ps": (high_steps + target_steps) * float(relax_cfg.get("timestep_fs", 0.5)) / 1000.0,
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
