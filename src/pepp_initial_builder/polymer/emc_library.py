from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pandas as pd
import yaml

from pepp_initial_builder.common.openbabel import convert_with_obabel, obabel_executable
from pepp_initial_builder.common.paths import ensure_dirs, project_root
from pepp_initial_builder.common.tools import discover_tools

PURE_LIBRARY_RECIPES: Dict[str, Dict[str, Any]] = {
    "PE": {
        "density_g_cm3": 0.855,
        "temperature_K": 523.0,
        "groups": [
            "monomer *CC*,1,monomer:2",
            "terminator *CC,1,monomer:1,1,monomer:2",
        ],
        "cluster": "polymer random,1",
        "polymer_line": "{n_chains} monomer,{repeat_units},terminator,2",
        "basis": "Local polyethylene EMC recipe, adapted to charmm/c36a/cgenff.",
    },
    "PP": {
        "density_g_cm3": 0.85,
        "temperature_K": 523.0,
        "groups": [
            "monomer *C(C)C*, 1,monomer:2, 1,term:1, 2,term:1",
            "term *C",
        ],
        "cluster": "polymer alternate,1",
        "polymer_line": "{n_chains} monomer,{repeat_units},term,2",
        "basis": "EMC t_glass 20230726 atactic polypropylene example.",
    },
    "PS": {
        "density_g_cm3": 1.00,
        "temperature_K": 523.0,
        "groups": [
            "monomer *C(c1ccccc1)C*, 1,monomer:2, 1,term:1, 2,term:1",
            "term *C",
        ],
        "cluster": "polymer alternate,1",
        "polymer_line": "{n_chains} monomer,{repeat_units},term,2",
        "basis": "EMC t_glass 20230726 atactic polystyrene example.",
    },
}


def _component_specs(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    if row.get("components"):
        specs = []
        for item in row["components"]:
            spec = dict(item)
            spec["component"] = str(spec["component"]).upper()
            specs.append(spec)
        return specs
    return [
        {
            "component": str(row["component"]).upper(),
            "n_chains": int(row["n_chains"]),
            "repeat_units": int(row["repeat_units"]),
        }
    ]


def _component_chain_counts(row: Dict[str, Any]) -> Dict[str, int]:
    return {spec["component"]: int(spec["n_chains"]) for spec in _component_specs(row)}


def _component_repeat_units(row: Dict[str, Any]) -> Dict[str, int]:
    return {spec["component"]: int(spec["repeat_units"]) for spec in _component_specs(row)}


def _component_names(row: Dict[str, Any]) -> List[str]:
    return [spec["component"] for spec in _component_specs(row)]


def _component_recipe(component: str, prefix: str) -> Dict[str, Any]:
    if component == "PE":
        return {
            "groups": [
                f"{prefix}_monomer *CC*,1,{prefix}_monomer:2",
                f"{prefix}_terminator *CC,1,{prefix}_monomer:1,1,{prefix}_monomer:2",
            ],
            "cluster": f"{prefix}_polymer random,1",
            "polymer_line": "{n_chains} " + f"{prefix}_monomer" + ",{repeat_units}," + f"{prefix}_terminator" + ",2",
        }
    if component == "PP":
        return {
            "groups": [
                f"{prefix}_monomer *C(C)C*, 1,{prefix}_monomer:2, 1,{prefix}_term:1, 2,{prefix}_term:1",
                f"{prefix}_term *C",
            ],
            "cluster": f"{prefix}_polymer alternate,1",
            "polymer_line": "{n_chains} " + f"{prefix}_monomer" + ",{repeat_units}," + f"{prefix}_term" + ",2",
        }
    if component == "PS":
        return {
            "groups": [
                f"{prefix}_monomer *C(c1ccccc1)C*, 1,{prefix}_monomer:2, 1,{prefix}_term:1, 2,{prefix}_term:1",
                f"{prefix}_term *C",
            ],
            "cluster": f"{prefix}_polymer alternate,1",
            "polymer_line": "{n_chains} " + f"{prefix}_monomer" + ",{repeat_units}," + f"{prefix}_term" + ",2",
        }
    raise ValueError(f"Unsupported EMC library component: {component}")


def pure_library_rows(config: Dict[str, Any], mode: str = "pilot") -> List[Dict[str, Any]]:
    library = config.get("emc_library", {})
    rows = library.get(f"{mode}_systems") or library.get("systems") or []
    return [dict(row) for row in rows]


def emc_library_dir(config: Dict[str, Any]) -> Path:
    return project_root(config) / config["paths"].get("emc_library_dir", "data/polymer/emc_library")


def pure_library_row(config: Dict[str, Any], system_id: str, mode: str = "pilot") -> Dict[str, Any]:
    for row in pure_library_rows(config, mode):
        if str(row.get("system_id")) == system_id:
            return row
    raise KeyError(f"No EMC library system_id {system_id!r} in {mode} config")


def _emc_paths(config: Dict[str, Any]) -> Dict[str, str]:
    tools = discover_tools(config)
    emc = tools["emc"]
    missing = [name for name in ["executable", "emc_pl"] if not emc.get(name)]
    if missing:
        raise RuntimeError(f"EMC is required but missing: {', '.join(missing)}")
    root = emc["root"] or str(Path(config["tools"]["known_emc_root"]).expanduser())
    return {"emc": emc["executable"], "emc_pl": emc["emc_pl"], "root": root}


def _run_checked(command: List[str], cwd: Path, env: Dict[str, str], log_path: Path, timeout_s: int) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            return_code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.wait()
            raise RuntimeError(f"Command timed out after {timeout_s}s: {' '.join(command)}; see {log_path}") from exc
    if return_code != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}; see {log_path}")


def _run_captured(command: List[str], cwd: Path, env: Dict[str, str], log_path: Path, timeout_s: int) -> None:
    proc = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout_s,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}; see {log_path}")


@contextmanager
def _thermal_relax_lock(system_dir: Path) -> Iterator[Path]:
    lock_path = system_dir / ".thermal_relax.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Thermal relaxation is already running in {system_dir}") from exc
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def render_pure_recipe(row: Dict[str, Any], config: Dict[str, Any]) -> str:
    specs = _component_specs(row)
    field = row.get("force_field") or config.get("emc_library", {}).get("force_field", "charmm/c36a/cgenff")
    first_component = specs[0]["component"]
    density = float(row.get("density_g_cm3", PURE_LIBRARY_RECIPES[first_component]["density_g_cm3"]))
    temperature = float(row.get("temperature_K", PURE_LIBRARY_RECIPES[first_component]["temperature_K"]))
    groups: List[str] = []
    clusters: List[str] = []
    polymers: List[str] = []
    bases: List[str] = []
    for spec in specs:
        component = str(spec["component"]).upper()
        if component not in PURE_LIBRARY_RECIPES:
            raise ValueError(f"Unsupported EMC library component: {component}")
        prefix = component.lower()
        recipe = _component_recipe(component, prefix)
        groups.extend(recipe["groups"])
        clusters.append(recipe["cluster"])
        polymers.append(
            f"{prefix}_polymer\n"
            + recipe["polymer_line"].format(
                n_chains=int(spec["n_chains"]),
                repeat_units=int(spec["repeat_units"]),
            )
        )
        bases.append(f"{component}: {PURE_LIBRARY_RECIPES[component]['basis']}")
    component_label = "/".join(spec["component"] for spec in specs)
    return f"""#!/usr/bin/env emc.pl
# Generated by pepp_initial_builder.polymer.emc_library
# Components: {component_label}
# Recipe basis: {' | '.join(bases)}

ITEM OPTIONS
replace true
field {field}
field_increment empty
ntotal {int(row["ntotal"])}
density {density:.6f}
temperature {temperature:.3f}
pressure {float(row.get("pressure_atm", 1.0)):.6f}
build_dir .
ITEM END

ITEM GROUPS
{chr(10).join(groups)}
ITEM END

ITEM CLUSTERS
{chr(10).join(clusters)}
ITEM END

ITEM POLYMERS
{chr(10).join(polymers)}
ITEM END
"""


def _gunzip(src: Path, dst: Path) -> None:
    with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout)


def _box_from_lammps_data(path: Path) -> List[float]:
    bounds: Dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw_line.split()
        if len(parts) >= 4 and parts[2:4] == ["xlo", "xhi"]:
            bounds["x"] = float(parts[1]) - float(parts[0])
        elif len(parts) >= 4 and parts[2:4] == ["ylo", "yhi"]:
            bounds["y"] = float(parts[1]) - float(parts[0])
        elif len(parts) >= 4 and parts[2:4] == ["zlo", "zhi"]:
            bounds["z"] = float(parts[1]) - float(parts[0])
    if set(bounds) != {"x", "y", "z"}:
        raise ValueError(f"Could not parse LAMMPS box from {path}")
    return [bounds["x"], bounds["y"], bounds["z"]]


def _patch_extxyz_cell(path: Path, box: List[float]) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        return
    lines[1] = (
        f'Lattice="{box[0]} 0 0 0 {box[1]} 0 0 0 {box[2]}" '
        'Properties=species:S:1:pos:R:3 pbc="T T T"'
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _element_from_mass(mass: float) -> str:
    reference = {
        "H": 1.008,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "F": 18.998,
        "Si": 28.085,
        "P": 30.974,
        "S": 32.06,
        "Cl": 35.45,
    }
    element, reference_mass = min(reference.items(), key=lambda item: abs(item[1] - mass))
    delta = abs(reference_mass - mass)
    if delta > 0.25:
        raise ValueError(f"Could not infer element from LAMMPS mass {mass}")
    return element


def _type_elements_from_lammps_data(path: Path) -> Dict[int, str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_masses = False
    masses: Dict[int, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line == "Masses":
            in_masses = True
            continue
        if in_masses and line[0].isalpha():
            break
        if not in_masses:
            continue
        body, _, comment = line.partition("#")
        parts = body.split()
        if len(parts) < 2:
            continue
        atom_type = int(parts[0])
        mass = float(parts[1])
        commented_element = comment.strip().split()[0] if comment.strip() else ""
        masses[atom_type] = commented_element if commented_element.isalpha() else _element_from_mass(mass)
    if not masses:
        raise ValueError(f"Could not parse Masses section from {path}")
    return masses


def _rewrite_extxyz_species_from_lammps_xyz(xyz_path: Path, extxyz_path: Path, data_path: Path) -> None:
    type_elements = _type_elements_from_lammps_data(data_path)
    lines = xyz_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid XYZ file: {xyz_path}")
    natoms = int(lines[0].strip())
    atom_lines = lines[2 : 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"XYZ atom count mismatch in {xyz_path}")
    out = [str(natoms), extxyz_path.read_text(encoding="utf-8", errors="replace").splitlines()[1]]
    for raw_line in atom_lines:
        parts = raw_line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom line in {xyz_path}: {raw_line}")
        atom_type = int(parts[0])
        element = type_elements[atom_type]
        out.append(f"{element:>4} {float(parts[1]):15.8f} {float(parts[2]):15.8f} {float(parts[3]):15.8f}")
    extxyz_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _convert_pdb_outputs(system_dir: Path, config: Dict[str, Any]) -> None:
    pdb_gz = system_dir / "polymer.pdb.gz"
    pdb = system_dir / "polymer.pdb"
    if pdb_gz.exists():
        _gunzip(pdb_gz, pdb)
    if not pdb.exists():
        raise FileNotFoundError(f"Missing EMC PDB output in {system_dir}")
    configured_obabel = config.get("tools", {}).get("known_openbabel_executable")
    convert_with_obabel(pdb, system_dir / "polymer.xyz", configured_obabel)
    convert_with_obabel(pdb, system_dir / "polymer.extxyz", configured_obabel)
    _patch_extxyz_cell(system_dir / "polymer.extxyz", _box_from_lammps_data(system_dir / "polymer.data"))


def _write_relax_input(system_dir: Path, row: Dict[str, Any], config: Dict[str, Any]) -> Path:
    relax = config.get("emc_library", {}).get("thermal_relax", {})
    out = system_dir / "in.thermal_relax.lmp"
    temp = float(row.get("temperature_K", 523.0))
    anneal = float(row.get("anneal_temperature_K", relax.get("anneal_temperature_K", 650.0)))
    pressure = float(row.get("pressure_atm", 1.0))
    timestep = float(relax.get("timestep_fs", 1.0))
    tdamp = float(relax.get("tdamp_fs", 100.0))
    pdamp = float(relax.get("pdamp_fs", 1000.0))
    warmup = int(relax.get("warmup_steps", 10000))
    heat = int(relax.get("nvt_heat_steps", 50000))
    cool = int(relax.get("nvt_cool_steps", 50000))
    npt = int(relax.get("npt_steps", 200000))
    settle = int(relax.get("nvt_settle_steps", 100000))
    thermo = int(relax.get("thermo_every", 1000))
    dump = int(relax.get("dump_every", 10000))
    minimize_etol = str(relax.get("minimize_etol", "1.0e-6"))
    minimize_ftol = str(relax.get("minimize_ftol", "1.0e-8"))
    minimize_maxiter = int(relax.get("minimize_maxiter", 10000))
    minimize_maxeval = int(relax.get("minimize_maxeval", 20000))
    out.write_text(
        f"""units real
atom_style full
boundary p p p
read_data polymer.data
include polymer.params
if "${{flag_charged}} != 0" then "kspace_style pppm/cg 1.0e-4"
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
thermo {thermo}
thermo_style custom step temp density press pe ke etotal vol lx ly lz
thermo_modify lost error flush yes

min_style fire
minimize {minimize_etol} {minimize_ftol} {minimize_maxiter} {minimize_maxeval}

velocity all create {temp:.3f} 4928459 mom yes rot yes dist gaussian
timestep {timestep:.6f}
fix temp all langevin {temp:.3f} {temp:.3f} {tdamp:.3f} 91023
fix int all nve/limit 0.1
run {warmup}
unfix int
unfix temp

fix mom all momentum 200 linear 1 1 1 angular
fix int all nvt temp {temp:.3f} {anneal:.3f} {tdamp:.3f}
run {heat}
unfix int
fix int all nvt temp {anneal:.3f} {temp:.3f} {tdamp:.3f}
run {cool}
unfix int
fix int all npt temp {temp:.3f} {temp:.3f} {tdamp:.3f} iso {pressure:.6f} {pressure:.6f} {pdamp:.3f}
run {npt}
unfix int
fix int all nvt temp {temp:.3f} {temp:.3f} {tdamp:.3f}
dump traj all custom {dump} thermal_relax.lammpstrj id mol type q x y z
dump_modify traj sort id
run {settle}
unfix int
unfix mom
write_data relaxed.data nocoeff
write_dump all xyz relaxed.xyz modify sort id
""",
        encoding="utf-8",
    )
    return out


def _run_thermal_relax(system_dir: Path, row: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    tools = discover_tools(config)
    lmp = config.get("tools", {}).get("known_lammps_executable") or tools["lammps"]["executable"]
    if not lmp:
        raise RuntimeError("LAMMPS executable not found for EMC library thermal relaxation")
    script = _write_relax_input(system_dir, row, config)
    timeout = int(config.get("emc_library", {}).get("thermal_relax", {}).get("timeout_seconds", 21600))
    _run_checked(
        [str(lmp), "-log", "thermal_relax.lammps.log", "-in", script.name],
        system_dir,
        os.environ.copy(),
        system_dir / "thermal_relax.log",
        timeout,
    )
    configured_obabel = config.get("tools", {}).get("known_openbabel_executable")
    exe = obabel_executable(configured_obabel)
    if not exe:
        raise RuntimeError("Open Babel executable 'obabel' was not found")
    proc = subprocess.run(
        [exe, "-ixyz", str(system_dir / "relaxed.xyz"), "-oexyz", "-O", str(system_dir / "relaxed.extxyz")],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (system_dir / "thermal_relax_obabel.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0 or not (system_dir / "relaxed.extxyz").exists():
        raise RuntimeError(f"Open Babel conversion failed for relaxed.xyz: {proc.stdout.strip()}")
    _rewrite_extxyz_species_from_lammps_xyz(
        system_dir / "relaxed.xyz",
        system_dir / "relaxed.extxyz",
        system_dir / "relaxed.data",
    )
    _patch_extxyz_cell(system_dir / "relaxed.extxyz", _box_from_lammps_data(system_dir / "relaxed.data"))
    return {
        "lammps_thermal_relax_performed": True,
        "relax_is_training_data": False,
        "relax_is_production_md": False,
        "relax_protocol": config.get("emc_library", {}).get("thermal_relax", {}).get(
            "protocol",
            "emc_polymer_lammps_thermal_relax_v1",
        ),
        "thermal_relax_input": str(script),
        "thermal_relax_log": str(system_dir / "thermal_relax.log"),
        "thermal_relax_lammps_log": str(system_dir / "thermal_relax.lammps.log"),
        "relaxed_lammps_data": str(system_dir / "relaxed.data"),
        "relaxed_extxyz": str(system_dir / "relaxed.extxyz"),
    }


def _default_relaxation() -> Dict[str, Any]:
    return {
        "lammps_thermal_relax_performed": False,
        "relax_is_training_data": False,
        "relax_is_production_md": False,
    }


def _metadata_for_system(
    config: Dict[str, Any],
    row: Dict[str, Any],
    system_dir: Path,
    relaxation: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    relaxation = relaxation or _default_relaxation()
    recipe = system_dir / "polymer.esh"

    components = _component_names(row)
    first_component = components[0]
    task_lane = str(row.get("task_lane") or config.get("emc_library", {}).get("task_lane", "mlff_direct"))
    if task_lane != "mlff_direct":
        raise ValueError("Pure EMC polymer library currently builds only the mlff_direct structure lane")
    return {
        "system_id": row["system_id"],
        "components": components,
        "n_chains": int(sum(_component_chain_counts(row).values())),
        "component_chain_counts": _component_chain_counts(row),
        "component_chain_counts_arg": ",".join(f"{name}:{count}" for name, count in _component_chain_counts(row).items()),
        "repeat_units": _component_repeat_units(row) if len(components) > 1 else int(_component_specs(row)[0]["repeat_units"]),
        "target_density_g_cm3": float(row.get("density_g_cm3", PURE_LIBRARY_RECIPES[first_component]["density_g_cm3"])),
        "target_temperature_K": float(row.get("temperature_K", PURE_LIBRARY_RECIPES[first_component]["temperature_K"])),
        "builder": {
            "builder_used": "emc",
            "emc_success": True,
            "coordinate_source": "emc",
            "topology_source": "emc",
            "force_field": row.get("force_field") or config.get("emc_library", {}).get("force_field", "charmm/c36a/cgenff"),
            "recipe_basis": {component: PURE_LIBRARY_RECIPES[component]["basis"] for component in components},
        },
        "structure_task": {
            "lane": task_lane,
            "intended_runner": config.get("emc_library", {}).get("intended_runner", "zero_shot_mace_mh0"),
            "allowed_consumers": ["pepp_mlff_finetune"],
            "not_for_aimd": True,
            "not_for_fine_tuned_mlff_without_reselection": True,
        },
        "paths": {
            "emc_recipe": str(recipe),
            "lammps_data": str(system_dir / "polymer.data"),
            "params": str(system_dir / "polymer.params"),
            "pdb": str(system_dir / "polymer.pdb"),
            "xyz": str(system_dir / "polymer.xyz"),
            "extxyz": str(system_dir / "polymer.extxyz"),
            "mlff_start_extxyz": relaxation.get("relaxed_extxyz", str(system_dir / "polymer.extxyz")),
            "mlff_start_lammps_data": relaxation.get("relaxed_lammps_data", str(system_dir / "polymer.data")),
        },
        "relaxation": relaxation,
        "label_policy": {
            "classical_relaxation_is_label_source": False,
            "zero_shot_mace_outputs_are": config.get("emc_library", {}).get("output_label_status", "provisional MLFF labels"),
        },
        "status": "available_relaxed"
        if relaxation.get("lammps_thermal_relax_performed") is True
        else "available_emc_built",
    }


def _write_metadata(
    config: Dict[str, Any],
    row: Dict[str, Any],
    system_dir: Path,
    relaxation: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metadata = _metadata_for_system(config, row, system_dir, relaxation)
    (system_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return metadata


def metadata_is_relaxed(system_dir: Path) -> bool:
    meta_path = system_dir / "metadata.yaml"
    if not meta_path.exists():
        return False
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    relaxed_extxyz = metadata.get("paths", {}).get("mlff_start_extxyz")
    relaxed_data = metadata.get("paths", {}).get("mlff_start_lammps_data")
    return (
        metadata.get("status") == "available_relaxed"
        and metadata.get("relaxation", {}).get("lammps_thermal_relax_performed") is True
        and bool(relaxed_extxyz)
        and bool(relaxed_data)
        and Path(str(relaxed_extxyz)).exists()
        and Path(str(relaxed_data)).exists()
    )


def _update_manifest_row(config: Dict[str, Any], mode: str, metadata: Dict[str, Any]) -> Path:
    library_dir = emc_library_dir(config)
    manifest = library_dir / f"emc_library_manifest_{mode}.csv"
    row = {
        "system_id": metadata["system_id"],
        "task_lane": metadata.get("structure_task", {}).get("lane", ""),
        "status": "available",
        "system_dir": str((library_dir / metadata["system_id"]).resolve()),
        "metadata_yaml_path": str((library_dir / metadata["system_id"] / "metadata.yaml").resolve()),
        "mlff_start_extxyz_path": metadata.get("paths", {}).get("mlff_start_extxyz", ""),
        "failure_reason": "",
    }
    if manifest.exists():
        frame = pd.read_csv(manifest)
        frame = frame[frame["system_id"] != metadata["system_id"]]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame.to_csv(manifest, index=False)
    return manifest


def build_pure_library_system(config: Dict[str, Any], row: Dict[str, Any], run_relax: bool = False) -> Path:
    paths = _emc_paths(config)
    library_dir = emc_library_dir(config)
    system_dir = library_dir / str(row["system_id"])
    system_dir.mkdir(parents=True, exist_ok=True)
    recipe = system_dir / "polymer.esh"
    recipe.write_text(render_pure_recipe(row, config), encoding="utf-8")
    env = os.environ.copy()
    env.update({"EMC_ROOT": paths["root"], "PATH": f"{Path(paths['root']) / 'scripts'}:{Path(paths['root']) / 'bin'}:{env.get('PATH', '')}"})
    timeout_setup = int(config.get("emc", {}).get("attempt_timeout_seconds", 300))
    timeout_build = int(config.get("emc", {}).get("build_timeout_seconds", 900))
    _run_captured([paths["emc_pl"], "-replace", "polymer"], system_dir, env, system_dir / "emc_setup.log", timeout_setup)
    _run_captured([paths["emc"], "build.emc"], system_dir, env, system_dir / "emc_build.log", timeout_build)
    for required in ["polymer.data", "polymer.params", "polymer.pdb.gz"]:
        if not (system_dir / required).exists():
            raise FileNotFoundError(f"EMC build did not produce {required} in {system_dir}")
    _convert_pdb_outputs(system_dir, config)

    relaxation = _default_relaxation()
    if run_relax:
        relaxation = _run_thermal_relax(system_dir, row, config)

    _write_metadata(config, row, system_dir, relaxation)
    return system_dir


def run_pure_library_relax_system(config: Dict[str, Any], system_id: str, mode: str = "pilot", force: bool = False) -> Path:
    row = pure_library_row(config, system_id, mode)
    system_dir = emc_library_dir(config) / system_id
    for required in ["polymer.data", "polymer.params", "polymer.extxyz", "metadata.yaml"]:
        if not (system_dir / required).exists():
            raise FileNotFoundError(f"Missing {required} in existing EMC library system {system_dir}")
    if metadata_is_relaxed(system_dir) and not force:
        metadata = yaml.safe_load((system_dir / "metadata.yaml").read_text(encoding="utf-8")) or {}
        _update_manifest_row(config, mode, metadata)
        return system_dir
    with _thermal_relax_lock(system_dir):
        if metadata_is_relaxed(system_dir) and not force:
            metadata = yaml.safe_load((system_dir / "metadata.yaml").read_text(encoding="utf-8")) or {}
            _update_manifest_row(config, mode, metadata)
            return system_dir
        relaxation = _run_thermal_relax(system_dir, row, config)
        metadata = _write_metadata(config, row, system_dir, relaxation)
    _update_manifest_row(config, mode, metadata)
    return system_dir


def run_pure_library_relax(
    config: Dict[str, Any],
    mode: str = "pilot",
    system_ids: List[str] | None = None,
    max_systems: int | None = None,
    force: bool = False,
) -> Path:
    ensure_dirs(config)
    selected = system_ids or [str(row["system_id"]) for row in pure_library_rows(config, mode)]
    if max_systems is not None:
        selected = selected[:max_systems]
    rows = []
    for system_id in selected:
        try:
            path = run_pure_library_relax_system(config, system_id, mode, force)
            metadata = yaml.safe_load((path / "metadata.yaml").read_text(encoding="utf-8")) or {}
            rows.append({
                "system_id": system_id,
                "task_lane": metadata.get("structure_task", {}).get("lane", ""),
                "status": "available",
                "system_dir": str(path),
                "metadata_yaml_path": str(path / "metadata.yaml"),
                "mlff_start_extxyz_path": metadata.get("paths", {}).get("mlff_start_extxyz", ""),
                "failure_reason": "",
            })
        except Exception as exc:
            rows.append({
                "system_id": system_id,
                "task_lane": config.get("emc_library", {}).get("task_lane", "mlff_direct"),
                "status": "failed",
                "system_dir": "",
                "metadata_yaml_path": "",
                "mlff_start_extxyz_path": "",
                "failure_reason": str(exc),
            })
    manifest = emc_library_dir(config) / f"emc_library_relax_manifest_{mode}.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def build_pure_library(config: Dict[str, Any], mode: str = "pilot", run_relax: bool = False, max_systems: int | None = None) -> Path:
    ensure_dirs(config)
    library_dir = emc_library_dir(config)
    library_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    selected_rows = pure_library_rows(config, mode)
    if max_systems is not None:
        selected_rows = selected_rows[:max_systems]
    for row in selected_rows:
        try:
            path = build_pure_library_system(config, row, run_relax=run_relax)
            rows.append({
                "system_id": row["system_id"],
                "task_lane": row.get("task_lane", config.get("emc_library", {}).get("task_lane", "mlff_direct")),
                "status": "available",
                "system_dir": str(path),
                "metadata_yaml_path": str(path / "metadata.yaml"),
                "mlff_start_extxyz_path": str(path / ("relaxed.extxyz" if run_relax else "polymer.extxyz")),
                "failure_reason": "",
            })
        except Exception as exc:
            rows.append({
                "system_id": row.get("system_id", ""),
                "task_lane": row.get("task_lane", config.get("emc_library", {}).get("task_lane", "mlff_direct")),
                "status": "failed",
                "system_dir": "",
                "metadata_yaml_path": "",
                "mlff_start_extxyz_path": "",
                "failure_reason": str(exc),
            })
    manifest = library_dir / f"emc_library_manifest_{mode}.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest
