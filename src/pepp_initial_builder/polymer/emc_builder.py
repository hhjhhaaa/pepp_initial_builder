from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

from pepp_initial_builder.common.openbabel import convert_with_obabel
from pepp_initial_builder.common.paths import ensure_dirs, project_root
from pepp_initial_builder.common.tools import discover_tools
from pepp_initial_builder.polymer.matrix import matrix_rows


def select_rows(config: Dict[str, Any], tiny: bool = False, pilot: bool = False, max_systems: int | None = None):
    rows = matrix_rows(config, "tiny" if tiny else "pilot" if pilot else "matrix")
    return rows[:max_systems] if max_systems is not None else rows


def _emc_env(paths: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    conda_bin = Path(sys.executable).resolve().parent
    emc_scripts = Path(paths["root"]) / "scripts"
    emc_bin = Path(paths["root"]) / "bin"
    env.update({"EMC_ROOT": paths["root"], "PATH": f"{conda_bin}:{emc_scripts}:{emc_bin}:{env.get('PATH', '')}"})
    return env


def _emc_paths(config: Dict[str, Any]) -> Dict[str, str]:
    tools = discover_tools(config)
    emc = tools["emc"]
    missing = [name for name in ["executable", "emc_pl"] if not emc.get(name)]
    if missing:
        raise RuntimeError(f"EMC is required but missing: {', '.join(missing)}")
    return {"emc": emc["executable"], "emc_pl": emc["emc_pl"], "root": emc["root"] or str(Path(config["tools"]["known_emc_root"]).expanduser())}


def _recipe_text(row: Dict[str, Any], config: Dict[str, Any]) -> str:
    n = int(row["chain_length_backbone"])
    pe = int(row["n_pe_chains"])
    pp = int(row["n_pp_chains"])
    ps = int(row.get("n_ps_chains", 0))
    if pe + pp + ps < 1:
        raise RuntimeError("EMC recipe requires at least one PE, PP, or PS chain")
    groups: List[str] = []
    clusters: List[str] = []
    polymers: List[str] = []
    if pe:
        groups += ["ethyl *CC*,1,ethyl:2", "methyl *C,1,ethyl:1,1,ethyl:2"]
        clusters.append("pe_poly alternate 1")
        polymers.append(f"pe_poly\n{max(pe, 1)} ethyl,{max(n // 2, 1)},methyl,2")
    if pp:
        groups += ["pp_monomer *C(C)C*,1,pp_monomer:2,1,pp_term:1,2,pp_term:1", "pp_term *C"]
        clusters.append("pp_poly alternate 1")
        polymers.append(f"pp_poly\n{max(pp, 1)} pp_monomer,{max(n // 2, 1)},pp_term,2")
    if ps:
        groups += ["ps_monomer *C(c1ccccc1)C*,1,ps_monomer:2,1,ps_term:1,2,ps_term:1", "ps_term *C"]
        clusters.append("ps_poly alternate 1")
        polymers.append(f"ps_poly\n{max(ps, 1)} ps_monomer,{max(n // 2, 1)},ps_term,2")
    return f"""#!/usr/bin/env emc.pl
ITEM OPTIONS
replace true
field {config.get("emc", {}).get("force_field", "pcff")}
density {float(row["initial_packing_density_g_cm3"]):.6f}
ntotal {max(int(row["estimated_total_atoms"]), 50)}
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


def write_emc_chain_template(config: Dict[str, Any], chain_type: str, chain_length: int, seed: int, outdir: str | Path) -> Dict[str, str]:
    if chain_type not in {"PE", "PP", "PS"}:
        raise RuntimeError(f"Unsupported EMC chain_type: {chain_type}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    row = {
        "system_id": f"{chain_type.lower()}_chain_N{chain_length}_seed{seed}",
        "chain_length_backbone": int(chain_length),
        "seed": int(seed),
        "n_pe_chains": 1 if chain_type == "PE" else 0,
        "n_pp_chains": 1 if chain_type == "PP" else 0,
        "n_ps_chains": 1 if chain_type == "PS" else 0,
        "estimated_total_atoms": max(50, 3 * int(chain_length) + 20 if chain_type == "PE" else int(4.5 * int(chain_length)) + 20 if chain_type == "PP" else int(8.0 * int(chain_length)) + 20),
        "initial_packing_density_g_cm3": float(config.get("density", {}).get("initial_packing_density_g_cm3", 0.85)),
    }
    paths = _emc_paths(config)
    recipe = outdir / "polymer.esh"
    recipe.write_text(_recipe_text(row, config), encoding="utf-8")
    env = os.environ.copy()
    env = _emc_env(paths)
    setup = subprocess.run([paths["emc_pl"], f"-ntotal={row['estimated_total_atoms']}", f"-field={config.get('emc', {}).get('force_field', 'pcff')}", "-replace", "polymer"], cwd=outdir, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=int(config.get("emc", {}).get("attempt_timeout_seconds", 300)))
    (outdir / "emc_setup.log").write_text(setup.stdout, encoding="utf-8")
    if setup.returncode != 0:
        raise RuntimeError(f"EMC setup failed for {chain_type} chain")
    build = subprocess.run([paths["emc"], "build.emc"], cwd=outdir, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=int(config.get("emc", {}).get("build_timeout_seconds", 900)))
    (outdir / "emc_build.log").write_text(build.stdout, encoding="utf-8")
    if build.returncode != 0 or not (outdir / "polymer.data").exists():
        raise RuntimeError(f"EMC build failed for {chain_type} chain")
    if (outdir / "polymer.pdb.gz").exists():
        _gunzip(outdir / "polymer.pdb.gz", outdir / "polymer.pdb")
    convert_with_obabel(outdir / "polymer.pdb", outdir / "polymer.xyz", config.get("tools", {}).get("known_openbabel_executable"))
    return {"pdb": str(outdir / "polymer.pdb"), "xyz": str(outdir / "polymer.xyz"), "data": str(outdir / "polymer.data")}



def _estimate_atoms_for_components(component_counts: Dict[str, int], chain_length: int) -> int:
    per_chain = {"PE": 3 * int(chain_length) + 20, "PP": int(4.5 * int(chain_length)) + 20, "PS": int(8.0 * int(chain_length)) + 20}
    return max(50, sum(per_chain[str(component).upper()] * int(count) for component, count in component_counts.items() if int(count) > 0))


def write_emc_mixed_template(config: Dict[str, Any], component_counts: Dict[str, int], chain_length: int, seed: int, outdir: str | Path) -> Dict[str, str]:
    counts = {str(component).upper(): int(count) for component, count in component_counts.items() if int(count) > 0}
    unsupported = sorted(set(counts) - {"PE", "PP", "PS"})
    if unsupported:
        raise RuntimeError(f"Unsupported EMC mixed chain types: {unsupported}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    row = {
        "system_id": "mixed_" + "_".join(f"{component}{count}" for component, count in sorted(counts.items())) + f"_N{chain_length}_seed{seed}",
        "chain_length_backbone": int(chain_length),
        "seed": int(seed),
        "n_pe_chains": counts.get("PE", 0),
        "n_pp_chains": counts.get("PP", 0),
        "n_ps_chains": counts.get("PS", 0),
        "estimated_total_atoms": _estimate_atoms_for_components(counts, int(chain_length)),
        "initial_packing_density_g_cm3": float(config.get("density", {}).get("initial_packing_density_g_cm3", 0.85)),
    }
    paths = _emc_paths(config)
    recipe = outdir / "polymer.esh"
    recipe.write_text(_recipe_text(row, config), encoding="utf-8")
    env = _emc_env(paths)
    setup = subprocess.run([paths["emc_pl"], f"-ntotal={row['estimated_total_atoms']}", f"-field={config.get('emc', {}).get('force_field', 'pcff')}", "-replace", "polymer"], cwd=outdir, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=int(config.get("emc", {}).get("attempt_timeout_seconds", 300)))
    (outdir / "emc_setup.log").write_text(setup.stdout, encoding="utf-8")
    if setup.returncode != 0:
        raise RuntimeError("EMC setup failed for mixed polymer template")
    build = subprocess.run([paths["emc"], "build.emc"], cwd=outdir, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=int(config.get("emc", {}).get("build_timeout_seconds", 900)))
    (outdir / "emc_build.log").write_text(build.stdout, encoding="utf-8")
    if build.returncode != 0 or not (outdir / "polymer.params").exists() or not (outdir / "polymer.data").exists():
        raise RuntimeError("EMC build failed for mixed polymer template")
    if (outdir / "polymer.pdb.gz").exists():
        _gunzip(outdir / "polymer.pdb.gz", outdir / "polymer.pdb")
    if (outdir / "polymer.pdb").exists():
        convert_with_obabel(outdir / "polymer.pdb", outdir / "polymer.xyz", config.get("tools", {}).get("known_openbabel_executable"))
    return {"pdb": str(outdir / "polymer.pdb"), "xyz": str(outdir / "polymer.xyz"), "data": str(outdir / "polymer.data"), "params": str(outdir / "polymer.params")}


def _gunzip(src: Path, dst: Path) -> None:
    with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout)


def build_system(config: Dict[str, Any], row: Dict[str, Any]) -> Path:
    paths = _emc_paths(config)
    system_dir = project_root(config) / config["paths"]["systems_dir"] / row["system_id"]
    system_dir.mkdir(parents=True, exist_ok=True)
    recipe = system_dir / "polymer.esh"
    recipe.write_text(_recipe_text(row, config), encoding="utf-8")
    env = os.environ.copy()
    env = _emc_env(paths)
    setup = subprocess.run([paths["emc_pl"], f"-ntotal={max(int(row['estimated_total_atoms']), 50)}", f"-field={config.get('emc', {}).get('force_field', 'pcff')}", "-replace", "polymer"], cwd=system_dir, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=int(config.get("emc", {}).get("attempt_timeout_seconds", 300)))
    (system_dir / "emc_setup.log").write_text(setup.stdout, encoding="utf-8")
    if setup.returncode != 0:
        raise RuntimeError(f"EMC setup failed for {row['system_id']}")
    build = subprocess.run([paths["emc"], "build.emc"], cwd=system_dir, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=int(config.get("emc", {}).get("build_timeout_seconds", 900)))
    (system_dir / "emc_build.log").write_text(build.stdout, encoding="utf-8")
    if build.returncode != 0 or not (system_dir / "polymer.data").exists():
        raise RuntimeError(f"EMC build failed for {row['system_id']}")
    if (system_dir / "polymer.pdb.gz").exists():
        _gunzip(system_dir / "polymer.pdb.gz", system_dir / "polymer.pdb")
    convert_with_obabel(system_dir / "polymer.pdb", system_dir / "polymer.xyz", config.get("tools", {}).get("known_openbabel_executable"))
    convert_with_obabel(system_dir / "polymer.pdb", system_dir / "polymer.extxyz", config.get("tools", {}).get("known_openbabel_executable"))
    metadata = {
        "system_id": row["system_id"],
        "builder": {"builder_used": "emc", "force_field": config.get("emc", {}).get("force_field", "pcff"), "coordinate_source": "emc", "topology_source": "emc"},
        "composition": {"pe_fraction_actual": row["pe_fraction_actual"], "pp_fraction_actual": row["pp_fraction_actual"]},
        "paths": {"emc_recipe": str(recipe), "lammps_data": str(system_dir / "polymer.data"), "pdb": str(system_dir / "polymer.pdb"), "xyz": str(system_dir / "polymer.xyz"), "extxyz": str(system_dir / "polymer.extxyz")},
        "status": "available",
    }
    (system_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    return system_dir


def build_systems(config: Dict[str, Any], tiny: bool = False, pilot: bool = False, max_systems: int | None = None):
    ensure_dirs(config)
    rows: List[Dict[str, Any]] = []
    outputs = []
    for row in select_rows(config, tiny, pilot, max_systems):
        try:
            path = build_system(config, row)
            outputs.append(path)
            rows.append({"system_id": row["system_id"], "status": "available", "system_dir": str(path), "failure_reason": ""})
        except Exception as exc:
            rows.append({"system_id": row["system_id"], "status": "failed_emc", "system_dir": "", "failure_reason": str(exc)})
    out = project_root(config) / config["paths"]["systems_dir"] / "polymer_emc_manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return outputs
