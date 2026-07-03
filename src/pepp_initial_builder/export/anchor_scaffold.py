from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def _root(config_path: Path, config: Dict[str, Any]) -> Path:
    value = config.get("paths", {}).get("root", ".")
    return config_path.resolve().parents[1] if value in {None, "", "."} else Path(str(value)).expanduser()


def _anchor_dir(root: Path, config: Dict[str, Any]) -> Path:
    run_id = str(config.get("run", {}).get("run_id", "anchor_run")).strip()
    anchor_root = Path(str(config.get("paths", {}).get("anchor_root", "data/mlff_aimd_anchor")))
    base = anchor_root if anchor_root.is_absolute() else root / anchor_root
    return base / run_id


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _domain_rows(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    run_id = str(config.get("run", {}).get("run_id", "anchor_run")).strip()
    for domain, spec in config.get("domains", {}).items():
        for system in spec.get("systems", []):
            rows.append(
                {
                    "run_id": run_id,
                    "domain": domain,
                    "system_id": system,
                    "priority": spec.get("priority", ""),
                    "initial_structure_status": spec.get("initial_structure_status", "planned_not_generated"),
                    "structure_path": "",
                    "topology_path": "",
                    "builder": "",
                    "source_stage": "",
                    "eligible_for_dft_labeling": "false",
                    "eligible_for_training": "false",
                    "failure_reason": "structure_not_generated",
                    "notes": spec.get("description", ""),
                }
            )
    return rows


def _frame_manifest_headers() -> List[str]:
    return [
        "frame_id",
        "trajectory_id",
        "system_id",
        "domain",
        "composition",
        "PE_fraction",
        "PP_fraction",
        "PC_fraction",
        "chain_length",
        "degree_of_polymerization",
        "density",
        "rho_over_rho_eq",
        "temperature",
        "ensemble",
        "seed",
        "source_type",
        "DFT_code",
        "DFT_functional",
        "dispersion",
        "basis_or_cutoff",
        "pseudopotential",
        "kpoints",
        "charge",
        "spin_multiplicity",
        "time_fs",
        "frame_index",
        "energy",
        "forces_available",
        "stress_available",
        "SCF_converged",
        "geometry_valid",
        "split",
        "surface_type",
        "hydroxyl_density",
        "silica_fixed",
        "mobile_shell_radius",
        "polymer_wall_distance",
        "adsorption_state",
        "confined_geometry",
        "pore_size_or_wall_distance",
    ]


def scaffold_mlff_aimd_anchor(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = _root(config_path, config)
    anchor = _anchor_dir(root, config)
    dirs = [
        "configs",
        "initial_structures/bulk_core",
        "initial_structures/pc_extension",
        "initial_structures/silica_interface",
        "cp2k_inputs/bulk_core",
        "cp2k_inputs/conformer_torsion",
        "cp2k_inputs/perturbation",
        "cp2k_inputs/pc_extension",
        "cp2k_inputs/silica_interface",
        "raw_outputs/cp2k",
        "raw_outputs/logs",
        "raw_outputs/slurm",
        "parsed_frames/extxyz",
        "parsed_frames/ase_db",
        "parsed_frames/npz_optional",
        "manifests",
        "reports",
    ]
    for name in dirs:
        (anchor / name).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, anchor / "configs" / config_path.name)
    rows = _domain_rows(config)
    system_fields = [
        "run_id",
        "domain",
        "system_id",
        "priority",
        "initial_structure_status",
        "structure_path",
        "topology_path",
        "builder",
        "source_stage",
        "eligible_for_dft_labeling",
        "eligible_for_training",
        "failure_reason",
        "notes",
    ]
    _write_csv(anchor / "manifests" / "system_manifest.csv", rows, system_fields)
    _write_csv(anchor / "manifests" / "frame_manifest.csv", [], _frame_manifest_headers())
    _write_csv(anchor / "manifests" / "trajectory_manifest.csv", [], ["trajectory_id", "system_id", "domain", "source_type", "path", "status", "failure_reason"])
    _write_csv(anchor / "manifests" / "split_manifest.csv", [], ["frame_id", "trajectory_id", "system_id", "domain", "split", "split_reason"])
    _write_csv(anchor / "manifests" / "failed_jobs.csv", [], ["job_id", "system_id", "domain", "stage", "status", "failure_reason", "path"])
    _write_csv(anchor / "manifests" / "rejected_frames.csv", [], ["frame_id", "system_id", "domain", "rejection_reason", "path"])
    readme = anchor / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# MLFF AIMD Anchor Run: {config.get('run', {}).get('run_id', 'anchor_run')}",
                "",
                "This run namespace is for real DFT/AIMD anchor labels for one joint C/H/O/Si MLFF.",
                "Initial structures are registered before CP2K labeling. Planned rows are not training data.",
                "",
                "Rules:",
                "- no fake labels",
                "- no classical FF/xTB labels in training manifests",
                "- no random frame split across one trajectory",
                "- PC remains builder-gated until topology and LAMMPS stability checks pass",
                "- silica-interface training frames must record source_stage and surface metadata",
                "",
            ]
        ),
        encoding="utf-8",
    )
    for report in ["dataset_summary.md", "dft_convergence_report.md", "coverage_report.md", "holdout_report.md", "known_limitations.md"]:
        (anchor / "reports" / report).write_text("# Pending\n\nNo real DFT labels have been accepted in this anchor namespace yet.\n", encoding="utf-8")
    return anchor
