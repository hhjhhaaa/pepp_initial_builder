from __future__ import annotations

from typing import Any, Dict


def run_id(config: Dict[str, Any]) -> str:
    return str(config.get("run", {}).get("run_id", "") or "").strip()


def apply_run_namespace(config: Dict[str, Any]) -> Dict[str, Any]:
    rid = run_id(config)
    if not rid or not isinstance(config.get("paths"), dict):
        return config
    data_base = f"data/runs/{rid}"
    out_base = f"outputs/runs/{rid}"
    paths = config["paths"]
    paths.update(
        {
            "porems_models_dir": f"{data_base}/mesoporous_silica/pore_models",
            "silica_patches_dir": f"{data_base}/mesoporous_silica/surface_patches",
            "surface_sites_dir": f"{data_base}/mesoporous_silica/surface_sites",
            "full_pore_seed_structures_dir": f"{data_base}/mlff_seed/structures",
            "aimd_local_structures_dir": f"{data_base}/cp2k_aimd/seed_structures",
            "cp2k_jobs_dir": f"{data_base}/cp2k_aimd/jobs",
            "cp2k_parsed_dir": f"{data_base}/cp2k_aimd/parsed",
            "aimd_dataset_dir": f"{data_base}/aimd_dataset",
            "exports_dir": f"{data_base}/exports",
            "aimd_exports_dir": f"{data_base}/exports",
            "logs_dir": f"{out_base}/logs",
            "jobs_dir": f"{out_base}/jobs",
            "figures_dir": f"{out_base}/figures",
            "aimd_seed_manifest": f"{data_base}/exports/aimd_seed_manifest.csv",
            "aimd_structure_manifest": f"{data_base}/exports/all_data_generation_manifest.csv",
            "aimd_local_manifest": f"{data_base}/cp2k_aimd/seed_structures/aimd_local_manifest.csv",
            "cp2k_structure_input_manifest": f"{data_base}/exports/cp2k_structure_input_manifest.csv",
        }
    )
    return config
