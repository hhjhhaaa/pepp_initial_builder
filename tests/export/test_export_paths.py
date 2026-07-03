import pandas as pd

from pepp_initial_builder.export.all_manifests import export_all_manifests
from pepp_initial_builder.export.summary import summarize_outputs


def test_export_and_summary_respect_configured_run_paths(tmp_path):
    cfg = {
        "paths": {
            "root": str(tmp_path),
            "exports_dir": "data/runs/run_a/exports",
            "logs_dir": "outputs/runs/run_a/logs",
            "figures_dir": "outputs/runs/run_a/figures",
            "jobs_dir": "outputs/runs/run_a/jobs",
            "bulk_initial_manifest": "data/runs/run_a/exports/mlff_start_manifest.csv",
            "porems_models_dir": "data/runs/run_a/mesoporous_silica/pore_models",
            "silica_patches_dir": "data/runs/run_a/mesoporous_silica/surface_patches",
            "full_pore_seed_structures_dir": "data/runs/run_a/mlff_seed/structures",
            "aimd_local_structures_dir": "data/runs/run_a/cp2k_aimd/seed_structures",
        }
    }
    root = tmp_path
    for rel in [
        "data/runs/run_a/exports/mlff_start_manifest.csv",
        "data/runs/run_a/mesoporous_silica/pore_models/pore_model_manifest.csv",
        "data/runs/run_a/mesoporous_silica/surface_patches/silica_patch_manifest.csv",
        "data/runs/run_a/mlff_seed/structures/full_pore_seed_manifest.csv",
        "data/runs/run_a/cp2k_aimd/seed_structures/aimd_local_manifest.csv",
        "outputs/runs/run_a/jobs/cp2k_hpc_job_manifest.csv",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"status": "available"}]).to_csv(path, index=False)
    (root / "outputs/runs/run_a/logs").mkdir(parents=True)
    (root / "outputs/runs/run_a/logs/run_report.txt").write_text("ok\n", encoding="utf-8")
    (root / "outputs/runs/run_a/logs/run_validation.csv").write_text("ok\n", encoding="utf-8")
    (root / "outputs/runs/run_a/figures").mkdir(parents=True)
    (root / "outputs/runs/run_a/figures/plot.png").write_text("png\n", encoding="utf-8")

    combined, summary = export_all_manifests(cfg)
    assert combined == root / "data/runs/run_a/exports/all_data_generation_manifest.csv"
    assert summary == root / "data/runs/run_a/exports/all_data_generation_manifest.json"
    assert not (root / "data/exports/all_data_generation_manifest.csv").exists()

    summary_path = summarize_outputs(cfg)
    assert summary_path == root / "outputs/runs/run_a/logs/data_generation_summary.json"
    assert not (root / "outputs/logs/data_generation_summary.json").exists()
