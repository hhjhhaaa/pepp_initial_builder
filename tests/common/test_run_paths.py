from pepp_initial_builder.common.run import apply_run_namespace


def test_run_namespace_uses_run_scoped_production_paths():
    cfg = {"run": {"run_id": "run_a"}, "paths": {"root": "."}}
    apply_run_namespace(cfg)
    paths = cfg["paths"]
    assert paths["porems_models_dir"] == "data/runs/run_a/mesoporous_silica/pore_models"
    assert paths["silica_patches_dir"] == "data/runs/run_a/mesoporous_silica/surface_patches"
    assert paths["surface_sites_dir"] == "data/runs/run_a/mesoporous_silica/surface_sites"
    assert paths["exports_dir"] == "data/runs/run_a/exports"
    assert paths["logs_dir"] == "outputs/runs/run_a/logs"
