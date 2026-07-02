from pathlib import Path

from pepp_initial_builder.pore_workflow import build_porems_pores, load_pore_config


def tmp_cfg(tmp_path):
    cfg = load_pore_config("configs/aimd_pore_builder.yaml")
    cfg["paths"]["root"] = str(tmp_path)
    return cfg


def test_porems_unavailable_does_not_fake_model(tmp_path):
    cfg = tmp_cfg(tmp_path)
    cfg["tools"]["porems_search_roots"] = [str(tmp_path / "none")]
    cfg["tools"]["porems_path_hint"] = str(tmp_path / "none")
    manifest = build_porems_pores(cfg, "tiny")
    text = manifest.read_text()
    assert "failed_porems_not_available" in text
    assert "manual_user_input" not in text
