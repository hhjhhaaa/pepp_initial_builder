from pathlib import Path

from pepp_initial_builder.pore.config import load_pore_config
from pepp_initial_builder.pore.porems_builder import build_porems_pores


def tmp_cfg(tmp_path):
    cfg = load_pore_config("configs/pore.yaml")
    cfg["paths"]["root"] = str(tmp_path)
    return cfg


def test_porems_unavailable_does_not_fake_model(tmp_path):
    cfg = tmp_cfg(tmp_path)
    cfg["porems"]["enabled"] = False
    cfg["tools"]["porems_search_roots"] = [str(tmp_path / "none")]
    cfg["tools"]["porems_path_hint"] = str(tmp_path / "none")
    manifest = build_porems_pores(cfg, "tiny")
    text = manifest.read_text()
    assert "failed_porems_not_available" in text
    assert "manual_user_input" not in text


def test_porems_model_with_no_surface_h_can_be_available(tmp_path, monkeypatch):
    cfg = tmp_cfg(tmp_path)
    cfg['porems'].update({
        'surface': 'bare_porems_sio2',
        'require_explicit_surface_H': False,
        'pore_diameter_nm_candidates': [4.0],
        'pore_length_nm_candidates': [6.0],
        'hydroxylation_modes': ['bare_porems_sio2'],
        'max_pore_models_tiny': 1,
    })

    def fake_discover(_cfg):
        return {'available': True, 'python_executable': '/fake/python', 'version': 'test'}

    def fake_run(_python, _model_dir, *, pore_diameter_nm, pore_length_nm, hydroxylation_mode):
        return ['Si', 'O'], [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]], (20.0, 20.0, 60.0), 'porems_build.log'

    monkeypatch.setattr('pepp_initial_builder.pore.porems_builder.discover_porems', fake_discover)
    monkeypatch.setattr('pepp_initial_builder.pore.porems_builder.run_porems_external', fake_run)
    manifest = build_porems_pores(cfg, 'tiny')
    text = manifest.read_text(encoding='utf-8')
    assert 'available' in text
    assert 'failed_validation' not in text
