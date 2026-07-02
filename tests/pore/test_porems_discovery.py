from pepp_initial_builder.pore.config import load_pore_config
from pepp_initial_builder.pore.porems_discovery import discover_porems


def test_porems_discovery_shape():
    cfg = load_pore_config("configs/pore.yaml")
    report = discover_porems(cfg)
    assert "available" in report
    assert "candidates" in report
    assert report["manual_user_input_allowed"] is True
