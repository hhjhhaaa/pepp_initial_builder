from pepp_initial_builder.pore_workflow import discover_porems, load_pore_config


def test_porems_discovery_shape():
    cfg = load_pore_config("configs/aimd_pore_builder.yaml")
    report = discover_porems(cfg)
    assert "available" in report
    assert "candidates" in report
    assert report["manual_user_input_allowed"] is True
