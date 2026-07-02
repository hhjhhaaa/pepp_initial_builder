import yaml

from pepp_initial_builder.polymer.matrix import matrix_rows


def test_polymer_matrix_module_uses_configured_tiny_mode():
    config = yaml.safe_load(open("configs/polymer.yaml", "r", encoding="utf-8"))
    rows = matrix_rows(config, "tiny")
    assert rows
    assert rows[0]["builder_status"] == "pending"
