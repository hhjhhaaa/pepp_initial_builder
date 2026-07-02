from pepp_initial_builder.common.config import load_config
from pepp_initial_builder.polymer.emc_builder import _recipe_text


def test_emc_recipe_uses_pcff_and_no_python_builder():
    cfg = load_config("configs/polymer.yaml")
    row = {
        "system_id": "pe",
        "chain_length_backbone": 40,
        "n_pe_chains": 1,
        "n_pp_chains": 0,
        "estimated_total_atoms": 200,
        "initial_packing_density_g_cm3": 0.85,
    }
    text = _recipe_text(row, cfg)
    assert "field pcff" in text
    assert "*CC*" in text
    assert "random" not in text.lower()


def test_emc_recipe_supports_pp():
    cfg = load_config("configs/polymer.yaml")
    row = {
        "system_id": "pp",
        "chain_length_backbone": 40,
        "n_pe_chains": 0,
        "n_pp_chains": 1,
        "estimated_total_atoms": 220,
        "initial_packing_density_g_cm3": 0.85,
    }
    assert "*C(C)C*" in _recipe_text(row, cfg)
