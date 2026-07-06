from pathlib import Path

from pepp_initial_builder.common.config import load_config
from pepp_initial_builder.common.openbabel import _obabel_format
from pepp_initial_builder.polymer.emc_builder import _recipe_text
from pepp_initial_builder.polymer.emc_library import (
    _metadata_for_system,
    metadata_is_relaxed,
    pure_library_row,
    render_pure_recipe,
)


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


def test_emc_library_recipes_cover_pe_pp_ps():
    cfg = load_config("configs/polymer.yaml")
    base = {"system_id": "x", "n_chains": 2, "repeat_units": 12, "ntotal": 500}
    assert "*CC*" in render_pure_recipe({**base, "component": "PE"}, cfg)
    assert "*C(C)C*" in render_pure_recipe({**base, "component": "PP"}, cfg)
    assert "*C(c1ccccc1)C*" in render_pure_recipe({**base, "component": "PS"}, cfg)


def test_openbabel_format_mapping_supports_extxyz_and_pdb_gz():
    assert _obabel_format(Path("polymer.extxyz")) == "exyz"
    assert _obabel_format(Path("polymer.pdb.gz")) == "pdb"


def test_emc_library_can_select_system_by_id():
    cfg = load_config("configs/polymer.yaml")
    row = pure_library_row(cfg, "PS100_N12_C16_emc_seed1", "pilot")
    assert row["component"] == "PS"


def test_relaxed_metadata_points_mlff_start_to_relaxed_structure(tmp_path: Path):
    cfg = load_config("configs/polymer.yaml")
    row = {
        "system_id": "PE_test",
        "component": "PE",
        "n_chains": 1,
        "repeat_units": 2,
        "density_g_cm3": 0.855,
        "temperature_K": 523.0,
        "ntotal": 20,
    }
    relaxation = {
        "lammps_thermal_relax_performed": True,
        "relaxed_extxyz": str(tmp_path / "relaxed.extxyz"),
        "relaxed_lammps_data": str(tmp_path / "relaxed.data"),
    }
    metadata = _metadata_for_system(cfg, row, tmp_path, relaxation)
    assert metadata["status"] == "available_relaxed"
    assert metadata["paths"]["mlff_start_extxyz"].endswith("relaxed.extxyz")


def test_metadata_is_relaxed_requires_files(tmp_path: Path):
    relaxed_extxyz = tmp_path / "relaxed.extxyz"
    relaxed_data = tmp_path / "relaxed.data"
    (tmp_path / "metadata.yaml").write_text(
        "\n".join(
            [
                "status: available_relaxed",
                "paths:",
                f"  mlff_start_extxyz: {relaxed_extxyz}",
                f"  mlff_start_lammps_data: {relaxed_data}",
                "relaxation:",
                "  lammps_thermal_relax_performed: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert metadata_is_relaxed(tmp_path) is False
    relaxed_extxyz.write_text("1\n\nH 0 0 0\n", encoding="utf-8")
    relaxed_data.write_text("LAMMPS data\n", encoding="utf-8")
    assert metadata_is_relaxed(tmp_path) is True
