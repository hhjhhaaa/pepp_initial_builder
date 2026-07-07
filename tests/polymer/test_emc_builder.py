from pathlib import Path

import pytest

from pepp_initial_builder.common.config import load_config
from pepp_initial_builder.common.openbabel import _obabel_format
from pepp_initial_builder.polymer.emc_builder import _recipe_text
from pepp_initial_builder.polymer.emc_library import (
    _metadata_for_system,
    _rewrite_extxyz_species_from_lammps_xyz,
    _thermal_relax_lock,
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


def test_batch2a_emc_config_renders_binary_mixture_recipes(tmp_path: Path):
    cfg = load_config("configs/polymer_batch2a_emc.yaml")
    row = cfg["emc_library"]["pilot_systems"][0]
    text = render_pure_recipe(row, cfg)
    assert "Components: PE/PP" in text
    assert "pe_polymer" in text
    assert "pp_polymer" in text
    assert "*CC*" in text
    assert "*C(C)C*" in text
    assert "packmol" not in text.lower()
    metadata = _metadata_for_system(cfg, row, tmp_path)
    assert metadata["builder"]["builder_used"] == "emc"
    assert metadata["components"] == ["PE", "PP"]
    assert metadata["component_chain_counts"] == {"PE": 5, "PP": 5}
    assert metadata["component_chain_counts_arg"] == "PE:5,PP:5"
    assert metadata["structure_task"]["lane"] == "mlff_direct"


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


def test_thermal_relax_lock_blocks_concurrent_system_runs(tmp_path: Path):
    with _thermal_relax_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with _thermal_relax_lock(tmp_path):
                pass


def test_relaxed_extxyz_rewrites_lammps_types_to_elements(tmp_path: Path):
    data = tmp_path / "relaxed.data"
    data.write_text(
        "\n".join(
            [
                "LAMMPS data",
                "",
                "Masses",
                "",
                "1 12.011",
                "2 1.008",
                "",
                "Atoms # full",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    xyz = tmp_path / "relaxed.xyz"
    xyz.write_text("2\nAtoms. Timestep: 1\n1 0 0 0\n2 1 0 0\n", encoding="utf-8")
    extxyz = tmp_path / "relaxed.extxyz"
    extxyz.write_text(
        '2\nLattice="1 0 0 0 1 0 0 0 1" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "H 0 0 0\nHe 1 0 0\n",
        encoding="utf-8",
    )
    _rewrite_extxyz_species_from_lammps_xyz(xyz, extxyz, data)
    text = extxyz.read_text(encoding="utf-8")
    assert "\n   C" in text
    assert "\n   H" in text
    assert "He" not in text
