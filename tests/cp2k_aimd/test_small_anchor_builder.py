from pathlib import Path

import pandas as pd
import yaml

from pepp_initial_builder.cp2k_aimd.dataset_builder import build_aimd_dataset
from pepp_initial_builder.cp2k_aimd.input_writer import write_cp2k_label_inputs
from pepp_initial_builder.cp2k_aimd.small_anchor_builder import build_small_anchor_structures


def _cfg(tmp_path: Path) -> dict:
    return {
        "paths": {
            "root": str(tmp_path),
            "aimd_structure_manifest": "data/exports/all_data_generation_manifest.csv",
            "aimd_local_manifest": "data/cp2k_aimd/seed_structures/aimd_local_manifest.csv",
            "cp2k_structure_input_manifest": "data/exports/cp2k_structure_input_manifest.csv",
            "aimd_local_structures_dir": "data/cp2k_aimd/seed_structures",
            "cp2k_jobs_dir": "data/cp2k_aimd/jobs",
            "cp2k_parsed_dir": "data/cp2k_aimd/parsed",
            "aimd_dataset_dir": "data/aimd_dataset",
            "exports_dir": "data/exports",
            "logs_dir": "outputs/logs",
            "jobs_dir": "outputs/jobs",
        },
        "dataset_scope": {"elements": ["C", "H", "O", "Si"]},
        "small_anchor_structures": {
            "cell_abc_A": [18.0, 18.0, 18.0],
            "min_polymer_surface_distance_A": 1.7,
            "tiny_max_structures": 13,
        },
        "cp2k": {
            "method": "PBE_D3_BJ",
            "xc_functional": "PBE",
            "dispersion": "D3_BJ",
            "basis_set_file": "BASIS_MOLOPT",
            "potential_file": "GTH_POTENTIALS",
            "dftd3_file": "dftd3.dat",
            "basis_set": "DZVP-MOLOPT-SR-GTH",
            "potential": "GTH-PBE",
            "cutoff_Ry": 400,
            "rel_cutoff_Ry": 50,
            "eps_scf": 1.0e-5,
            "max_scf": 100,
            "scf_solver": "OT",
            "outer_scf_max": 10,
            "temperatures_K": [523.0],
        },
        "label_modes": {
            "short_nvt_aimd": {
                "ensemble": "NVT",
                "timestep_fs": 0.25,
                "steps_tiny": 80,
                "frame_stride": 5,
                "eps_scf": 1.0e-4,
                "max_scf": 100,
                "outer_scf_max": 5,
                "thermostat_type": "CSVR",
                "thermostat_timecon_fs": 30.0,
            }
        },
        "dataset": {
            "dataset_id": "small_anchor_test",
            "tiny_max_structures": 13,
            "main_max_structures": 36,
            "min_frames_for_training": 2,
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "accepted_source_stages": ["designed_small_interface_anchor"],
        },
        "units": {
            "output_energy": "eV",
            "output_forces": "eV_per_A",
            "cp2k_energy_hartree_to_eV": 27.211386245988,
            "cp2k_force_hartree_per_bohr_to_eV_per_A": 51.422067476,
        },
    }


def test_small_anchor_structures_feed_cp2k_inputs(tmp_path):
    cfg = _cfg(tmp_path)
    manifest = build_small_anchor_structures(cfg, "tiny")
    rows = pd.read_csv(manifest)

    assert len(rows) == 13
    assert set(rows["source_stage"]) == {"designed_small_interface_anchor"}
    assert set(rows["status"]) == {"available"}
    assert rows["n_atoms"].max() < 120
    assert {"PC", "PE_PC", "PP_PC", "PE_PP_PC"}.issubset(set(rows["polymer_architecture"]))
    assert "PC_carbonate_silanol_contact" in set(rows["family"])
    assert "PC_phenyl_siloxane_contact" in set(rows["family"])
    assert {"nearest_site_type", "n_silanol_OH_within_5A", "n_siloxane_O_within_5A"}.issubset(rows.columns)
    for path in rows["extxyz_path"]:
        text = Path(path).read_text(encoding="utf-8")
        assert "energy=" not in text
        assert "forces" not in text.lower()
        assert 'Lattice="18.0 0 0 0 18.0 0 0 0 18.0"' in text

    input_manifest = write_cp2k_label_inputs(cfg, "tiny")
    input_rows = pd.read_csv(input_manifest)
    assert len(input_rows) == 13
    assert set(input_rows["label_mode"]) == {"sp_force"}
    first_input = Path(input_rows.iloc[0]["input_inp_path"]).read_text(encoding="utf-8")
    assert "RUN_TYPE ENERGY_FORCE" in first_input
    assert "&CELL" in first_input
    for element in ["C", "H", "O", "Si"]:
        assert f"&KIND {element}" in first_input


def _write_labeled_extxyz(path: Path, family: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '4\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3:forces:R:3 '
        f'energy=-10.0 source_stage=designed_small_interface_anchor family={family} pbc="T T T"\n'
        "Si 0 0 0 0.0 0.0 0.0\n"
        "O 1.6 0 0 0.1 0.0 0.0\n"
        "C 3.0 0 0 -0.1 0.0 0.0\n"
        "H 3.0 1 0 0.0 0.0 0.0\n",
        encoding="utf-8",
    )


def test_dataset_builder_accepts_configured_small_anchor_source(tmp_path):
    cfg = _cfg(tmp_path)
    parsed_dir = tmp_path / "data/cp2k_aimd/parsed"
    a = parsed_dir / "a/sp_force/frames.extxyz"
    b = parsed_dir / "b/sp_force/frames.extxyz"
    _write_labeled_extxyz(a, "PE_short_CH2_silanol_contact")
    _write_labeled_extxyz(b, "PP_methyl_silanol_contact")
    pd.DataFrame(
        [
            {
                "aimd_structure_id": "a",
                "family": "PE_short_CH2_silanol_contact",
                "crop_family": "PE_short_CH2_silanol_contact",
                "label_mode": "sp_force",
                "cp2k_run_type": "ENERGY_FORCE",
                "source_stage": "designed_small_interface_anchor",
                "polymer_architecture": "PE",
                "status": "parsed_real_cp2k_output",
                "usable_frame_count": 1,
                "frames_extxyz_path": str(a),
            },
            {
                "aimd_structure_id": "b",
                "family": "PP_methyl_silanol_contact",
                "crop_family": "PP_methyl_silanol_contact",
                "label_mode": "sp_force",
                "cp2k_run_type": "ENERGY_FORCE",
                "source_stage": "designed_small_interface_anchor",
                "polymer_architecture": "PP",
                "status": "parsed_real_cp2k_output",
                "usable_frame_count": 1,
                "frames_extxyz_path": str(b),
            },
        ]
    ).to_csv(parsed_dir / "cp2k_parsed_manifest.csv", index=False)

    build_aimd_dataset(cfg)
    summary = yaml.safe_load((tmp_path / "data/aimd_dataset/dataset_summary.yaml").read_text(encoding="utf-8"))
    assert summary["dataset_status"] == "ready"
    assert summary["num_frames_from_accepted_source_stages"] == 2
    assert summary["num_frames_from_rejected_source_stages"] == 0
    assert summary["num_frames_from_lammps_relaxed_full_pore"] == 0
