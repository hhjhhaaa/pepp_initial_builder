from pathlib import Path

import pandas as pd
import yaml

from pepp_initial_builder.cp2k_aimd.dataset_builder import build_aimd_dataset
from pepp_initial_builder.cp2k_aimd.hpc_jobs import make_hpc_cp2k_jobs
from pepp_initial_builder.cp2k_aimd.input_writer import write_cp2k_label_inputs
from pepp_initial_builder.cp2k_aimd.manifest import export_aimd_dataset_manifest
from pepp_initial_builder.cp2k_aimd.parser import parse_cp2k_outputs
from pepp_initial_builder.reuse.lmp_proj_discovery import discover_lmp_proj_modules


def _write_extxyz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '4\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "Si 0 0 0\nO 1.6 0 0\nC 3 0 0\nH 3 1 0\n",
        encoding="utf-8",
    )


def _cfg(tmp_path: Path) -> dict:
    lmp = tmp_path / "lmp-proj"
    (lmp / "microcal" / "cp2k").mkdir(parents=True)
    (lmp / "microcal" / "cp2k" / "parse.py").write_text("# cp2k parser\n", encoding="utf-8")
    (lmp / "microcal" / "common").mkdir(parents=True)
    (lmp / "microcal" / "common" / "slurm.py").write_text("# sbatch slurm\n", encoding="utf-8")
    return {
        "paths": {
            "root": str(tmp_path),
            "aimd_structure_manifest": "data/exports/pore_aimd_master_manifest.csv",
            "aimd_local_manifest": "data/cp2k_aimd/seed_structures/aimd_local_manifest.csv",
            "cp2k_structure_input_manifest": "data/exports/cp2k_structure_input_manifest.csv",
            "cp2k_jobs_dir": "data/cp2k_aimd/jobs",
            "cp2k_parsed_dir": "data/cp2k_aimd/parsed",
            "aimd_dataset_dir": "data/aimd_dataset",
            "exports_dir": "data/exports",
            "logs_dir": "outputs/logs",
            "jobs_dir": "outputs/jobs",
            "lmp_proj_root": str(lmp),
        },
        "dataset_scope": {"elements": ["C", "H", "O", "Si"]},
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
            "scf_mixing": "BROYDEN",
            "electronic_smearing_K": 300,
            "temperatures_K": [523.0],
        },
        "label_modes": {
            "short_nvt_aimd": {"ensemble": "NVT", "timestep_fs": 0.5, "steps_tiny": 100, "steps_main": 2000, "frame_stride": 20}
        },
        "hpc": {
            "cp2k_module_placeholder": "__SET_CP2K_MODULE_ON_HPC__",
            "cp2k_command_default": "cp2k.psmp",
            "nodes": 1,
            "ntasks_per_node": 32,
            "cpus_per_task": 1,
            "time_sp": "04:00:00",
            "time_aimd": "24:00:00",
        },
        "dataset": {
            "dataset_id": "test",
            "tiny_max_structures": 4,
            "main_max_structures": 100,
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "min_frames_for_training": 10,
            "require_energy": True,
            "require_forces": True,
        },
        "units": {
            "output_energy": "eV",
            "output_forces": "eV_per_A",
            "cp2k_energy_hartree_to_eV": 27.211386245988,
            "cp2k_force_hartree_per_bohr_to_eV_per_A": 51.422067476,
        },
    }


def test_discover_lmp_proj_reuse_report(tmp_path):
    cfg = _cfg(tmp_path)
    report = discover_lmp_proj_modules(cfg)
    assert Path(report["txt"]).exists()
    assert Path(report["json"]).exists()
    assert Path(report["file_index"]).exists()


def test_cp2k_inputs_slurm_and_no_output_status(tmp_path):
    cfg = _cfg(tmp_path)
    a = tmp_path / "data/cp2k_aimd/seed_structures/a/structure.extxyz"
    b = tmp_path / "data/cp2k_aimd/seed_structures/b/structure.extxyz"
    _write_extxyz(a)
    _write_extxyz(b)
    manifest = tmp_path / "data/cp2k_aimd/seed_structures/aimd_local_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"aimd_structure_id": "aimd_a", "status": "available", "family": "pe_near_silanol_wall", "extxyz_path": str(a)},
            {"aimd_structure_id": "aimd_b", "status": "available", "family": "silica_patch_only", "extxyz_path": str(b)},
        ]
    ).to_csv(manifest, index=False)

    input_manifest = write_cp2k_label_inputs(cfg, "tiny")
    text_a = (tmp_path / "data/cp2k_aimd/jobs/aimd_a/sp_force/input.inp").read_text(encoding="utf-8")
    assert not (tmp_path / "data/cp2k_aimd/jobs/aimd_b/short_aimd/input.inp").exists()
    assert "RUN_TYPE ENERGY_FORCE" in text_a
    assert "DFTD3(BJ)" in text_a
    assert "&CELL" in text_a
    assert "ABC 10.0000000000 10.0000000000 10.0000000000" in text_a
    assert "@INCLUDE coords.inc" in text_a
    for element in ["C", "H", "O", "Si"]:
        assert f"&KIND {element}" in text_a
    assert Path(input_manifest).exists()

    selected = tmp_path / "data/exports/selected_short_aimd_manifest.csv"
    pd.DataFrame(
        [
            {
                "aimd_structure_id": "aimd_b",
                "status": "selected_for_short_aimd",
                "source_sp_status": "parsed_real_cp2k_output",
                "family": "silica_only_wall_baseline",
                "extxyz_path": str(b),
            }
        ]
    ).to_csv(selected, index=False)
    write_cp2k_label_inputs(cfg, "tiny")
    text_b = (tmp_path / "data/cp2k_aimd/jobs/aimd_b/short_aimd/input.inp").read_text(encoding="utf-8")
    assert "RUN_TYPE MD" in text_b
    assert "&MOTION" in text_b

    job_manifest = make_hpc_cp2k_jobs(cfg, "tiny")
    sp_script = (tmp_path / "outputs/jobs/run_cp2k_sp_array.sbatch").read_text(encoding="utf-8")
    assert "__SET_CP2K_MODULE_ON_HPC__" in sp_script
    assert '# export CP2K_DATA_DIR="__SET_CP2K_DATA_DIR_ON_HPC__"' in sp_script
    assert (tmp_path / "outputs/jobs/submit_cp2k_sp_tiny.sh").exists()
    assert (tmp_path / "outputs/jobs/submit_cp2k_short_aimd_tiny.sh").exists()
    assert (tmp_path / "outputs/jobs/submit_cp2k_seed_tiny.sh").exists()
    assert Path(job_manifest).exists()

    parsed = parse_cp2k_outputs(cfg)
    parsed_df = pd.read_csv(parsed)
    assert set(parsed_df["status"]) == {"not_run_no_cp2k_output"}

    dataset_manifest = build_aimd_dataset(cfg)
    summary = yaml.safe_load((tmp_path / "data/aimd_dataset/dataset_summary.yaml").read_text(encoding="utf-8"))
    assert summary["dataset_status"] == "insufficient_real_cp2k_frames"
    assert summary["usable_for_mlff_training"] is False
    assert Path(dataset_manifest).exists()

    exported = export_aimd_dataset_manifest(cfg)
    exported_df = pd.read_csv(exported)
    assert exported_df.iloc[0]["usable_for_mlff_training"] == False
