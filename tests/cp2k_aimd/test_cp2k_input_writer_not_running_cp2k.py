import yaml

from pepp_initial_builder.cp2k_aimd.hpc_jobs import make_hpc_cp2k_jobs


def test_hpc_job_has_module_placeholder(tmp_path):
    config = yaml.safe_load(open("configs/cp2k_aimd.yaml", "r", encoding="utf-8"))
    config["paths"]["root"] = str(tmp_path)
    config["paths"]["cp2k_jobs_dir"] = "jobs"
    config["paths"]["jobs_dir"] = "slurm"
    config["paths"]["cp2k_parsed_dir"] = "parsed"
    config["paths"]["aimd_dataset_dir"] = "dataset"
    config["paths"]["exports_dir"] = "exports"
    config["paths"]["logs_dir"] = "logs"
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "cp2k_label_input_manifest.csv").write_text(
        "aimd_structure_id,family,label_mode,status,job_dir\n"
        "s1,silica_patch_only,short_aimd,cp2k_input_written_no_cp2k_run,/tmp/job\n",
        encoding="utf-8",
    )
    manifest = make_hpc_cp2k_jobs(config, "tiny")
    text = (tmp_path / "slurm" / "run_cp2k_short_aimd_array.sbatch").read_text(encoding="utf-8")
    sp_text = (tmp_path / "slurm" / "run_cp2k_sp_array.sbatch").read_text(encoding="utf-8")
    assert manifest.exists()
    assert "module purge" in text
    assert "module load cp2k-2024.1" in text
    assert "#SBATCH --time" not in text
    assert "#SBATCH --time" not in sp_text
    assert (tmp_path / "slurm" / "verify_cp2k_module.sh").exists()
    assert '# export CP2K_DATA_DIR="__SET_CP2K_DATA_DIR_ON_HPC__"' in text
    assert 'export OMPI_MCA_btl="${OMPI_MCA_btl:-^openib}"' in text
    assert "CP2K_CMD=${CP2K_CMD:-cp2k.psmp}" in text
    assert (tmp_path / "slurm" / "submit_cp2k_sp_tiny.sh").exists()
    assert (tmp_path / "slurm" / "submit_cp2k_short_aimd_tiny.sh").exists()
    assert (tmp_path / "slurm" / "submit_cp2k_seed_tiny.sh").exists()
