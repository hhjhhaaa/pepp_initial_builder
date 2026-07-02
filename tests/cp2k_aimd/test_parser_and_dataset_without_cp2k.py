import yaml

from pepp_initial_builder.cp2k_aimd.dataset_builder import build_aimd_dataset
from pepp_initial_builder.cp2k_aimd.parser import parse_cp2k_outputs


def _config(tmp_path):
    config = yaml.safe_load(open("configs/cp2k_aimd.yaml", "r", encoding="utf-8"))
    config["paths"]["root"] = str(tmp_path)
    config["paths"]["cp2k_jobs_dir"] = "jobs"
    config["paths"]["cp2k_parsed_dir"] = "parsed"
    config["paths"]["aimd_dataset_dir"] = "dataset"
    config["paths"]["exports_dir"] = "exports"
    config["paths"]["logs_dir"] = "logs"
    config["paths"]["jobs_dir"] = "slurm"
    return config


def test_parser_marks_not_run_without_cp2k_out(tmp_path):
    config = _config(tmp_path)
    job_dir = tmp_path / "jobs" / "s1" / "sp_force"
    job_dir.mkdir(parents=True)
    (tmp_path / "jobs" / "cp2k_label_input_manifest.csv").write_text(
        "aimd_structure_id,family,label_mode,status,job_dir\n"
        f"s1,pe_near_silanol_wall,sp_force,cp2k_input_written_no_cp2k_run,{job_dir}\n",
        encoding="utf-8",
    )
    manifest = parse_cp2k_outputs(config)
    text = manifest.read_text(encoding="utf-8")
    assert "not_run_no_cp2k_output" in text


def test_dataset_builder_refuses_fake_frames(tmp_path):
    config = _config(tmp_path)
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    (parsed / "cp2k_parsed_manifest.csv").write_text(
        "aimd_structure_id,family,label_mode,status,usable_frame_count,failure_reason,frames_extxyz_path\n"
        "s1,pe_near_silanol_wall,sp_force,not_run_no_cp2k_output,0,cp2k.out missing,\n",
        encoding="utf-8",
    )
    build_aimd_dataset(config)
    summary = (tmp_path / "dataset" / "dataset_summary.yaml").read_text(encoding="utf-8")
    assert "insufficient_real_cp2k_frames" in summary
    assert "usable_for_mlff_training: false" in summary
