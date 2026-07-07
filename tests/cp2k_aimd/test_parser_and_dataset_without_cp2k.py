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


def _write_coords(path):
    path.write_text(
        '2\nLattice="12 0 0 0 12 0 0 0 12" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "Si 5.0 5.0 5.0\n"
        "O 6.6 5.0 5.0\n",
        encoding="utf-8",
    )


def _write_xyz_frames(path, comments_and_rows):
    text = []
    for comment, rows in comments_and_rows:
        text.append(str(len(rows)))
        text.append(comment)
        text.extend(rows)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def test_parser_fixture_energy_force_units_and_provenance(tmp_path):
    config = _config(tmp_path)
    job_dir = tmp_path / "jobs" / "s_sp" / "sp_force"
    job_dir.mkdir(parents=True)
    _write_coords(job_dir / "coords.xyz")
    (job_dir / "cp2k.out").write_text(
        " ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):              -1.000000000000\n"
        " PROGRAM ENDED AT 2026-07-02 00:00:00\n",
        encoding="utf-8",
    )
    _write_xyz_frames(
        job_dir / "s_sp_sp_force-frc-1.xyz",
        [
            (
                "forces in Hartree/Bohr",
                [
                    "Si 0.010000 0.000000 0.000000",
                    "O 0.000000 -0.020000 0.000000",
                ],
            )
        ],
    )
    (tmp_path / "jobs" / "cp2k_label_input_manifest.csv").write_text(
        "aimd_structure_id,family,patch_id,label_mode,cp2k_project,cp2k_run_type,status,job_dir\n"
        f"s_sp,pe_near_silanol_wall,patch_A,sp_force,s_sp_sp_force,ENERGY_FORCE,cp2k_input_written_no_cp2k_run,{job_dir}\n",
        encoding="utf-8",
    )

    manifest = parse_cp2k_outputs(config)
    parsed = manifest.read_text(encoding="utf-8")
    assert "parsed_real_cp2k_output" in parsed
    frames = (tmp_path / "parsed" / "s_sp" / "sp_force" / "frames.extxyz").read_text(encoding="utf-8")
    assert "energy=-27.211386245988" in frames
    assert "forces_unit=\"eV/Angstrom\"" in frames
    assert "aimd_seed_id=s_sp" in frames
    assert "patch_id=patch_A" in frames
    assert "cp2k_run_type=ENERGY_FORCE" in frames
    summary = yaml.safe_load((tmp_path / "parsed" / "s_sp" / "sp_force" / "parse_summary.yaml").read_text(encoding="utf-8"))
    assert summary["detected_cp2k_out"].endswith("cp2k.out")
    assert summary["detected_force_file"].endswith("s_sp_sp_force-frc-1.xyz")
    assert summary["n_frames_written"] == 1


def test_parser_reads_energy_force_forces_from_cp2k_out(tmp_path):
    config = _config(tmp_path)
    job_dir = tmp_path / "jobs" / "s_out_force" / "sp_force"
    job_dir.mkdir(parents=True)
    _write_coords(job_dir / "coords.xyz")
    (job_dir / "cp2k.out").write_text(
        " ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:              -2.000000000000\n"
        " ATOMIC FORCES in [a.u.]\n"
        " # Atom   Kind   Element          X              Y              Z\n"
        "      1      1      Si        0.010000000000  0.000000000000  0.000000000000\n"
        "      2      2      O         0.000000000000 -0.020000000000  0.000000000000\n"
        " SUM OF ATOMIC FORCES           0.010000000000 -0.020000000000  0.000000000000\n"
        " PROGRAM ENDED AT 2026-07-02 00:00:00\n",
        encoding="utf-8",
    )
    (tmp_path / "jobs" / "cp2k_label_input_manifest.csv").write_text(
        "aimd_structure_id,family,patch_id,label_mode,cp2k_project,cp2k_run_type,status,job_dir\n"
        f"s_out_force,silica_patch_only,patch_C,sp_force,s_out_force_sp_force,ENERGY_FORCE,cp2k_input_written_no_cp2k_run,{job_dir}\n",
        encoding="utf-8",
    )

    parse_cp2k_outputs(config)
    summary = yaml.safe_load((tmp_path / "parsed" / "s_out_force" / "sp_force" / "parse_summary.yaml").read_text(encoding="utf-8"))
    frames = (tmp_path / "parsed" / "s_out_force" / "sp_force" / "frames.extxyz").read_text(encoding="utf-8")
    assert summary["status"] == "parsed_real_cp2k_output"
    assert summary["detected_force_file"].endswith("cp2k.out")
    assert summary["n_force_frames"] == 1
    assert "energy=-54.422772491976" in frames
    assert "0.514220674760" in frames


def test_parser_reads_timestamped_cp2k_run_dir(tmp_path):
    config = _config(tmp_path)
    job_dir = tmp_path / "jobs" / "s_timestamped" / "sp_force"
    run_dir = job_dir / "runs" / "20260707_171500_123_0"
    run_dir.mkdir(parents=True)
    _write_coords(run_dir / "coords.xyz")
    (run_dir / "cp2k.out").write_text(
        " ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:              -2.000000000000\n"
        " ATOMIC FORCES in [a.u.]\n"
        " # Atom   Kind   Element          X              Y              Z\n"
        "      1      1      Si        0.010000000000  0.000000000000  0.000000000000\n"
        "      2      2      O         0.000000000000 -0.020000000000  0.000000000000\n"
        " SUM OF ATOMIC FORCES           0.010000000000 -0.020000000000  0.000000000000\n"
        " PROGRAM ENDED AT 2026-07-02 00:00:00\n",
        encoding="utf-8",
    )
    (tmp_path / "jobs" / "cp2k_label_input_manifest.csv").write_text(
        "aimd_structure_id,family,patch_id,label_mode,cp2k_project,cp2k_run_type,status,job_dir\n"
        f"s_timestamped,silica_patch_only,patch_C,sp_force,s_timestamped_sp_force,ENERGY_FORCE,cp2k_input_written_no_cp2k_run,{job_dir}\n",
        encoding="utf-8",
    )

    parse_cp2k_outputs(config)

    summary = yaml.safe_load((tmp_path / "parsed" / "s_timestamped" / "sp_force" / "parse_summary.yaml").read_text(encoding="utf-8"))
    assert summary["status"] == "parsed_real_cp2k_output"
    assert summary["detected_cp2k_out"].endswith("runs/20260707_171500_123_0/cp2k.out")
    assert summary["detected_position_file"].endswith("runs/20260707_171500_123_0/coords.xyz")


def test_parser_fixture_md_matched_frames_and_mismatch_summary(tmp_path):
    config = _config(tmp_path)
    job_dir = tmp_path / "jobs" / "s_md" / "short_aimd"
    job_dir.mkdir(parents=True)
    _write_coords(job_dir / "coords.xyz")
    (job_dir / "cp2k.out").write_text(" PROGRAM ENDED AT 2026-07-02 00:00:00\n", encoding="utf-8")
    _write_xyz_frames(
        job_dir / "s_md_short_aimd-pos-1.xyz",
        [
            ("pos frame 0", ["Si 5.0 5.0 5.0", "O 6.6 5.0 5.0"]),
            ("pos frame 1", ["Si 5.1 5.0 5.0", "O 6.7 5.0 5.0"]),
        ],
    )
    _write_xyz_frames(
        job_dir / "s_md_short_aimd-frc-1.xyz",
        [("force frame 0", ["Si 0.010000 0.000000 0.000000", "O 0.000000 0.020000 0.000000"])],
    )
    (job_dir / "s_md_short_aimd-1.ener").write_text(
        "# step time kin temp pot\n"
        "0 0.0 0.0 523.0 -2.000000\n"
        "1 0.5 0.0 523.0 -1.900000\n",
        encoding="utf-8",
    )
    (tmp_path / "jobs" / "cp2k_label_input_manifest.csv").write_text(
        "aimd_structure_id,family,patch_id,label_mode,cp2k_project,cp2k_run_type,status,job_dir\n"
        f"s_md,silica_patch_only,patch_B,short_aimd,s_md_short_aimd,MD,cp2k_input_written_no_cp2k_run,{job_dir}\n",
        encoding="utf-8",
    )

    parse_cp2k_outputs(config)
    summary = yaml.safe_load((tmp_path / "parsed" / "s_md" / "short_aimd" / "parse_summary.yaml").read_text(encoding="utf-8"))
    assert summary["parser_mode"] == "MD"
    assert summary["n_position_frames"] == 2
    assert summary["n_force_frames"] == 1
    assert summary["n_energy_frames"] == 2
    assert summary["n_frames_written"] == 1
    assert summary["frame_count_mismatch"] is True
    frames = (tmp_path / "parsed" / "s_md" / "short_aimd" / "frames.extxyz").read_text(encoding="utf-8")
    assert frames.count("\n2\n") == 0
    assert frames.startswith("2\n")
    assert "source_frame_index=0" in frames
    assert "cp2k_run_type=MD" in frames


def test_dataset_split_keeps_same_structure_block_together(tmp_path):
    config = _config(tmp_path)
    config["dataset"]["min_frames_for_training"] = 2
    config["dataset"]["train_fraction"] = 0.5
    config["dataset"]["val_fraction"] = 0.25
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    for name in ["s1_sp", "s1_md", "s2_sp", "s2_md"]:
        (parsed / f"{name}.extxyz").write_text(
            f'1\nProperties=species:S:1:pos:R:3 source_job_dir="{name}" aimd_seed_id={name.split("_")[0]}\nH 0 0 0\n',
            encoding="utf-8",
        )
    (parsed / "cp2k_parsed_manifest.csv").write_text(
        "aimd_structure_id,family,crop_family,patch_id,label_mode,cp2k_run_type,source_stage,polymer_architecture,status,usable_frame_count,failure_reason,frames_extxyz_path\n"
        f"s1,pe_near_silanol_wall,pe_near_silanol_wall,patch_A,sp_force,ENERGY_FORCE,lammps_relaxed_full_pore,PE_HDPE100_PP00,parsed_real_cp2k_output,1,,{parsed / 's1_sp.extxyz'}\n"
        f"s1,pe_near_silanol_wall,pe_near_silanol_wall,patch_A,short_aimd,MD,lammps_relaxed_full_pore,PE_HDPE100_PP00,parsed_real_cp2k_output,1,,{parsed / 's1_md.extxyz'}\n"
        f"s2,pe_near_silanol_wall,pe_near_silanol_wall,patch_B,sp_force,ENERGY_FORCE,lammps_relaxed_full_pore,PE00_PP100,parsed_real_cp2k_output,1,,{parsed / 's2_sp.extxyz'}\n"
        f"s2,pe_near_silanol_wall,pe_near_silanol_wall,patch_B,short_aimd,MD,lammps_relaxed_full_pore,PE00_PP100,parsed_real_cp2k_output,1,,{parsed / 's2_md.extxyz'}\n",
        encoding="utf-8",
    )

    build_aimd_dataset(config)
    train = (tmp_path / "dataset" / "train.extxyz").read_text(encoding="utf-8")
    val = (tmp_path / "dataset" / "val.extxyz").read_text(encoding="utf-8")
    assert "source_job_dir=\"s1_sp\"" in train
    assert "source_job_dir=\"s1_md\"" in train
    assert "source_job_dir=\"s1_sp\"" not in val
    assert "source_job_dir=\"s2_sp\"" in val
    assert "source_job_dir=\"s2_md\"" in val
