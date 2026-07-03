import csv
from pathlib import Path

import yaml

from pepp_initial_builder.export.anchor_scaffold import scaffold_mlff_aimd_anchor


def test_anchor_scaffold_creates_planned_manifests_without_fake_frames(tmp_path):
    cfg = yaml.safe_load(Path("configs/mlff_aimd_anchor.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["root"] = str(tmp_path)
    cfg_path = tmp_path / "mlff_aimd_anchor.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    anchor = scaffold_mlff_aimd_anchor(cfg_path)

    assert (anchor / "initial_structures/bulk_core").is_dir()
    assert (anchor / "initial_structures/pc_extension").is_dir()
    assert (anchor / "initial_structures/silica_interface").is_dir()
    assert (anchor / "cp2k_inputs/silica_interface").is_dir()
    system_rows = list(csv.DictReader((anchor / "manifests/system_manifest.csv").open(encoding="utf-8")))
    assert {row["domain"] for row in system_rows} == {"bulk_core", "pc_extension", "silica_interface"}
    assert all(row["eligible_for_training"] == "false" for row in system_rows)
    assert all(not row["structure_path"] for row in system_rows)
    pc_rows = [row for row in system_rows if row["domain"] == "pc_extension"]
    assert pc_rows
    assert all(row["initial_structure_status"] == "pending_builder_gate" for row in pc_rows)
    frame_manifest = (anchor / "manifests/frame_manifest.csv").read_text(encoding="utf-8")
    assert "forces_available" in frame_manifest
    assert len(list(csv.DictReader((anchor / "manifests/frame_manifest.csv").open(encoding="utf-8")))) == 0
