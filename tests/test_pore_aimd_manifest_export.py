from pathlib import Path

from test_silica_patch_crop import make_manual_pore
from pepp_initial_builder.pore_workflow import (
    build_aimd_local_structures,
    build_full_pore_seed_structures,
    crop_silica_patches,
    export_pore_aimd_manifests,
    write_cp2k_structure_inputs,
)


def test_pore_aimd_manifest_export(tmp_path):
    cfg = make_manual_pore(tmp_path)
    crop_silica_patches(cfg, "tiny")
    build_aimd_local_structures(cfg, "tiny")
    build_full_pore_seed_structures(cfg, "tiny")
    write_cp2k_structure_inputs(cfg)
    csv_path, json_path = export_pore_aimd_manifests(cfg)
    assert Path(csv_path).exists()
    assert Path(json_path).exists()
    assert "aimd_local" in Path(csv_path).read_text()
