from pathlib import Path

from tests.pore.test_silica_patch_crop import make_manual_pore
from pepp_initial_builder.cp2k_aimd.seed_patch_builder import build_aimd_local_structures
from pepp_initial_builder.cp2k_aimd.structure_writer import write_cp2k_structure_inputs
from pepp_initial_builder.mlff_seed.full_pore_seed import build_full_pore_seed_structures
from pepp_initial_builder.pore.manifest import export_pore_aimd_manifests
from pepp_initial_builder.pore.patch_crop import crop_silica_patches


def test_pore_aimd_manifest_export(tmp_path):
    cfg = make_manual_pore(tmp_path, ("configs/mlff_seed.yaml", "configs/cp2k_aimd.yaml"))
    crop_silica_patches(cfg, "tiny")
    build_aimd_local_structures(cfg, "tiny")
    build_full_pore_seed_structures(cfg, "tiny")
    write_cp2k_structure_inputs(cfg)
    csv_path, json_path = export_pore_aimd_manifests(cfg)
    assert Path(csv_path).exists()
    assert Path(json_path).exists()
    assert "aimd_local" in Path(csv_path).read_text()
