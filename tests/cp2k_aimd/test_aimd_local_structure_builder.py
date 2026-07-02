import pandas as pd

from tests.pore.test_silica_patch_crop import make_manual_pore
from pepp_initial_builder.cp2k_aimd.seed_patch_builder import build_aimd_local_structures
from pepp_initial_builder.pore.patch_crop import crop_silica_patches


def test_aimd_local_structure_builder(tmp_path):
    cfg = make_manual_pore(tmp_path, "configs/cp2k_aimd.yaml")
    crop_silica_patches(cfg, "tiny")
    manifest = build_aimd_local_structures(cfg, "tiny")
    df = pd.read_csv(manifest)
    assert len(df[df["status"] == "available"]) > 0
    assert df["extxyz_path"].iloc[0].endswith(".extxyz")
