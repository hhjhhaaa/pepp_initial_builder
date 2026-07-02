import pandas as pd

from tests.pore.test_silica_patch_crop import make_manual_pore
from pepp_initial_builder.mlff_seed.full_pore_seed import build_full_pore_seed_structures


def test_full_pore_seed_builder(tmp_path):
    cfg = make_manual_pore(tmp_path, "configs/mlff_seed.yaml")
    manifest = build_full_pore_seed_structures(cfg, "tiny")
    df = pd.read_csv(manifest)
    assert len(df[df["status"] == "available"]) > 0
    assert df["extxyz_path"].iloc[0].endswith(".extxyz")
