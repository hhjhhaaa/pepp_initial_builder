import pandas as pd

from tests.pore.test_silica_patch_crop import make_manual_pore
from pepp_initial_builder.cp2k_aimd.seed_patch_builder import build_aimd_local_structures
from pepp_initial_builder.mlff_seed.full_pore_seed import build_full_pore_seed_structures


def test_aimd_local_structure_builder(tmp_path):
    cfg = make_manual_pore(tmp_path, ("configs/mlff_seed.yaml", "configs/cp2k_aimd.yaml"))
    build_full_pore_seed_structures(cfg, "tiny")
    manifest = build_aimd_local_structures(cfg, "tiny")
    df = pd.read_csv(manifest)
    assert len(df[df["status"] == "available"]) > 0
    assert df["extxyz_path"].iloc[0].endswith(".extxyz")
    assert set(df["crop_source"]) == {"full_pore_snapshot"}
    assert set(df["source_stage"]) == {"raw_full_pore_seed"}
    assert "source_full_pore_id" in df.columns
    assert "boundary_treatment" in df.columns
