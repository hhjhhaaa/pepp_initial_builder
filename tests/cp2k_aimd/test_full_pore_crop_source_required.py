import pandas as pd

from tests.pore.test_silica_patch_crop import make_manual_pore
from pepp_initial_builder.cp2k_aimd.seed_patch_builder import build_aimd_local_structures


def test_cp2k_seed_requires_full_pore_source(tmp_path):
    cfg = make_manual_pore(tmp_path, "configs/cp2k_aimd.yaml")
    manifest = build_aimd_local_structures(cfg, "tiny")
    df = pd.read_csv(manifest)
    assert df.iloc[0]["status"] == "skipped_no_full_pore_snapshot_source"
    assert df.iloc[0]["failure_reason"] == "no_full_pore_seed_or_snapshot_manifest"
