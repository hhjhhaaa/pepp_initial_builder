import pandas as pd

from tests.pore.test_silica_patch_crop import make_manual_pore
from pepp_initial_builder.cp2k_aimd.seed_patch_builder import build_aimd_local_structures
from tests.cp2k_aimd.helpers import write_relaxed_full_pore_source


def test_aimd_local_structure_builder(tmp_path):
    cfg = make_manual_pore(tmp_path, ("configs/mlff_seed.yaml", "configs/cp2k_aimd.yaml"))
    write_relaxed_full_pore_source(cfg, tmp_path)
    manifest = build_aimd_local_structures(cfg, "tiny")
    df = pd.read_csv(manifest)
    available = df[df["status"] == "available"]
    assert len(available) > 0
    assert available["extxyz_path"].iloc[0].endswith(".extxyz")
    assert set(df["crop_source"]) == {"full_pore_snapshot"}
    assert set(df["source_stage"]) == {"lammps_relaxed_full_pore"}
    assert "source_full_pore_id" in df.columns
    assert "boundary_treatment" in df.columns
