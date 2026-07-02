from pathlib import Path

import pandas as pd

from pepp_initial_builder.pore_workflow import build_porems_pores, crop_silica_patches, load_pore_config


def make_manual_pore(tmp_path):
    cfg = load_pore_config("configs/aimd_pore_builder.yaml")
    cfg["paths"]["root"] = str(tmp_path)
    d = tmp_path / "data/porems_models/manual_test"
    d.mkdir(parents=True)
    (d / "pore_model.extxyz").write_text(
        '9\nLattice="24 0 0 0 24 0 0 0 24" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        "Si 0 0 0\nO 1.6 0 0\nO -1.6 0 0\nO 0 1.6 0\nO 0 -1.6 0\nH 2.2 0 0\nH -2.2 0 0\nH 0 2.2 0\nH 0 -2.2 0\n"
    )
    build_porems_pores(cfg, "tiny")
    return cfg


def test_silica_patch_crop_from_manual_pore(tmp_path):
    cfg = make_manual_pore(tmp_path)
    manifest = crop_silica_patches(cfg, "tiny")
    df = pd.read_csv(manifest)
    assert df.iloc[0]["status"] == "available"
    assert Path(df.iloc[0]["patch_extxyz_path"]).exists()
