from pathlib import Path

import pandas as pd

from pepp_initial_builder.pore_workflow import build_porems_pores, crop_silica_patches, load_pore_config


def make_manual_pore(tmp_path):
    cfg = load_pore_config("configs/aimd_pore_builder.yaml")
    cfg["paths"]["root"] = str(tmp_path)
    d = tmp_path / "data/porems_models/manual_test"
    d.mkdir(parents=True)
    atoms = []
    for i in range(16):
        x = 15.0 + (6.0 if i % 2 == 0 else 5.2)
        y = 15.0 + (i - 8) * 0.45
        z = 8.0 + (i % 4) * 1.2
        atoms.append(("Si", x, y, z))
        atoms.append(("O", x - 0.9, y + 0.3, z))
        atoms.append(("O", x - 1.4, y - 0.3, z))
        atoms.append(("H", x - 1.9, y - 0.3, z))
    text = '64\nLattice="30 0 0 0 30 0 0 0 30" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
    text += "\n".join(f"{el} {x:.6f} {y:.6f} {z:.6f}" for el, x, y, z in atoms) + "\n"
    (d / "pore_model.extxyz").write_text(text, encoding="utf-8")
    build_porems_pores(cfg, "tiny")
    return cfg


def test_silica_patch_crop_from_manual_pore(tmp_path):
    cfg = make_manual_pore(tmp_path)
    manifest = crop_silica_patches(cfg, "tiny")
    df = pd.read_csv(manifest)
    assert df.iloc[0]["status"] == "available"
    assert Path(df.iloc[0]["patch_extxyz_path"]).exists()
    assert df.iloc[0]["usable_for_cp2k_aimd"] in (True, "True")
    assert int(df.iloc[0]["n_atoms"]) <= 100
    assert df.iloc[0]["cell_source"] == "rebuilt_local_patch_cell"
    assert df.iloc[0]["classification_confidence"] in {"low", "medium", "high"}
    assert "inward_normal_xyz" in df.columns
