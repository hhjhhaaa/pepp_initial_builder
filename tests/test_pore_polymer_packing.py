from pepp_initial_builder.pore_workflow import _read_xyz_like
from test_silica_patch_crop import make_manual_pore
from pepp_initial_builder.pore_workflow import build_full_pore_seed_structures
import pandas as pd


def test_full_pore_seed_contains_polymer_and_silica(tmp_path):
    cfg = make_manual_pore(tmp_path)
    manifest = build_full_pore_seed_structures(cfg, "tiny")
    row = pd.read_csv(manifest).query("status == 'available'").iloc[0]
    elems, _, _ = _read_xyz_like(row["extxyz_path"])
    assert {"Si", "O", "H", "C"}.issubset(set(elems))
