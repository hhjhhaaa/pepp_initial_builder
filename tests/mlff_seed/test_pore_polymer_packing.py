from pepp_initial_builder.pore.porems_builder import read_xyz_like
from tests.pore.test_silica_patch_crop import make_manual_pore
from pepp_initial_builder.mlff_seed.full_pore_seed import build_full_pore_seed_structures
import pandas as pd


def test_full_pore_seed_contains_polymer_and_silica(tmp_path):
    cfg = make_manual_pore(tmp_path, "configs/mlff_seed.yaml")
    manifest = build_full_pore_seed_structures(cfg, "tiny")
    row = pd.read_csv(manifest).query("status == 'available'").iloc[0]
    elems, _, _ = read_xyz_like(row["extxyz_path"])
    assert {"Si", "O", "H", "C"}.issubset(set(elems))
