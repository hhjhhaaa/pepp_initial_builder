import pandas as pd

from pepp_initial_builder.cp2k_aimd.full_pore_snapshot_reader import read_full_pore_snapshot_sources


def test_raw_full_pore_seed_is_not_cp2k_crop_source(tmp_path):
    seed = tmp_path / "data/mlff_seed/structures/full_pore_seed_0001/seed.extxyz"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text('1\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3 pbc="T T T"\nC 5 5 5\n', encoding="utf-8")
    pd.DataFrame(
        [
            {
                "full_pore_seed_id": "full_pore_seed_0001",
                "status": "available",
                "source_stage": "raw_full_pore_seed",
                "extxyz_path": str(seed),
                "usable_for_mlff_start": True,
            }
        ]
    ).to_csv(tmp_path / "data/mlff_seed/structures/full_pore_seed_manifest.csv", index=False)
    cfg = {
        "paths": {
            "root": str(tmp_path),
            "exports_dir": "data/exports",
            "full_pore_seed_structures_dir": "data/mlff_seed/structures",
            "logs_dir": "outputs/logs",
        },
        "cp2k_crop": {"source_priority": ["raw_full_pore_seed", "lammps_relaxed_full_pore"]},
    }
    assert read_full_pore_snapshot_sources(cfg) == []
