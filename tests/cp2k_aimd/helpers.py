from pathlib import Path

import pandas as pd


def write_relaxed_full_pore_source(cfg: dict, tmp_path: Path) -> Path:
    source = tmp_path / "data/mlff_seed/structures/full_pore_seed_0001_PE100_PP00_seed1/relaxed.extxyz"
    source.parent.mkdir(parents=True, exist_ok=True)
    atoms = []
    for i in range(12):
        atoms.append(("Si", 15.0 + 5.5, 12.0 + i * 0.4, 8.0 + (i % 4) * 1.2))
        atoms.append(("O", 15.0 + 4.5, 12.2 + i * 0.4, 8.0 + (i % 4) * 1.2))
        atoms.append(("H", 15.0 + 4.0, 12.2 + i * 0.4, 8.0 + (i % 4) * 1.2))
    for i in range(8):
        atoms.append(("C", 15.0 + 3.1, 13.0 + i * 0.35, 9.0))
        atoms.append(("H", 15.0 + 2.6, 13.0 + i * 0.35, 9.8))
    text = f'{len(atoms)}\nLattice="30 0 0 0 30 0 0 0 30" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
    text += "\n".join(f"{e} {x:.6f} {y:.6f} {z:.6f}" for e, x, y, z in atoms) + "\n"
    source.write_text(text, encoding="utf-8")
    manifest = tmp_path / "data/mlff_seed/structures/full_pore_seed_manifest.csv"
    pd.DataFrame(
        [
            {
                "full_pore_seed_id": "full_pore_seed_0001_PE100_PP00_seed1",
                "status": "available",
                "source_stage": "lammps_relaxed_full_pore",
                "source_full_pore_id": "full_pore_seed_0001_PE100_PP00_seed1",
                "extxyz_path": str(source),
                "mlff_start_structure_kind": "lammps_relaxed_full_pore",
                "usable_for_mlff_start": True,
            }
        ]
    ).to_csv(manifest, index=False)
    return source
