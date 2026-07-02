from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_old_and_new_script_entrypoints_exist():
    scripts = [
        "scripts/build_initial_structures.py",
        "scripts/build_porems_pores.py",
        "scripts/build_aimd_local_structures.py",
        "scripts/build_full_pore_seed_structures.py",
        "scripts/write_cp2k_label_inputs.py",
        "scripts/polymer/build_structures.py",
        "scripts/pore/build_pore_models.py",
        "scripts/mlff_seed/pack_polymer_into_pore.py",
        "scripts/cp2k_aimd/write_cp2k_inputs.py",
        "scripts/export/export_all_manifests.py",
    ]
    for rel in scripts:
        assert (ROOT / rel).exists()


def test_split_configs_exist():
    for rel in ["configs/polymer.yaml", "configs/pore.yaml", "configs/mlff_seed.yaml", "configs/cp2k_aimd.yaml"]:
        assert (ROOT / rel).exists()
