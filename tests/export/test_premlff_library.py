import pandas as pd
import yaml

from pepp_initial_builder.export.premlff_library import build_premlff_structure_library


def test_build_premlff_structure_library_merges_snapshots_and_metrics(tmp_path):
    run_id = "pilot_test_pe"
    exports = tmp_path / "data" / "runs" / run_id / "exports"
    logs = tmp_path / "outputs" / "runs" / run_id / "logs"
    pore = tmp_path / "data" / "runs" / run_id / "mesoporous_silica" / "pore_models" / "pore"
    exports.mkdir(parents=True)
    logs.mkdir(parents=True)
    pore.mkdir(parents=True)
    pd.DataFrame([
        {
            "run_id": run_id,
            "full_pore_seed_id": "seed1",
            "source_stage": "lammps_relaxed_full_pore",
            "snapshot_path": "/tmp/relaxed.extxyz",
            "polymer_architecture": "PE75_PP00_PS25",
            "composition": "PE75_PP00_PS25",
            "pe_variant": "PE_HDPE_linear",
            "ps_variant": "PS_atactic_phenyl_v0",
            "atom_roles_path": "/tmp/roles.csv",
            "time_ps": 70.0,
            "usable_for_cp2k_crop": True,
        }
    ]).to_csv(exports / "full_pore_snapshot_manifest.csv", index=False)
    seeds = tmp_path / "data" / "runs" / run_id / "mlff_seed" / "structures"
    seeds.mkdir(parents=True)
    pd.DataFrame([{"full_pore_seed_id": "seed1", "component_chain_counts": "{'PE': 3, 'PS': 1}"}]).to_csv(seeds / "full_pore_seed_manifest.csv", index=False)
    pd.DataFrame([
        {
            "full_pore_seed_id": "seed1",
            "hold_temperature_mean_K": 523.0,
            "hold_temperature_std_K": 10.0,
            "min_polymer_silica_distance_A": 3.0,
            "polymer_inside_pore_fraction": 1.0,
            "polymer_silica_contact_count_5p0A": 12,
        }
    ]).to_csv(logs / "full_pore_relax_metrics.csv", index=False)
    (pore / "pore_metadata.yaml").write_text(yaml.safe_dump({
        "pore_model_id": "pore",
        "surface": "bare_porems_sio2",
        "hydroxylation_mode": "bare_porems_sio2",
        "pore_diameter_nm": 4.0,
        "pore_length_nm": 6.0,
        "validation": {"element_counts": {"Si": 1, "O": 2, "H": 0}},
    }), encoding="utf-8")
    out = build_premlff_structure_library({
        "paths": {
            "root": str(tmp_path),
            "exports_dir": "data/runs/library/exports",
            "logs_dir": "outputs/runs/library/logs",
        },
        "run": {"run_id": "library"},
        "premlff_structure_library": {"source_run_ids": [run_id]},
    })
    df = pd.read_csv(out)
    assert df.loc[0, "polymer"] == "PE/PS"
    assert df.loc[0, "component_chain_counts"] == "{'PE': 3, 'PS': 1}"
    assert df.loc[0, "library_status"] == "available_relaxed_premlff_seed"
    assert df.loc[0, "min_polymer_silica_distance_A"] == 3.0
    assert (tmp_path / "outputs" / "runs" / "library" / "logs" / "premlff_porems_structure_library.md").exists()
