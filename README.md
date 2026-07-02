# PE/PP All-Atom Initial Builder

This module creates all-atom C/H PE/PP initial structures, topology tables, metadata, validation reports, and an MLFF start manifest. It does not modify `/home/jinhao/mlff/pepp_graph_spib`.

Boundary: no MLFF production, no formal LAMMPS production, no Graph-SPIB graph windows, no future labels, no transport-property calculation, no ML training, no `rho_eq` calculation, and no `0.95/1.00/1.05 rho_eq` structures. Any LAMMPS/MD here is cleanup-only and is not training data.

Workflow:

```text
pepp_initial_builder -> pepp_mlff_runner -> pepp_graph_spib
```

EMC is discovered and attempted first, but v0 only accepts EMC output if it can be verified as explicit all-atom C/H PE/PP topology with usable segment-center mapping. Otherwise the module records a structured EMC failure and uses Python all-atom topology plus Packmol coordinates.

Packmol output is coordinates only. Bonds, angles, atom metadata, chain metadata, and segment metadata come from the Python all-atom builder or verified EMC output. The module never infers polymer bonds from Packmol PDB files using distance guessing, RDKit, OpenBabel, or MDAnalysis.

`system_id` does not contain rho because this module only creates base initial structures. Later `pepp_mlff_runner` is responsible for MLFF NPT, `rho_eq`, density perturbation, and production trajectory generation.

Run from `/home/jinhao/mlff/pepp_initial_builder`:

```bash
python scripts/check_env.py
python scripts/discover_local_tools.py
python scripts/generate_base_matrix.py --config configs/initial_builder.yaml --tiny
python scripts/build_initial_structures.py --config configs/initial_builder.yaml --tiny --max-systems 3
python scripts/write_lammps_cleanup_inputs.py --config configs/initial_builder.yaml --tiny --max-systems 3
python scripts/validate_initial_structures.py --config configs/initial_builder.yaml --tiny
python scripts/export_mlff_start_manifest.py --config configs/initial_builder.yaml
pytest -q
```

Forbidden production trajectory names: `prod.lammpstrj`, `production.lammpstrj`. Cleanup diagnostic dumps may only use `cleanup_check.lammpstrj` and are not exported as MLFF training trajectories.

## PoreMS / AIMD Local Structure Workflow

The module also contains a second workflow for silica pore structure preparation:

```text
PoreMS silica pore / wall structures
-> AIMD local C/H/O/Si training structures
-> full-pore MLFF exploration seed structures
```

This workflow only writes structures, metadata, manifests, and validation reports. It does not run CP2K, does not generate AIMD trajectories, does not train MLFF models, and does not run MLFF production.

Run the discovery and manifest-producing steps with:

```bash
python scripts/discover_porems.py --config configs/aimd_pore_builder.yaml
python scripts/build_porems_pores.py --config configs/aimd_pore_builder.yaml --tiny
python scripts/crop_silica_patches.py --config configs/aimd_pore_builder.yaml --tiny
python scripts/build_aimd_local_structures.py --config configs/aimd_pore_builder.yaml --tiny
python scripts/build_full_pore_seed_structures.py --config configs/aimd_pore_builder.yaml --tiny
python scripts/write_cp2k_structure_inputs.py --config configs/aimd_pore_builder.yaml
python scripts/validate_pore_structures.py --config configs/aimd_pore_builder.yaml
python scripts/validate_aimd_local_structures.py --config configs/aimd_pore_builder.yaml
python scripts/validate_full_pore_seed_structures.py --config configs/aimd_pore_builder.yaml
python scripts/export_pore_aimd_manifests.py --config configs/aimd_pore_builder.yaml
```

If PoreMS is unavailable, `build_porems_pores.py` writes `failed_porems_not_available` and does not fabricate pore models. A user may place manually generated pore models under `data/porems_models/manual_*`; those are marked with `source: manual_user_input`.
