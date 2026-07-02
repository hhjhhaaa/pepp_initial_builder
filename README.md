# PE/PP-Silica Data Generation Module

`pepp_initial_builder` is the PE/PP-silica data-generation module. It creates all-atom C/H PE/PP initial structures, PoreMS silica pores, silica patches, AIMD seed structures, CP2K-ready structures and inputs, HPC CP2K job scripts, parsed CP2K labels, AIMD train/val/test extxyz datasets, and dataset manifests.

It does not train MLFF models, fine-tune MACE or SevenNet, run formal MLFF production, train Graph-SPIB, or do descriptor distillation. Any LAMMPS/MD here is cleanup-only and is not training data.

The project boundary has three top-level modules:

```text
/home/jinhao/mlff/pepp_initial_builder  = data generation module
/home/jinhao/mlff/pepp_mlff_train       = MLFF training module
/home/jinhao/mlff/pepp_graph_spib       = Graph-SPIB model training and descriptor mining
```

Workflow:

```text
pepp_initial_builder
-> pepp_mlff_train
-> pepp_graph_spib
```

EMC is discovered and attempted first, but v0 only accepts EMC output if it can be verified as explicit all-atom C/H PE/PP topology with usable segment-center mapping. Otherwise the module records a structured EMC failure and uses Python all-atom topology plus Packmol coordinates.

Packmol output is coordinates only. Bonds, angles, atom metadata, chain metadata, and segment metadata come from the Python all-atom builder or verified EMC output. The module never infers polymer bonds from Packmol PDB files using distance guessing, RDKit, OpenBabel, or MDAnalysis.

`system_id` does not contain rho because this module only creates base initial/data-generation structures. Later `pepp_mlff_train` is responsible for MLFF pretrain loading, fine-tuning, validation, and short stability checks; formal MLFF production trajectories are outside this module.

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

The module also contains a workflow for silica pore structure preparation:

```text
PoreMS silica pore / wall structures
-> AIMD local C/H/O/Si training structures
-> full-pore MLFF exploration seed structures
```

This workflow writes structures, metadata, manifests, and validation reports. It does not train MLFF models and does not run MLFF production.

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

## CP2K / AIMD Dataset Workflow

CP2K data production remains inside this data-generation module. It starts from `data/aimd_local_structures/aimd_local_manifest.csv`, writes CP2K ENERGY_FORCE and short AIMD inputs, writes Slurm job scripts, parses real CP2K outputs, and exports the AIMD dataset manifest consumed by `pepp_mlff_train`.

Run the CP2K/AIMD dataset steps with:

```bash
python scripts/discover_reusable_lmp_proj_modules.py --config configs/cp2k_dataset.yaml
python scripts/write_cp2k_label_inputs.py --config configs/cp2k_dataset.yaml --tiny
python scripts/make_hpc_cp2k_jobs.py --config configs/cp2k_dataset.yaml --tiny
python scripts/parse_cp2k_outputs.py --config configs/cp2k_dataset.yaml
python scripts/build_aimd_dataset.py --config configs/cp2k_dataset.yaml
python scripts/validate_aimd_dataset.py --config configs/cp2k_dataset.yaml
python scripts/export_aimd_dataset_manifest.py --config configs/cp2k_dataset.yaml
```

The generated CP2K inputs do not hard-code a CP2K data directory. On HPC, load the correct CP2K module and set the CP2K data path according to that installation before submitting jobs.

`parse_cp2k_outputs.py` only accepts existing CP2K output with parseable energy and force labels. If no real CP2K output is present, downstream dataset manifests are marked as skipped/no valid dataset rather than fabricated.

Key outputs:

```text
data/exports/mlff_start_manifest.csv
data/aimd_exports/aimd_structure_manifest.csv
data/aimd_exports/full_pore_exploration_seed_manifest.csv
data/exports/aimd_dataset_manifest.csv
data/aimd_dataset/train.extxyz
data/aimd_dataset/val.extxyz
data/aimd_dataset/test.extxyz
```
