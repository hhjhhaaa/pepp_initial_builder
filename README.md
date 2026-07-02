# PE/PP-Silica Data Generation Module

`pepp_initial_builder` is the PE/PP-silica data-generation module. It prepares polymer structures, PoreMS silica pores and patches, full-pore MLFF starting structures, CP2K/AIMD labeling inputs, parsed CP2K labels, AIMD train/val/test datasets, manifests, and validation reports.

It does not train MLFF models, run formal MLFF production trajectories, train Graph-SPIB, or do descriptor distillation. LAMMPS relax here is used only to prepare full-pore starting structures for later MLFF production; CP2K/AIMD is used to produce DFT-level labels for MLFF training.

The project boundary has three top-level repositories:

```text
/home/jinhao/mlff/pepp_initial_builder  = data generation
/home/jinhao/mlff/pepp_mlff_train       = MLFF fine-tuning and validation
/home/jinhao/mlff/pepp_graph_spib       = Graph-SPIB model training and descriptor mining
```

Internal modules:

```text
src/pepp_initial_builder/polymer/    PE/PP initial structures
src/pepp_initial_builder/pore/       PoreMS full pores and silica patches
src/pepp_initial_builder/mlff_seed/  full pore + PE/PP packing + LAMMPS relax starting structures
src/pepp_initial_builder/cp2k_aimd/  CP2K ENERGY_FORCE / short NVT AIMD labeling datasets
src/pepp_initial_builder/export/     data-generation manifests and summaries
src/pepp_initial_builder/reuse/      /home/jinhao/lmp-proj reuse discovery reports
```

Install:

```bash
pip install -e .
```

Main workflow:

```bash
python scripts/check_env.py

python scripts/polymer/generate_matrix.py --config configs/polymer.yaml --tiny
python scripts/polymer/build_structures.py --config configs/polymer.yaml --tiny --max-systems 3
python scripts/polymer/validate.py --config configs/polymer.yaml --tiny
python scripts/polymer/export_manifest.py --config configs/polymer.yaml

python scripts/pore/discover_porems.py --config configs/pore.yaml
python scripts/pore/build_pore_models.py --config configs/pore.yaml --tiny
python scripts/pore/crop_patches.py --config configs/pore.yaml --tiny
python scripts/pore/validate.py --config configs/pore.yaml --tiny
python scripts/pore/export_manifest.py --config configs/pore.yaml

python scripts/mlff_seed/build_packmol_inputs.py --config configs/mlff_seed.yaml --tiny
python scripts/mlff_seed/pack_polymer_into_pore.py --config configs/mlff_seed.yaml --tiny
python scripts/mlff_seed/write_lammps_relax.py --config configs/mlff_seed.yaml --tiny
python scripts/mlff_seed/validate.py --config configs/mlff_seed.yaml --tiny
python scripts/mlff_seed/export_manifest.py --config configs/mlff_seed.yaml

python scripts/cp2k_aimd/discover_lmp_proj_reuse.py --config configs/cp2k_aimd.yaml
python scripts/cp2k_aimd/build_seed_structures.py --config configs/cp2k_aimd.yaml --tiny
python scripts/cp2k_aimd/write_cp2k_inputs.py --config configs/cp2k_aimd.yaml --tiny
python scripts/cp2k_aimd/make_hpc_jobs.py --config configs/cp2k_aimd.yaml --tiny
python scripts/cp2k_aimd/parse_cp2k_outputs.py --config configs/cp2k_aimd.yaml
python scripts/cp2k_aimd/build_dataset.py --config configs/cp2k_aimd.yaml
python scripts/cp2k_aimd/validate_dataset.py --config configs/cp2k_aimd.yaml
python scripts/cp2k_aimd/export_dataset_manifest.py --config configs/cp2k_aimd.yaml

python scripts/export/export_all_manifests.py
python scripts/export/summarize_outputs.py
pytest -q
```

Old top-level scripts are kept as compatibility wrappers and still accept the original configs.

Important outputs:

```text
data/exports/polymer_initial_manifest.csv
data/exports/pore_model_manifest.csv
data/exports/silica_patch_manifest.csv
data/exports/mlff_seed_manifest.csv
data/exports/aimd_seed_manifest.csv
data/exports/cp2k_job_manifest.csv
data/exports/aimd_dataset_manifest.csv
```

CP2K is not assumed to run in local WSL. Slurm scripts contain:

```bash
module purge
module load __SET_CP2K_MODULE_ON_HPC__
# export CP2K_DATA_DIR="__SET_CP2K_DATA_DIR_ON_HPC__"
CP2K_CMD=${CP2K_CMD:-cp2k.psmp}
```

Recommended HPC tiny first-run order:

1. Submit only 1-3 `ENERGY_FORCE` jobs with `outputs/jobs/submit_cp2k_sp_tiny.sh` after editing the Slurm array range on the HPC side.
2. Check each `cp2k.out`, then rsync the small outputs back and run `scripts/cp2k_aimd/parse_cp2k_outputs.py`.
3. Submit the remaining SP jobs only after the parser reports real frames with eV energies and eV/Angstrom forces.
4. Submit `outputs/jobs/submit_cp2k_short_aimd_tiny.sh` only after SP inputs, module loading, basis/potential lookup, normal-end detection, and parsing are confirmed.

The combined `outputs/jobs/submit_cp2k_seed_tiny.sh` exists for convenience, but the split SP-then-short-AIMD path is the recommended validation route.

For local AIMD patches, tiny validation is capped at `<=100` atoms. Patch cells are rebuilt as orthorhombic local cells with vacuum padding, atoms are translated near the local cell center, and `PERIODIC XYZ` is used for v0 with the padded cell so CP2K outputs stay compatible with periodic MLFF datasets. If mirror interactions become problematic in measured runs, a later version should switch these patches to a slab or cluster strategy.

If no real CP2K output is present, parsing and dataset building report `not_run_no_cp2k_output`, `insufficient_real_cp2k_frames`, and `usable_for_mlff_training = false`. The repository must not fabricate CP2K outputs, AIMD trajectories, forces, energies, stress, MLFF models, or MLFF production trajectories.
