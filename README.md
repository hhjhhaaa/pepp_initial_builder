# PE/PP-Silica Data Generation Module

`pepp_initial_builder` is the PE/PP-silica data-generation module. It prepares EMC PE/PP polymer structures, PoreMS silica pores and patches, Packmol-packed full-pore starting structures, CP2K/AIMD labeling inputs, parsed CP2K labels, AIMD train/val/test datasets, manifests, and validation reports.

It does not train MLFF models, run formal MLFF production trajectories, train Graph-SPIB, or do descriptor distillation. The modeling line is EMC for PE/PP polymer generation, Packmol for constrained pore packing, Open Babel for format conversion, LAMMPS for full-pore equilibration, then CP2K/AIMD crops from LAMMPS-relaxed full-pore structures.

The project boundary has three top-level repositories:

```text
/home/jinhao/mlff/pepp_initial_builder  = data generation
/home/jinhao/mlff/pepp_mlff_train       = MLFF fine-tuning and validation
/home/jinhao/mlff/pepp_graph_spib       = Graph-SPIB model training and descriptor mining
```

Internal modules:

```text
src/pepp_initial_builder/polymer/    EMC PE/PP initial structures
src/pepp_initial_builder/pore/       PoreMS full pores and silica patches
src/pepp_initial_builder/mlff_seed/  full pore + PE/PP packing + LAMMPS relax starting structures
src/pepp_initial_builder/cp2k_aimd/  CP2K ENERGY_FORCE / short NVT AIMD labeling datasets
src/pepp_initial_builder/export/     data-generation manifests and summaries
src/pepp_initial_builder/reuse/      /home/jinhao/lmp-proj reuse discovery reports
```

`src/` is the Python packaging source layout, not a separate dependency or runtime environment. Runtime dependencies are declared in `pyproject.toml`; `requirements.txt` only installs the package and test runner.

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

No legacy layers are kept. No Python random-walk polymer builder or internal Packmol substitute is kept. No synthetic label data are generated. No fake CP2K/AIMD/MLFF outputs are allowed.

Full-pore structure method:

```text
1. EMC builds PE/PP chains with PCFF typing and LAMMPS topology.
2. PoreMS provides the hydroxylated cylindrical silica pore.
3. Packmol places whole EMC chain templates inside the pore cylinder:
   r < pore_radius - wall_buffer_A
   end_buffer_A < z < box_lz - end_buffer_A
4. LAMMPS pre-equilibrates the packed full-pore structure before any CP2K crop:
   minimization -> high-temperature NVT anneal -> target-temperature NVT.
5. CP2K/AIMD local crops are cut only from LAMMPS-relaxed full-pore structures
   or later LAMMPS exploration snapshots.
```

The polymer is intentionally packed inside the pore before MD. The LAMMPS stage is not used to wait for chains to diffuse from outside the pore into the pore, because that insertion/adsorption process is too slow for the tiny first-run validation and would make the initial dataset irreproducible. LAMMPS is used to remove packing contacts and equilibrate chain conformations, local density, and polymer-silica contacts at fixed pore geometry.

Default full-pore LAMMPS pre-equilibration parameters:

```text
force field path: EMC PCFF polymer parameters + fixed nonreactive silica host terms
ensemble: polymer NVT, silica fixed by setforce 0 0 0
timestep: 0.5 fs
minimization: fire, etol 1e-6, ftol 1e-8
anneal stage: 650 K NVT
target stage: 523 K NVT

tiny:  10,000 + 10,000 steps = 10 ps, smoke test only
pilot: 400,000 + 1,600,000 steps = 1 ns, first credible pre-equilibrated structures
main:  2,000,000 + 8,000,000 steps = 5 ns, production starting structures
```

The tiny setting is deliberately short so the HPC toolchain can be validated quickly. It is not a claim of structural equilibration. For production analysis, density/contact convergence should be checked from the pilot/main LAMMPS trajectories before increasing CP2K crop volume.

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

`data/` is a generated workspace and is ignored by Git. The repository tracks code, configs, scripts, tests, and small audit text only; generated structures, manifests, CP2K jobs, parsed outputs, and AIMD extxyz files stay local or move by rsync.

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

CP2K/AIMD seed structures are cropped only from LAMMPS-relaxed full-pore PE/PP-silica sources or later LAMMPS exploration snapshots. Raw Packmol full-pore seeds are not accepted as CP2K crop sources. Hand-designed silica patches are kept in the pore workflow for bootstrap geometry checks, but the CP2K seed builder requires full-pore seed/snapshot sources and records source stage, local environment reason, frame provenance, crop boundary treatment, local composition, wall distance, and cell centering metadata.

For tiny CP2K crop validation, local crops are capped at `<=100` atoms. Crop cells are rebuilt as orthorhombic local cells with vacuum padding, atoms are translated near the local cell center, and `PERIODIC XYZ` is used for v0 with the padded cell so CP2K outputs stay compatible with periodic MLFF datasets. If mirror interactions become problematic in measured runs, a later version should switch these crops to a slab or cluster strategy.

If no real CP2K output is present, parsing and dataset building report `not_run_no_cp2k_output`, `insufficient_real_cp2k_frames`, and `usable_for_mlff_training = false`. The repository must not fabricate CP2K outputs, AIMD trajectories, forces, energies, stress, MLFF models, or MLFF production trajectories.
