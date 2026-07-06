# Structure Task Boundaries

`pepp_initial_builder` owns structure construction and preprocessing for three separate downstream tasks. The lanes must not be mixed.

## `mlff_direct`

Purpose: structures that will be run directly with a zero-shot machine-learning force field, currently MACE-MH-0.

Output status: zero-shot trajectories are **provisional MLFF labels** only.

Current library path:

```text
data/structure_library/mlff_direct/emc_polymer/<system_id>/
```

Allowed preprocessing: EMC structure generation and classical LAMMPS thermal relaxation. The classical trajectory is not a label source.

Thermal relaxation protocol:

```text
minimize -> Langevin + nve/limit warmup -> NVT heat/cool anneal -> NPT density relaxation -> target-temperature NVT settle
```

This follows the local initial-builder staging style, LAMMPS thermostat/barostat semantics, and polymer-melt equilibration practice where bad contacts are first relaxed before NVT/NPT equilibration. For long-chain production studies, this remains only a cleanup/equilibration stage; true reference labels must come from the `aimd` lane.

Protocol references:

- LAMMPS `fix nve/limit`: useful when starting from configurations with close contacts or overlaps.
- LAMMPS `fix nvt`/`fix npt`: Nose-Hoover canonical and isothermal-isobaric time integration.
- Auhl et al., "Equilibration of Long Chain Polymer Melts in Computer Simulations".
- Zhang et al., "Equilibration of High Molecular-Weight Polymer Melts: A Hierarchical Strategy".

## `aimd`

Purpose: structures selected for CP2K DFT/AIMD or DFT single-point labeling.

Output status: DFT/AIMD energy-force data are the reference labels used for fine-tuning or audit.

Current project paths remain the CP2K/AIMD run-scoped paths configured in `configs/cp2k_aimd.yaml` and related CP2K manifests.

## `fine_tuned_mlff`

Purpose: structures that will be run only after a DFT/AIMD-anchored fine-tuned MLFF exists.

Output status: these can become MLFF reference-label trajectories only when the model provenance points to the fine-tuned checkpoint and the validation gates pass.

Current library path:

```text
data/structure_library/fine_tuned_mlff/
```

## Rules

- Do not generate PE/PP/PS production structures inside `pepp_mlff_finetune`.
- Do not use Packmol-fallback polymer systems as production structures.
- Do not move a structure between lanes by copying files alone; create or update metadata with `structure_task.lane`.
- Do not call zero-shot MACE output a reference label.
- PS entries must include detectable `phenyl_rings` before production use.
- Do not rsync over HPC AIMD run data. Code syncs to HPC must be non-destructive and must exclude `data/`, `outputs/`, `*.out`, `__pycache__/`, `.pytest_cache/`, and generated EMC/LAMMPS artifacts.
- Do not use `rsync --delete` for this project unless explicitly requested for a named path after active AIMD jobs are confirmed stopped.
