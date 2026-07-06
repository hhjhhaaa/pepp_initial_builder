# Agent Constraints

This repository is the PE/PP/PS-silica initial data builder. Follow these rules before making code, config, path, cleanup, or sync changes.

## Workspaces

- Remote execution target: `ssh nusri`.
- Remote checkout used for this project: `/public/home/jinhao.hu/peppmixture/pepp_initial_builder`.
- Local WSL mirror used for handoff/sync: `/mnt/c/Users/Dog of Roman/Documents/PEmixure`.
- Do not assume a different checkout such as `/home/jinhao/mlff/pepp_initial_builder` or `/public/home/jinhao.hu/mlff/pepp_initial_builder` unless it exists and the user explicitly asks to use it.
- Before editing, run `pwd`, `git rev-parse --show-toplevel`, and `git status --short` in the target checkout.

## Python Environment

- On `nusri`, use `/public/home/jinhao.hu/.conda/envs/peppmixure/bin/python`.
- Run tests with:

```bash
/public/home/jinhao.hu/.conda/envs/peppmixure/bin/python -m pytest -q
```

- Do not use system `/usr/bin/python3`; it is Python 3.6 and is not valid for this project.

## Active Run Protection

Never delete, move, rename, or rsync over these active run directories unless the user explicitly says to:

```text
data/runs/pilot_20260703_fullpore_relaxed_cp2k
outputs/runs/pilot_20260703_fullpore_relaxed_cp2k
```

Cleanup candidates may be listed only. Do not delete them without a separate explicit user command:

```text
data_backup_minloop_/
slurm-560*.out
**/__pycache__/
src/pepp_initial_builder.egg-info/
data/pore/
data/mlff_seed/
data/cp2k_aimd/
```

## Current Chemistry Scope

- The active small-interface chemistry is **PE/PP/PS + SiO2**, not PE/PP/PC.
- PS means polystyrene-like chains/fragments with phenyl side groups.
- PC/polycarbonate/BPA/carbonate-anchor structures are historical mistakes for this task and must not be regenerated, submitted, parsed, or documented as active data.
- Do not restore legacy hand-built small-anchor interfaces or configs:

```text
configs/cp2k_small_anchors.yaml
scripts/cp2k_aimd/build_small_anchors.py
src/pepp_initial_builder/cp2k_aimd/small_anchor_builder.py
tests/cp2k_aimd/test_small_anchor_builder.py
configs/mlff_aimd_anchor.yaml
src/pepp_initial_builder/export/anchor_scaffold.py
```

- The current small-system review entry point is:

```text
configs/cp2k_closed_small_ps_anchors.yaml
scripts/cp2k_aimd/build_closed_small_anchors.py
data/runs/pilot_20260706_closed_small_ps_anchors
outputs/runs/pilot_20260706_closed_small_ps_anchors
```

- The local WSL structure handoff/check directory is:

```text
/mnt/c/Users/Dog of Roman/Documents/PEmixure/review_closed_small_ps_anchors
```

- Before submitting CP2K for small-interface work, verify the active manifests and reports contain no PC/polycarbonate/BPA/carbonate/PE_PC/PP_PC/PE_PP_PC/pepppc terms except incidental `CP2K` strings.
- The small PS anchor geometry should be slab-like: silica local tangent plane aligned to XY, surface normal along +Z, polymer fragments placed laterally/parallel near the surface, and polymer-silica nearest heavy-atom distance around 2.8 A.
- Silica cluster boundaries must be chemically closed by H/OH capping; `undercoordinated_Si_after_capping` and `uncapped_O_after_capping` must both be zero before CP2K input generation.

## Path Semantics

- Production data must be run-scoped: `data/runs/<run_id>/...`.
- Production outputs must be run-scoped: `outputs/runs/<run_id>/...`.
- New run-scoped mesoporous silica paths are:

```text
data/runs/<run_id>/mesoporous_silica/pore_models
data/runs/<run_id>/mesoporous_silica/surface_patches
data/runs/<run_id>/mesoporous_silica/surface_sites
```

- Existing run manifests under old names must remain readable:

```text
data/runs/<run_id>/pore/porems_models
data/runs/<run_id>/pore/silica_patches
```

- `export_all_manifests` and `summarize_outputs` must respect the configured `paths.exports_dir`, `paths.logs_dir`, `paths.figures_dir`, and `paths.jobs_dir`.
- Run-aware configs must not write back to global `data/exports` or `outputs/logs`.
- Do not change path semantics without adding or updating tests for both the new write path and old manifest compatibility.

## Sync Rules

- Use non-destructive sync by default. Do not pass `--delete` unless the user explicitly asks.
- When syncing code back to local WSL, exclude generated runtime artifacts:

```text
.git/
data/
outputs/
data_backup_minloop_/
*.out
__pycache__/
*.pyc
.pytest_cache/
```

## Final Checks

Before reporting completion:

- Run pytest in the `peppmixure` conda environment.
- List cleanup candidates only; do not delete them.
- State which directories are safe to archive, which active run directories must not be deleted, and where the current run data entry points are.
