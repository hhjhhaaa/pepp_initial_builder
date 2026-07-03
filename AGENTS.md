# Agent Constraints

This repository is the local WSL checkout for PE/PP-silica initial data generation.

## Workspace

- Primary workspace: `/home/jinhao/mlff/pepp_initial_builder`.
- Do work in this local WSL checkout unless the user explicitly asks for `ssh nusri`.
- `/mnt/c/Users/Dog of Roman/Documents/PEmixure` is not the canonical git checkout.
- Before editing, run `pwd`, `git rev-parse --show-toplevel`, and `git status --short`.

## Refactor Policy

- Cleanup refactors are allowed to be breaking.
- Do not keep legacy path compatibility unless the user explicitly asks for it.
- Do not add migration fallback readers for old directory names.
- If a path is renamed, new code should read and write only the new configured path.

## Path Semantics

- Production data must be run-scoped: `data/runs/<run_id>/...`.
- Production outputs must be run-scoped: `outputs/runs/<run_id>/...`.
- New run-scoped mesoporous silica paths are:

```text
data/runs/<run_id>/mesoporous_silica/pore_models
data/runs/<run_id>/mesoporous_silica/surface_patches
data/runs/<run_id>/mesoporous_silica/surface_sites
```

- `export_all_manifests` and `summarize_outputs` must respect configured `paths.exports_dir`, `paths.logs_dir`, `paths.figures_dir`, and `paths.jobs_dir`.
- Run-aware configs must not write back to global `data/exports` or `outputs/logs`.

## Active Run Protection

Never delete, move, rename, or overwrite these active run directories unless the user explicitly asks:

```text
data/runs/pilot_20260703_fullpore_relaxed_cp2k
outputs/runs/pilot_20260703_fullpore_relaxed_cp2k
```

Cleanup candidates may be listed only unless the user explicitly asks to delete:

```text
data_backup_minloop_/
slurm-560*.out
**/__pycache__/
src/pepp_initial_builder.egg-info/
data/pore/
data/mlff_seed/
data/cp2k_aimd/
```

## Testing

- Verified local WSL test command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/jinhao/miniforge3/bin/python -m pytest -q -s tests
```

- `/usr/bin/python3` is Python 3.10 but does not have pytest installed in this checkout.
- The base conda pytest can fail during capture cleanup here; keep `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and `-s`.
- Do not use a remote/HPC Python environment unless the user explicitly asks to work on the remote.
