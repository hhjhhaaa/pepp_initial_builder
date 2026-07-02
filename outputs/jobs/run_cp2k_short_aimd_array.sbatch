#!/bin/bash
#SBATCH --job-name=pepp_cp2k_short_aimd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --array=0-2

module purge
module load __SET_CP2K_MODULE_ON_HPC__
CP2K_CMD=${CP2K_CMD:-cp2k.psmp}

JOB_DIRS=("/home/jinhao/mlff/pepp_initial_builder/data/cp2k_seed_jobs/aimd_0001_silica_patch_only_seed1/short_aimd" "/home/jinhao/mlff/pepp_initial_builder/data/cp2k_seed_jobs/aimd_0002_silica_patch_only_seed2/short_aimd" "/home/jinhao/mlff/pepp_initial_builder/data/cp2k_seed_jobs/aimd_0003_silica_patch_only_seed3/short_aimd")
JOB_DIR="${JOB_DIRS[$SLURM_ARRAY_TASK_ID]}"
cd "$JOB_DIR"
"$CP2K_CMD" -i input.inp -o cp2k.out
