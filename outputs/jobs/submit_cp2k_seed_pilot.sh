#!/bin/bash
set -euo pipefail
sbatch run_cp2k_sp_array.sbatch
sbatch run_cp2k_short_aimd_array.sbatch
