from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from pepp_initial_builder.cp2k_aimd.config import ensure_dirs, p, read_rows, write_rows


def array_script(label_mode: str, rows: List[Dict[str, str]], config: Dict[str, Any]) -> str:
    hpc = config["hpc"]
    mode_rows = [row for row in rows if row.get("label_mode") == label_mode]
    job_dirs = " ".join(f'"{row["job_dir"]}"' for row in mode_rows)
    time_line = f"#SBATCH --time={hpc['time_sp']}\n" if label_mode == "sp_force" else ""
    throttle_key = "sp_array_throttle" if label_mode == "sp_force" else "aimd_array_throttle"
    throttle = int(hpc.get(throttle_key, 0) or 0)
    array_spec = f"0-{max(len(mode_rows) - 1, 0)}" + (f"%{throttle}" if throttle > 0 and mode_rows else "")
    selected_note = "# Short AIMD jobs are generated only from data/runs/<run_id>/exports/selected_short_aimd_manifest.csv.\n" if label_mode == "short_aimd" else ""
    return f"""#!/bin/bash
#SBATCH --job-name=pepp_cp2k_{label_mode}
#SBATCH --partition={hpc.get('partition', 'batch')}
#SBATCH --nodes={int(hpc['nodes'])}
#SBATCH --ntasks-per-node={int(hpc['ntasks_per_node'])}
#SBATCH --cpus-per-task={int(hpc['cpus_per_task'])}
{time_line.rstrip()}
#SBATCH --array={array_spec}

module purge
module load {hpc['cp2k_module_placeholder']}
# export CP2K_DATA_DIR="__SET_CP2K_DATA_DIR_ON_HPC__"
CP2K_CMD=${{CP2K_CMD:-{hpc['cp2k_command_default']}}}
{selected_note}

JOB_DIRS=({job_dirs})
JOB_DIR="${{JOB_DIRS[$SLURM_ARRAY_TASK_ID]}}"
cd "$JOB_DIR"
"$CP2K_CMD" -i input.inp -o cp2k.out
"""


def verify_cp2k_module_script(config: Dict[str, Any]) -> str:
    hpc = config["hpc"]
    return f"""#!/bin/bash
set -euo pipefail
module purge
module load {hpc['cp2k_module_placeholder']}
which {hpc['cp2k_command_default']}
{hpc['cp2k_command_default']} --version
"""


def make_hpc_cp2k_jobs(config: Dict[str, Any], mode: str = "tiny") -> Path:
    ensure_dirs(config)
    rows = [row for row in read_rows(p(config, "cp2k_jobs_dir") / "cp2k_label_input_manifest.csv") if row.get("status") == "cp2k_input_written_no_cp2k_run"]
    out_rows = []
    for label_mode, filename in [("sp_force", "run_cp2k_sp_array.sbatch"), ("short_aimd", "run_cp2k_short_aimd_array.sbatch")]:
        script = p(config, "jobs_dir") / filename
        script.write_text(array_script(label_mode, rows, config), encoding="utf-8")
        script.chmod(0o755)
        out_rows.append({"label_mode": label_mode, "script_path": str(script), "job_count": sum(1 for row in rows if row.get("label_mode") == label_mode)})
    verify = p(config, "jobs_dir") / "verify_cp2k_module.sh"
    verify.write_text(verify_cp2k_module_script(config), encoding="utf-8")
    verify.chmod(0o755)
    for name in ["tiny", "pilot"]:
        sp_submit = p(config, "jobs_dir") / f"submit_cp2k_sp_{name}.sh"
        sp_submit.write_text("#!/bin/bash\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\nsbatch run_cp2k_sp_array.sbatch\n", encoding="utf-8")
        sp_submit.chmod(0o755)
        aimd_submit = p(config, "jobs_dir") / f"submit_cp2k_short_aimd_{name}.sh"
        aimd_submit.write_text("#!/bin/bash\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\nsbatch run_cp2k_short_aimd_array.sbatch\n", encoding="utf-8")
        aimd_submit.chmod(0o755)
        submit = p(config, "jobs_dir") / f"submit_cp2k_seed_{name}.sh"
        submit.write_text("#!/bin/bash\nset -euo pipefail\ncd \"$(dirname \"$0\")\"\nsbatch run_cp2k_sp_array.sbatch\nsbatch run_cp2k_short_aimd_array.sbatch\n", encoding="utf-8")
        submit.chmod(0o755)
    return write_rows(p(config, "jobs_dir") / "cp2k_hpc_job_manifest.csv", out_rows)
