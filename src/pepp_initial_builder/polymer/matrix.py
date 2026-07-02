from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from pepp_initial_builder.common.paths import ensure_dirs, project_root


def sid(pe: float, pp: float, n: int, seed: int) -> str:
    return f"pepp_PE{int(round(pe * 100)):02d}_PP{int(round(pp * 100)):02d}_N{n}_seed{seed}"


def estimated_atoms(chain_type: str, n: int) -> int:
    if chain_type == "PE":
        return 3 * n + 2
    if n % 2:
        raise ValueError("PP chain_length_backbone must be even for -CH2-CH(CH3)- repeat pattern")
    return int(4.5 * n + 2)


def matrix_rows(config: Dict[str, Any], mode: str = "matrix") -> List[Dict[str, Any]]:
    matrix = config[mode]
    total = int(matrix["total_backbone_carbons"])
    rows: List[Dict[str, Any]] = []
    for pe, pp in matrix["compositions"]:
        for n_value in matrix["chain_lengths_backbone"]:
            n = int(n_value)
            if pp > 0 and n % 2:
                raise ValueError("PP chain_length_backbone must be even for -CH2-CH(CH3)- repeat pattern")
            if total % n:
                raise ValueError("total_backbone_carbons must be divisible by chain_length_backbone")
            total_chains = total // n
            n_pe = int(round(total_chains * float(pe)))
            n_pp = total_chains - n_pe
            pe_backbone = n_pe * n
            pp_backbone = n_pp * n
            actual = pe_backbone + pp_backbone
            for seed in matrix["seeds"]:
                rows.append(
                    {
                        "system_id": sid(float(pe), float(pp), n, int(seed)),
                        "pe_fraction_target": float(pe),
                        "pp_fraction_target": float(pp),
                        "pe_fraction_actual": pe_backbone / actual if actual else 0.0,
                        "pp_fraction_actual": pp_backbone / actual if actual else 0.0,
                        "chain_length_backbone": n,
                        "seed": int(seed),
                        "total_backbone_carbons_target": total,
                        "total_backbone_carbons_actual": actual,
                        "n_pe_chains": n_pe,
                        "n_pp_chains": n_pp,
                        "n_pe_backbone_carbons": pe_backbone,
                        "n_pp_backbone_carbons": pp_backbone,
                        "initial_packing_density_g_cm3": float(config["density"]["initial_packing_density_g_cm3"]),
                        "planned_downstream_density_scales": json.dumps(matrix["planned_downstream_density_scales"]),
                        "target_temperature_K_for_later_mlff": float(config["conditions_for_later_mlff"]["target_temperature_K"]),
                        "target_pressure_atm_for_later_mlff": float(config["conditions_for_later_mlff"]["target_pressure_atm"]),
                        "estimated_total_atoms": n_pe * estimated_atoms("PE", n) + n_pp * estimated_atoms("PP", n),
                        "builder_status": "pending",
                        "cleanup_status": "not_run",
                    }
                )
    return rows


def write_matrix(config: Dict[str, Any], mode: str):
    ensure_dirs(config)
    rows = matrix_rows(config, mode)
    out = project_root(config) / config["paths"]["matrix_dir"]
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "base_initial_matrix.csv"
    json_path = out / "base_initial_matrix.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path
