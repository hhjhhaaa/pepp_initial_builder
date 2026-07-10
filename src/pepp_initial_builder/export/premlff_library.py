from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def _root(config: Dict[str, Any]) -> Path:
    return Path(config.get("paths", {}).get("root", ".")).expanduser()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _polymer_from_composition(composition: str) -> str:
    text = str(composition)
    components = []
    if "PE_HDPE100" in text or "PE100" in text or ("PE" in text and "PE00" not in text):
        components.append("PE")
    if "PP100" in text or ("PP" in text and "PP00" not in text):
        components.append("PP")
    if "PS100" in text or ("PS" in text and "PS00" not in text):
        components.append("PS")
    return "/".join(dict.fromkeys(components)) if components else text


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _pore_metadata(root: Path, run_id: str) -> Dict[str, Any]:
    candidates = list((root / "data" / "runs" / run_id / "mesoporous_silica" / "pore_models").glob("*/pore_metadata.yaml"))
    if not candidates:
        return {}
    return yaml.safe_load(candidates[0].read_text(encoding="utf-8")) or {}


def _host_rows(root: Path, run_id: str, library_id: str, seen_hosts: set[str]) -> List[Dict[str, Any]]:
    manifest = _read_csv(root / "data" / "runs" / run_id / "mesoporous_silica" / "pore_models" / "pore_model_manifest.csv")
    rows: List[Dict[str, Any]] = []
    if manifest.empty:
        return rows
    for record in manifest.to_dict("records"):
        if str(record.get("status", "")) != "available":
            continue
        key = "|".join(str(record.get(name, "")) for name in ["pore_model_id", "pore_diameter_nm", "pore_length_nm", "hydroxylation_mode"])
        if key in seen_hosts:
            continue
        seen_hosts.add(key)
        meta_path = root / "data" / "runs" / run_id / "mesoporous_silica" / "pore_models" / str(record.get("pore_model_id", "")) / "pore_metadata.yaml"
        pore_meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        counts = ((pore_meta or {}).get("validation", {}) or {}).get("element_counts", {})
        rows.append(
            {
                "library_id": library_id,
                "source_run_id": run_id,
                "full_pore_seed_id": record.get("pore_model_id", ""),
                "polymer": "SiO2_host",
                "composition": record.get("hydroxylation_mode", "bare_porems_sio2"),
                "component_chain_counts": "",
                "polymer_architecture": "fixed_bare_porems_sio2_host",
                "pe_variant": "",
                "pp_variant": "",
                "ps_variant": "",
                "relaxed_extxyz_path": record.get("pore_model_extxyz_path", ""),
                "atom_roles_path": "",
                "library_status": "available_porems_host_structure",
                "lammps_time_ps": "",
                "hold_temperature_mean_K": "",
                "hold_temperature_std_K": "",
                "total_energy": "",
                "polymer_inside_pore_fraction": "",
                "min_polymer_silica_distance_A": "",
                "mean_wall_distance_A": "",
                "polymer_silica_contact_count_3p5A": "",
                "polymer_silica_contact_count_5p0A": "",
                "pe_near_wall_fraction": "",
                "pp_near_wall_fraction": "",
                "ps_near_wall_fraction": "",
                "pore_model_id": record.get("pore_model_id", ""),
                "pore_diameter_nm": record.get("pore_diameter_nm", ""),
                "pore_length_nm": record.get("pore_length_nm", ""),
                "wall_model": pore_meta.get("surface", record.get("hydroxylation_mode", "bare_porems_sio2")) if pore_meta else record.get("hydroxylation_mode", "bare_porems_sio2"),
                "hydroxylation_mode": record.get("hydroxylation_mode", ""),
                "pore_si_count": counts.get("Si", ""),
                "pore_o_count": counts.get("O", ""),
                "pore_h_count": counts.get("H", ""),
                "density_policy": "host-only PoreMS structure; no polymer density gate applies",
                "library_note": "PoreMS native bare_porems_sio2 host structure used as the fixed pore for polymer-in-pore relaxation",
            }
        )
    return rows


def _library_rows(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    root = _root(config)
    rows: List[Dict[str, Any]] = []
    library_id = config.get("run", {}).get("run_id", "premlff_structure_library")
    include_hosts = bool(config.get("premlff_structure_library", {}).get("include_pore_hosts", True))
    seen_hosts: set[str] = set()
    for run_id in config.get("premlff_structure_library", {}).get("source_run_ids", []):
        if include_hosts:
            rows.extend(_host_rows(root, run_id, library_id, seen_hosts))
        snapshots = _read_csv(root / "data" / "runs" / run_id / "exports" / "full_pore_snapshot_manifest.csv")
        metrics = _read_csv(root / "outputs" / "runs" / run_id / "logs" / "full_pore_relax_metrics.csv")
        seeds = _read_csv(root / "data" / "runs" / run_id / "mlff_seed" / "structures" / "full_pore_seed_manifest.csv")
        if snapshots.empty:
            rows.append({"source_run_id": run_id, "library_status": "missing_snapshot_manifest"})
            continue
        merged = snapshots.merge(metrics, on="full_pore_seed_id", how="left", suffixes=("", "_metric")) if not metrics.empty else snapshots
        if not seeds.empty and "component_chain_counts" in seeds.columns:
            merged = merged.merge(seeds[["full_pore_seed_id", "component_chain_counts"]], on="full_pore_seed_id", how="left")
        pore_meta = _pore_metadata(root, run_id)
        validation = pore_meta.get("validation", {}) if pore_meta else {}
        counts = validation.get("element_counts", {}) if validation else {}
        for record in merged.to_dict("records"):
            composition = record.get("composition", record.get("composition_metric", ""))
            usable = bool(record.get("usable_for_cp2k_crop", False))
            source_stage = str(record.get("source_stage", ""))
            status = "available_relaxed_premlff_seed" if source_stage == "lammps_relaxed_full_pore" and usable else "review"
            rows.append(
                {
                    "library_id": library_id,
                    "source_run_id": run_id,
                    "full_pore_seed_id": record.get("full_pore_seed_id", ""),
                    "polymer": _polymer_from_composition(str(composition)),
                    "composition": composition,
                    "component_chain_counts": record.get("component_chain_counts", ""),
                    "polymer_architecture": record.get("polymer_architecture", record.get("polymer_architecture_metric", "")),
                    "pe_variant": record.get("pe_variant", ""),
                    "pp_variant": record.get("pp_variant", ""),
                    "ps_variant": record.get("ps_variant", ""),
                    "relaxed_extxyz_path": record.get("snapshot_path", record.get("relaxed_extxyz_path", "")),
                    "atom_roles_path": record.get("atom_roles_path", ""),
                    "library_status": status,
                    "lammps_time_ps": record.get("time_ps", record.get("time_ps_metric", "")),
                    "hold_temperature_mean_K": record.get("hold_temperature_mean_K", ""),
                    "hold_temperature_std_K": record.get("hold_temperature_std_K", ""),
                    "total_energy": record.get("total_energy", ""),
                    "polymer_inside_pore_fraction": record.get("polymer_inside_pore_fraction", ""),
                    "min_polymer_silica_distance_A": record.get("min_polymer_silica_distance_A", ""),
                    "mean_wall_distance_A": record.get("mean_wall_distance_A", ""),
                    "polymer_silica_contact_count_3p5A": record.get("polymer_silica_contact_count_3p5A", ""),
                    "polymer_silica_contact_count_5p0A": record.get("polymer_silica_contact_count_5p0A", ""),
                    "pe_near_wall_fraction": record.get("pe_near_wall_fraction", ""),
                    "pp_near_wall_fraction": record.get("pp_near_wall_fraction", ""),
                    "ps_near_wall_fraction": record.get("ps_near_wall_fraction", ""),
                    "pore_model_id": pore_meta.get("pore_model_id", ""),
                    "pore_diameter_nm": pore_meta.get("pore_diameter_nm", ""),
                    "pore_length_nm": pore_meta.get("pore_length_nm", ""),
                    "wall_model": pore_meta.get("surface", "bare_porems_sio2"),
                    "hydroxylation_mode": pore_meta.get("hydroxylation_mode", ""),
                    "pore_si_count": counts.get("Si", ""),
                    "pore_o_count": counts.get("O", ""),
                    "pore_h_count": counts.get("H", ""),
                    "density_policy": "bulk density is calibrated separately; confined pore density is not a hard acceptance gate",
                    "library_note": "PoreMS native bare_porems_sio2 output; no wall-chemistry post-processing was applied",
                }
            )
    return rows



def _markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for record in df[columns].fillna("").to_dict("records"):
        body.append("| " + " | ".join(str(record.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def build_premlff_structure_library(config: Dict[str, Any]) -> Path:
    root = _root(config)
    paths = config.get("paths", {})
    exports_dir = _resolve(root, paths.get("exports_dir", "data/exports"))
    logs_dir = _resolve(root, paths.get("logs_dir", "outputs/logs"))
    exports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    rows = _library_rows(config)
    df = pd.DataFrame(rows)
    out = exports_dir / "premlff_porems_structure_library.csv"
    df.to_csv(out, index=False)
    available = int((df.get("library_status", pd.Series(dtype=str)) == "available_relaxed_premlff_seed").sum()) if not df.empty else 0
    host_available = int((df.get("library_status", pd.Series(dtype=str)) == "available_porems_host_structure").sum()) if not df.empty else 0
    lines = [
        "# Pre-MLFF PoreMS Structure Library",
        "",
        f"library_id: {config.get('run', {}).get('run_id', 'premlff_structure_library')}",
        "",
        f"Available relaxed polymer-in-pore entries: {available}/{len(df)}",
        f"Available PoreMS host entries: {host_available}",
        "",
        "Density policy: bulk polymer density is calibrated separately by LAMMPS thermal relaxation; confined polymer-in-pore structures are accepted by dynamics/QC, not by a hard nominal pore-density gate.",
        "",
        "Wall chemistry policy: PoreMS native bare_porems_sio2 output is used directly as the bare oxide wall; no wall-chemistry post-processing is applied.",
        "",
    ]
    if not df.empty:
        cols = ["polymer", "composition", "component_chain_counts", "library_status", "lammps_time_ps", "hold_temperature_mean_K", "min_polymer_silica_distance_A", "polymer_inside_pore_fraction", "polymer_silica_contact_count_5p0A", "relaxed_extxyz_path"]
        lines.append(_markdown_table(df, cols))
        lines.append("")
    report = logs_dir / "premlff_porems_structure_library.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return out
