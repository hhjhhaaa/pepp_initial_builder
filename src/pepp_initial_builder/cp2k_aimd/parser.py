from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml

from pepp_initial_builder.cp2k_aimd.config import ensure_dirs, p, read_rows, write_rows
from pepp_initial_builder.cp2k_aimd.extxyz_writer import FORCE_AU_TO_EV_A, HARTREE_TO_EV, read_xyz_frames, read_xyz_like, write_extxyz_frames


def parse_energy_hartree(text: str) -> float | None:
    for pattern in [r"ENERGY\|\s+Total FORCE_EVAL.*?:\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", r"Total energy:\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"]:
        matches = re.findall(pattern, text)
        if matches:
            return float(matches[-1])
    return None


def cp2k_health(text: str) -> Tuple[bool, str]:
    lower = text.lower()
    if any(token in lower for token in ("abort", "fatal", "segmentation fault", "forrtl", "error termination", "cannot allocate memory", "oom-kill")):
        return False, "cp2k_abort_or_error"
    if any(token in lower for token in ("scf run not converged", "scf not converged", "no scf convergence")):
        return False, "scf_not_converged"
    normal = "program ended at" in lower or "cp2k ended" in lower
    return normal, "" if normal else "cp2k_normal_end_missing"


def read_energies(path: Path) -> List[float]:
    if not path.exists():
        return []
    energies: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        try:
            nums = [float(x) for x in parts]
        except Exception:
            continue
        if len(nums) >= 5:
            energies.append(nums[4])
        elif len(nums) >= 2:
            energies.append(nums[-1])
    return energies


def find_first(job_dir: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        candidates = sorted(job_dir.glob(pattern))
        if candidates:
            return candidates[0]
    return None


def unit_factors(config: Dict[str, Any]):
    units = config.get("units", {})
    if units.get("output_energy", "eV") != "eV" or units.get("output_forces", "eV_per_A") != "eV_per_A":
        return None, None
    try:
        return float(units.get("cp2k_energy_hartree_to_eV", HARTREE_TO_EV)), float(units.get("cp2k_force_hartree_per_bohr_to_eV_per_A", FORCE_AU_TO_EV_A))
    except Exception:
        return None, None


def parse_cp2k_outputs(config: Dict[str, Any]) -> Path:
    ensure_dirs(config)
    rows = []
    energy_factor, force_factor = unit_factors(config)
    for row in read_rows(p(config, "cp2k_jobs_dir") / "cp2k_label_input_manifest.csv"):
        if row.get("status") != "cp2k_input_written_no_cp2k_run":
            continue
        job_dir = Path(row["job_dir"])
        out_dir = p(config, "cp2k_parsed_dir") / row["aimd_structure_id"] / row["label_mode"]
        out_dir.mkdir(parents=True, exist_ok=True)
        project = row.get("cp2k_project") or f"{row['aimd_structure_id']}_{row['label_mode']}"
        summary = {"status": "not_run_no_cp2k_output", "usable_frame_count": 0, "failure_reason": "cp2k.out missing", "detected_cp2k_out": "", "detected_position_file": "", "detected_force_file": "", "detected_energy_file": "", "parser_mode": row.get("cp2k_run_type", "ENERGY_FORCE" if row.get("label_mode") == "sp_force" else "MD"), "n_position_frames": 0, "n_force_frames": 0, "n_energy_frames": 0, "n_frames_written": 0, "frame_count_mismatch": False}
        frames = out_dir / "frames.extxyz"
        cp2k_out = job_dir / "cp2k.out"
        if cp2k_out.exists():
            summary["detected_cp2k_out"] = str(cp2k_out)
            text = cp2k_out.read_text(encoding="utf-8", errors="ignore")
            normal_end, health_reason = cp2k_health(text)
            if energy_factor is None or force_factor is None:
                summary.update({"status": "failed_units_uncertain", "failure_reason": "unit_conversion_config_missing_or_invalid"})
            elif not normal_end:
                summary.update({"status": "failed_cp2k", "failure_reason": health_reason})
            elif row.get("label_mode") == "short_aimd" or row.get("cp2k_run_type") == "MD":
                pos_file = find_first(job_dir, [f"{project}-pos-1.xyz", f"{project}-pos-*.xyz", "*pos*.xyz", "*POS*.xyz"])
                frc_file = find_first(job_dir, [f"{project}-frc-1.xyz", f"{project}-frc-*.xyz", "*frc*.xyz", "*FRC*.xyz", "*forces*.xyz", "*FORCES*.xyz"])
                ene_file = find_first(job_dir, [f"{project}-1.ener", f"{project}-*.ener", "*.ener", "*.ENER"])
                pos_frames = read_xyz_frames(pos_file) if pos_file else []
                frc_frames = read_xyz_frames(frc_file) if frc_file else []
                energies_h = read_energies(ene_file) if ene_file else []
                matched = min(len(pos_frames), len(frc_frames), len(energies_h))
                summary.update({"detected_position_file": str(pos_file or ""), "detected_force_file": str(frc_file or ""), "detected_energy_file": str(ene_file or ""), "n_position_frames": len(pos_frames), "n_force_frames": len(frc_frames), "n_energy_frames": len(energies_h), "n_frames_written": matched, "frame_count_mismatch": len({len(pos_frames), len(frc_frames), len(energies_h)}) > 1})
                if matched > 0:
                    elems = pos_frames[0][0]
                    _, _, box = read_xyz_like(job_dir / "coords.xyz")
                    positions = [frame[1] for frame in pos_frames[:matched]]
                    forces = [[(x * force_factor, y * force_factor, z * force_factor) for x, y, z in frame[1]] for frame in frc_frames[:matched]]
                    energies = [x * energy_factor for x in energies_h[:matched]]
                    write_extxyz_frames(frames, row, elems, positions, forces, energies, box, normal_end)
                    write_rows(out_dir / "thermo.csv", [{"frame": i, "energy_Hartree": energies_h[i], "energy_eV": energies[i]} for i in range(matched)])
                    summary.update({"status": "parsed_real_cp2k_output", "usable_frame_count": matched, "failure_reason": ""})
                else:
                    summary.update({"status": "failed_cp2k", "failure_reason": "missing_matched_md_position_force_energy_frames"})
            else:
                elems, coords, box = read_xyz_like(job_dir / "coords.xyz")
                energy_h = parse_energy_hartree(text)
                force_file = find_first(job_dir, [f"{project}-frc-1.xyz", f"{project}-frc-*.xyz", "forces.xyz", "*frc*.xyz", "*FRC*.xyz", "*forces*.xyz", "*FORCES*.xyz"])
                force_frames = read_xyz_frames(force_file) if force_file else []
                summary.update({"detected_position_file": str(job_dir / "coords.xyz"), "detected_force_file": str(force_file or ""), "detected_energy_file": str(cp2k_out), "n_position_frames": 1, "n_force_frames": len(force_frames), "n_energy_frames": 1 if energy_h is not None else 0})
                if energy_h is not None and force_frames:
                    force_values = force_frames[-1][1]
                    if len(force_values) == len(elems):
                        forces = [[(x * force_factor, y * force_factor, z * force_factor) for x, y, z in force_values]]
                        energy_eV = energy_h * energy_factor
                        write_extxyz_frames(frames, row, elems, [coords], forces, [energy_eV], box, normal_end)
                        write_rows(out_dir / "thermo.csv", [{"frame": 0, "energy_Hartree": energy_h, "energy_eV": energy_eV}])
                        summary.update({"status": "parsed_real_cp2k_output", "usable_frame_count": 1, "failure_reason": "", "n_frames_written": 1})
                    else:
                        summary.update({"status": "failed_cp2k", "failure_reason": "force_atom_count_mismatch"})
                else:
                    summary.update({"status": "failed_cp2k", "failure_reason": "missing_parseable_energy_or_forces"})
        (out_dir / "parse_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
        rows.append(
            {
                "aimd_structure_id": row["aimd_structure_id"],
                "family": row.get("family", ""),
                "crop_family": row.get("crop_family", row.get("family", "")),
                "patch_id": row.get("patch_id", ""),
                "label_mode": row["label_mode"],
                "cp2k_run_type": row.get("cp2k_run_type", ""),
                "source_stage": row.get("source_stage", ""),
                "polymer_architecture": row.get("polymer_architecture", ""),
                **summary,
                "frames_extxyz_path": str(frames) if frames.exists() else "",
                "parse_summary_path": str(out_dir / "parse_summary.yaml"),
            }
        )
    if not rows:
        rows.append({"aimd_structure_id": "none", "label_mode": "none", "status": "not_run_no_cp2k_output", "usable_frame_count": 0, "failure_reason": "no cp2k input jobs"})
    return write_rows(p(config, "cp2k_parsed_dir") / "cp2k_parsed_manifest.csv", rows)
