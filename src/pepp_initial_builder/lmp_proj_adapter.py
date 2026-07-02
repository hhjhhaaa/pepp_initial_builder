from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def root(config: Dict[str, Any]) -> Path:
    return Path(config["paths"]["root"])


def lmp_proj_root(config: Dict[str, Any]) -> Path:
    return Path(config["paths"].get("lmp_proj_root", "/home/jinhao/lmp-proj"))


def lmp_proj_code_root(config: Dict[str, Any]) -> Path:
    return lmp_proj_root(config) / "Mesoporous_structure_generation"


def add_lmp_proj_to_path(config: Dict[str, Any]) -> bool:
    code_root = lmp_proj_code_root(config)
    if not code_root.exists():
        return False
    text = str(code_root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return True


def _grep(root_path: Path, pattern: str, limit: int) -> List[str]:
    if not root_path.exists():
        return []
    regex = pattern.replace("\\|", "|")
    out: List[str] = []
    for path in _iter_candidate_files(root_path):
        try:
            for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if re.search(regex, line):
                    out.append(f"{_posix(path)}:{lineno}:{line}")
                    if len(out) >= limit:
                        return out
        except Exception:
            continue
    return out


def _iter_candidate_files(root_path: Path) -> List[Path]:
    suffixes = {".py", ".sh", ".inp", ".slurm", ".sbatch", ".j2", ".yaml", ".yml", ".md"}
    return sorted(
        (
            path
            for path in root_path.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes and ".git" not in path.parts and "__pycache__" not in path.parts
        ),
        key=lambda path: str(path),
    )


def _find_files(root_path: Path) -> List[str]:
    if not root_path.exists():
        return []
    allowed = {".py", ".sh", ".inp", ".slurm", ".sbatch"}
    return [_posix(path) for path in _iter_candidate_files(root_path) if path.suffix.lower() in allowed]


def _posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def discover_lmp_proj_modules(config: Dict[str, Any]) -> Dict[str, Any]:
    project_root = root(config)
    logs = project_root / config["paths"].get("logs_dir", "outputs/logs")
    logs.mkdir(parents=True, exist_ok=True)
    lmp_root = lmp_proj_root(config)
    files = _find_files(lmp_root)
    (logs / "lmp_proj_file_index.txt").write_text("\n".join(files) + ("\n" if files else ""), encoding="utf-8")
    py_files = [Path(p) for p in files if p.endswith(".py")]
    found_cp2k_input_templates = [_posix(p) for p in files if "cp2k" in p.lower() and (p.endswith(".j2") or p.endswith(".inp"))]
    found_slurm_templates = [_posix(p) for p in files if "slurm" in p.lower() or "sbatch" in p.lower()]
    found_cp2k_parsers = [_posix(p) for p in py_files if "cp2k" in str(p).lower() and ("parse" in p.name.lower() or "import" in p.name.lower())]
    found_hpc_status_tools = [_posix(p) for p in py_files if "slurm" in str(p).lower() or "submit" in p.name.lower() or "status" in p.name.lower()]
    found_patch_extractors = [_posix(p) for p in py_files if "patch_extract" in str(p) or "patch_pipeline" in p.name]
    files_to_wrap = []
    files_referenced_for_clean_reimplementation = [
        "/home/jinhao/lmp-proj/Mesoporous_structure_generation/microcal/common/slurm.py",
        "/home/jinhao/lmp-proj/Mesoporous_structure_generation/microcal/cp2k/parse.py",
        "/home/jinhao/lmp-proj/Mesoporous_structure_generation/microcal/aimd/parse.py",
    ]
    files_not_reused_and_reason = [
        {
            "file": "/home/jinhao/lmp-proj/Mesoporous_structure_generation/generator/metal_load/patch_extract/extract.py",
            "reason": "Patch extraction is discovered, but the public extractor requires a Ru/metal ClusterSelection and SurfacePatchContext from the old placement pipeline; PE/PP-silica AIMD local structures do not currently provide that contract. Keep as candidate for a later explicit context adapter.",
        },
        {
            "file": "/home/jinhao/lmp-proj/Mesoporous_structure_generation/microcal/cp2k/prepare.py",
            "reason": "CP2K input writer is GEO_OPT/Ru-catalyst oriented. The current workflow needs ENERGY_FORCE and short NVT AIMD for C/H/O/Si systems, so this module is referenced for conventions but not directly wrapped.",
        },
    ]
    report = {
        "lmp_proj_root": _posix(lmp_root),
        "found": lmp_root.exists(),
        "found_cp2k_input_templates": found_cp2k_input_templates,
        "found_slurm_templates": found_slurm_templates,
        "found_cp2k_parsers": found_cp2k_parsers,
        "found_hpc_status_tools": found_hpc_status_tools,
        "found_patch_extractors": found_patch_extractors,
        "recommended_reuse_strategy": "Review lmp-proj implementations and cleanly reimplement the needed CP2K/HPC/parser behavior inside pepp_initial_builder. Do not copy the old project and do not add runtime wrappers.",
        "files_to_wrap": files_to_wrap,
        "files_referenced_for_clean_reimplementation": files_referenced_for_clean_reimplementation,
        "files_not_reused_and_reason": files_not_reused_and_reason,
        "grep_samples": {
            "CP2K": _grep(lmp_root, "CP2K", 200),
            "cp2k": _grep(lmp_root, "cp2k", 200),
            "SLURM": _grep(lmp_root, "SLURM", 200),
            "sbatch": _grep(lmp_root, "sbatch", 200),
            "GEO_OPT_OR_ENERGY_FORCE_OR_AIMD": _grep(lmp_root, "GEO_OPT\\|ENERGY_FORCE\\|AIMD\\|MOTION\\|FORCE_EVAL", 200),
            "BASIS_OR_GTH_OR_DFTD3": _grep(lmp_root, "BASIS_MOLOPT\\|GTH_POTENTIALS\\|DFTD3", 100),
        },
    }
    json_path = logs / "lmp_proj_reuse_report.json"
    txt_path = logs / "lmp_proj_reuse_report.txt"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    txt_lines = [
        f"lmp_proj_root: {report['lmp_proj_root']}",
        f"found: {report['found']}",
        "",
        "found_cp2k_input_templates:",
        *[f"  - {p}" for p in found_cp2k_input_templates[:50]],
        "found_slurm_templates:",
        *[f"  - {p}" for p in found_slurm_templates[:50]],
        "found_cp2k_parsers:",
        *[f"  - {p}" for p in found_cp2k_parsers],
        "found_hpc_status_tools:",
        *[f"  - {p}" for p in found_hpc_status_tools[:50]],
        "found_patch_extractors:",
        *[f"  - {p}" for p in found_patch_extractors[:50]],
        "",
        f"recommended_reuse_strategy: {report['recommended_reuse_strategy']}",
        "files_to_wrap:",
        *[f"  - {p}" for p in files_to_wrap],
        "files_referenced_for_clean_reimplementation:",
        *[f"  - {p}" for p in files_referenced_for_clean_reimplementation],
        "files_not_reused_and_reason:",
        *[f"  - {item['file']}: {item['reason']}" for item in files_not_reused_and_reason],
    ]
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    return {"txt": str(txt_path), "json": str(json_path), "file_index": str(logs / "lmp_proj_file_index.txt"), **report}
