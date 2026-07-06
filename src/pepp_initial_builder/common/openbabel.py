from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


def obabel_executable(configured: object = None) -> str | None:
    if configured not in {None, ""}:
        path = Path(str(configured)).expanduser()
        if path.exists():
            return str(path)
    return shutil.which("obabel")


def convert_with_obabel(input_path: str | Path, output_path: str | Path, configured: object = None) -> Path:
    exe = obabel_executable(configured)
    if not exe:
        raise RuntimeError("Open Babel executable 'obabel' was not found")
    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    input_format = _obabel_format(src)
    output_format = _obabel_format(dst)
    cmd = [exe, f"-i{input_format}", str(src), f"-o{output_format}", "-O", str(dst)]
    if input_format == "pdb":
        cmd.append("-ab")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(f"Open Babel conversion failed for {src}: {proc.stdout.strip()}")
    return dst


def _obabel_format(path: Path) -> str:
    suffixes = [item.lower() for item in path.suffixes]
    if suffixes[-2:] == [".pdb", ".gz"]:
        return "pdb"
    suffix = suffixes[-1] if suffixes else ""
    formats = {
        ".pdb": "pdb",
        ".xyz": "xyz",
        ".extxyz": "exyz",
    }
    try:
        return formats[suffix]
    except KeyError as exc:
        raise ValueError(f"Unsupported Open Babel format for {path}") from exc


def pdb_coords_via_obabel(path: str | Path, configured: object = None) -> List[List[float]]:
    src = Path(path)
    xyz = src.with_suffix(".obabel.xyz")
    convert_with_obabel(src, xyz, configured)
    _elems, coords = read_xyz_elements_coords(xyz)
    return coords


def read_xyz_elements_coords(path: str | Path) -> Tuple[List[str], List[List[float]]]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    elems: List[str] = []
    coords: List[List[float]] = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            elems.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elems, coords


def pdb_elements_coords_via_obabel(path: str | Path, configured: object = None) -> Tuple[List[str], List[List[float]]]:
    src = Path(path)
    xyz = src.with_suffix(".obabel.xyz")
    convert_with_obabel(src, xyz, configured)
    return read_xyz_elements_coords(xyz)
