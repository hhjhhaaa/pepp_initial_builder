from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from pepp_initial_builder.pore.porems_builder import AIMD_ELEMENTS
from pepp_initial_builder.pore.validation import validate_from_manifest


def validate_full_pore_seed_structures(config: Dict[str, Any]) -> Path:
    manifest = f"{config['paths']['full_pore_seed_structures_dir']}/full_pore_seed_manifest.csv"
    return validate_from_manifest(config, manifest, "full_pore_seed_id", "extxyz_path", "full_pore_seed_validation.csv", AIMD_ELEMENTS, True)
