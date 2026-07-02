from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepp_initial_builder.cp2k_workflow import load_cp2k_config


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/cp2k_dataset.yaml")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--pilot", action="store_true")
    return p


def load(args):
    return load_cp2k_config(ROOT / args.config)


def mode(args) -> str:
    return "tiny" if args.tiny else "pilot" if args.pilot else "main"
