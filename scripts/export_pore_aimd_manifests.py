from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepp_initial_builder.pore_workflow import export_pore_aimd_manifests, load_pore_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aimd_pore_builder.yaml")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    print(export_pore_aimd_manifests(load_pore_config(ROOT / args.config)))
