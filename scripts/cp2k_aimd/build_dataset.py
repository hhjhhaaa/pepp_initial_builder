from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pepp_initial_builder.cp2k_aimd.cli import build_dataset_main


if __name__ == "__main__":
    build_dataset_main()
