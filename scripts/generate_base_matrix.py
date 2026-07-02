from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepp_initial_builder.polymer.cli import generate_matrix_main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "configs/initial_builder.yaml"])
    generate_matrix_main()
