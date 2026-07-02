from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pepp_initial_builder.pore.cli import build_pore_models_main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "configs/aimd_pore_builder.yaml"])
    build_pore_models_main()
