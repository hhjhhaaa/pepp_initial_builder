from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pepp_initial_builder.export.cli import summarize_outputs_main


if __name__ == "__main__":
    summarize_outputs_main()
