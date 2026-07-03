from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pepp_initial_builder.common.cli import config_path, workflow_parser
from pepp_initial_builder.export.anchor_scaffold import scaffold_mlff_aimd_anchor


if __name__ == "__main__":
    parser = workflow_parser("configs/mlff_aimd_anchor.yaml")
    args = parser.parse_args()
    print(scaffold_mlff_aimd_anchor(config_path(args.config)))
