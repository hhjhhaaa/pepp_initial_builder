from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pepp_initial_builder.common.cli import config_path, workflow_parser
from pepp_initial_builder.pore.config import load_pore_config
from pepp_initial_builder.export.premlff_library import build_premlff_structure_library


if __name__ == "__main__":
    args = workflow_parser("configs/premlff_porems_structure_library.yaml").parse_args()
    print(build_premlff_structure_library(load_pore_config(config_path(args.config))))
