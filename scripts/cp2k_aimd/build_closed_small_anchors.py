from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pepp_initial_builder.cp2k_aimd.closed_small_anchor_builder import build_closed_small_anchors
from pepp_initial_builder.cp2k_aimd.cli import _parser
from pepp_initial_builder.cp2k_aimd.config import load_cp2k_config
from pepp_initial_builder.common.cli import config_path, mode_from_args


if __name__ == "__main__":
    args = _parser().parse_args()
    print(build_closed_small_anchors(load_cp2k_config(config_path(args.config)), mode_from_args(args)))
