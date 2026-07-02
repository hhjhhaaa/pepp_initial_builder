from pathlib import Path
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pepp_initial_builder.core import load_config
from pepp_initial_builder.core import write_discovery_report
print(write_discovery_report(load_config(ROOT/'configs/initial_builder.yaml'), {'emc_adapter':'v0_discovery_only_packmol_fallback'}))

