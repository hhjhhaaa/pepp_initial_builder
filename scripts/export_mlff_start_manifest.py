from pathlib import Path
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pepp_initial_builder.core import load_config
from pepp_initial_builder.core import export_manifest
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/initial_builder.yaml'); a=p.parse_args(); print(export_manifest(load_config(ROOT/a.config)))

