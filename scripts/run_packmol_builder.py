from pathlib import Path
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pepp_initial_builder.core import load_config
from pepp_initial_builder.core import build_systems
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/initial_builder.yaml'); p.add_argument('--tiny',action='store_true'); p.add_argument('--max-systems',type=int); a=p.parse_args(); [print(x) for x in build_systems(load_config(ROOT/a.config),a.tiny,False,a.max_systems)]

