from pathlib import Path
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pepp_initial_builder.core import load_config
from pepp_initial_builder.core import run_cleanup
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/initial_builder.yaml'); p.add_argument('--tiny',action='store_true'); p.add_argument('--pilot',action='store_true'); p.add_argument('--max-systems',type=int); a=p.parse_args(); c=load_config(ROOT/a.config); [print(x) for x in run_cleanup(c,a.tiny,a.pilot,a.max_systems)]

