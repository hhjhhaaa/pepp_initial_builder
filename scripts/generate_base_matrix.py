from pathlib import Path
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pepp_initial_builder.core import load_config
from pepp_initial_builder.core import write_matrix
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/initial_builder.yaml'); p.add_argument('--tiny',action='store_true'); p.add_argument('--pilot',action='store_true'); a=p.parse_args(); c=load_config(ROOT/a.config); print(write_matrix(c,'tiny' if a.tiny else 'pilot' if a.pilot else 'matrix'))

