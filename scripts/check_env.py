from pathlib import Path
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pepp_initial_builder.core import load_config
from pepp_initial_builder.core import discover_tools, write_discovery_report
import json
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/initial_builder.yaml'); a=p.parse_args(); c=load_config(ROOT/a.config); print(json.dumps(discover_tools(c),indent=2)); print(f'wrote {write_discovery_report(c)}')

