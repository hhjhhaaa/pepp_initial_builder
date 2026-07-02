from pepp_initial_builder.common.config import load_config
from pepp_initial_builder.common.tools import discover_tools

def test_tool_discovery_reports_known_paths():
    r=discover_tools(load_config('configs/polymer.yaml'))
    assert 'root' in r['emc']
    assert 'executable' in r['packmol']
    assert 'executable' in r['lammps']
    assert r['python_modules']['ase'] == 'FOUND'
