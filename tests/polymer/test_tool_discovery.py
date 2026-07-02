from pepp_initial_builder.common.config import load_config
from pepp_initial_builder.common.tools import discover_tools

def test_tool_discovery_reports_known_paths():
    r=discover_tools(load_config('configs/polymer.yaml'))
    assert r['emc']['root']
    assert r['packmol']['executable']
    assert r['lammps']['executable']
    assert r['python_modules']['ase'] == 'FOUND'
