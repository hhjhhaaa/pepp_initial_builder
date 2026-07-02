from pepp_initial_builder.core import discover_tools, load_config

def test_tool_discovery_reports_known_paths():
    r=discover_tools(load_config('configs/initial_builder.yaml'))
    assert r['emc']['root']
    assert r['packmol']['executable']
    assert r['lammps']['executable']
    assert r['python_modules']['ase'] == 'FOUND'
