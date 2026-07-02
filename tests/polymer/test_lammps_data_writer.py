from pepp_initial_builder.common.config import load_config
from pepp_initial_builder.common.io import write_lammps_data
from pepp_initial_builder.polymer.chain_builder import build_python_topology
from pepp_initial_builder.polymer.matrix import matrix_rows

def test_lammps_data_full_zero_charges(tmp_path):
    topo=build_python_topology(matrix_rows(load_config('configs/polymer.yaml'),'tiny')[0]); p=tmp_path/'x.data'; write_lammps_data(p,topo); text=p.read_text()
    assert 'Masses' in text and 'Atoms # full' in text and 'Bonds' in text and 'Angles' in text
    assert float(text.split('Atoms # full')[1].splitlines()[2].split()[3]) == 0.0
