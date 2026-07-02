from pepp_initial_builder.core import build_python_topology, write_lammps_data, matrix_rows, load_config

def test_lammps_data_full_zero_charges(tmp_path):
    topo=build_python_topology(matrix_rows(load_config('configs/initial_builder.yaml'),'tiny')[0]); p=tmp_path/'x.data'; write_lammps_data(p,topo); text=p.read_text()
    assert 'Masses' in text and 'Atoms # full' in text and 'Bonds' in text and 'Angles' in text
    assert float(text.split('Atoms # full')[1].splitlines()[2].split()[3]) == 0.0
