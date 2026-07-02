from pepp_initial_builder.core import build_python_topology, write_packmol_inputs, matrix_rows, load_config

def test_per_chain_packmol_templates(tmp_path):
    cfg=load_config('configs/initial_builder.yaml'); row=matrix_rows(cfg,'tiny')[1]; topo=build_python_topology(row); inp=write_packmol_inputs(tmp_path,topo,cfg,row)
    assert len(list((tmp_path/'packmol'/'chain_templates').glob('chain_*.pdb'))) == row['n_pe_chains']+row['n_pp_chains']
    assert 'structure' in inp.read_text()
