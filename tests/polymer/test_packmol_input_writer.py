from pepp_initial_builder.common.config import load_config
from pepp_initial_builder.polymer.chain_builder import build_python_topology
from pepp_initial_builder.polymer.matrix import matrix_rows
from pepp_initial_builder.polymer.packmol import write_packmol_inputs

def test_per_chain_packmol_templates(tmp_path):
    cfg=load_config('configs/polymer.yaml'); row=matrix_rows(cfg,'tiny')[1]; topo=build_python_topology(row); inp=write_packmol_inputs(tmp_path,topo,cfg,row)
    assert len(list((tmp_path/'packmol'/'chain_templates').glob('chain_*.pdb'))) == row['n_pe_chains']+row['n_pp_chains']
    assert 'structure' in inp.read_text()
