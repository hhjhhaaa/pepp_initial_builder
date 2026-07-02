from pepp_initial_builder.common.config import load_config
from pepp_initial_builder.polymer.matrix import matrix_rows

def cfg(): return load_config('configs/polymer.yaml')
def test_tiny_and_main_matrix_sizes():
    assert len(matrix_rows(cfg(),'tiny')) == 3
    assert len(matrix_rows(cfg(),'matrix')) == 30
def test_system_id_has_no_rho():
    assert all('rho' not in r['system_id'].lower() for r in matrix_rows(cfg(),'matrix'))
def test_main_total_backbone_is_5120():
    assert cfg()['matrix']['total_backbone_carbons'] == 5120
