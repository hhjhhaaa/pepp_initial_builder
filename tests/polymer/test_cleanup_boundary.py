from pepp_initial_builder.common.config import load_config
from pepp_initial_builder.polymer.lammps_cleanup import cleanup_inputs_text

def test_cleanup_names_no_production():
    texts=cleanup_inputs_text(load_config('configs/polymer.yaml')); joined='\n'.join(texts.values())
    assert 'prod.lammpstrj' not in joined and 'production.lammpstrj' not in joined
    assert 'cleanup_check.lammpstrj' in joined and 'pair_style soft' in texts['in.01_soft_push.lmp']
