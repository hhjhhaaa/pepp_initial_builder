from pepp_initial_builder.core import cleanup_inputs_text, load_config

def test_cleanup_names_no_production():
    texts=cleanup_inputs_text(load_config('configs/initial_builder.yaml')); joined='\n'.join(texts.values())
    assert 'prod.lammpstrj' not in joined and 'production.lammpstrj' not in joined
    assert 'cleanup_check.lammpstrj' in joined and 'pair_style soft' in texts['in.01_soft_push.lmp']
