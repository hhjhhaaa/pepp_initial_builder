from pepp_initial_builder.core import make_chain
import random, pytest

def test_pp_side_methyl_count_and_odd_error():
    atoms,_,_=make_chain('PP',8,1,random.Random(1))
    assert sum(a.atom_type=='PP_CH3_SIDE' for a in atoms)==4
    with pytest.raises(ValueError, match='PP chain_length_backbone must be even'):
        make_chain('PP',7,1,random.Random(1))
