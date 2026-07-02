from pepp_initial_builder.common.constants import AMU_TO_G, MASS
from pepp_initial_builder.common.tables import Atom
from pepp_initial_builder.polymer.chain_builder import box_length_from_atoms

def test_box_formula():
    atoms=[Atom(1,'C','PE_C','PE',1,'PE',1,True,True,False,False,1,0,0,0),Atom(2,'H','H','PE',1,'PE',1,False,False,False,True,1,0,0,0)]
    assert abs(box_length_from_atoms(atoms,0.5)-(((MASS['C']+MASS['H'])*AMU_TO_G/0.5)*1e24)**(1/3)) < 1e-12
