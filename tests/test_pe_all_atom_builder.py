from pepp_initial_builder.core import make_chain
import random

def test_pe_valence_and_hydrogen_count():
    atoms,bonds,_=make_chain('PE',8,1,random.Random(1))
    assert sum(a.element=='C' for a in atoms)==8
    assert sum(a.element=='H' for a in atoms)==18
    nb={a.atom_id:[] for a in atoms}
    for b in bonds: nb[b.atom1].append(b.atom2); nb[b.atom2].append(b.atom1)
    assert all(len(nb[a.atom_id])==4 for a in atoms if a.element=='C')
