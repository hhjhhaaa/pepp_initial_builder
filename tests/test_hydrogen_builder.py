from pepp_initial_builder.core import make_chain
import random

def test_hydrogens_attach_to_carbon():
    atoms,bonds,_=make_chain('PP',10,1,random.Random(2)); amap={a.atom_id:a for a in atoms}; nb={a.atom_id:[] for a in atoms}
    for b in bonds: nb[b.atom1].append(b.atom2); nb[b.atom2].append(b.atom1)
    for a in atoms:
        if a.element=='H': assert len(nb[a.atom_id])==1 and amap[nb[a.atom_id][0]].element=='C'
