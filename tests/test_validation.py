from pepp_initial_builder.core import relation_exclusions, make_chain, build_angles, SystemTopology
import random

def test_overlap_exclusions_include_12_and_13():
    atoms,bonds,chain=make_chain('PE',4,1,random.Random(1)); topo=SystemTopology(atoms,bonds,build_angles(bonds),[chain],(20,20,20),'x'); ex=relation_exclusions(topo,False)
    assert tuple(sorted((bonds[0].atom1,bonds[0].atom2))) in ex
    assert tuple(sorted((bonds[0].atom1,bonds[1].atom2))) in ex
