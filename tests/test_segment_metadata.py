from pepp_initial_builder.core import build_python_topology, matrix_rows, load_config

def test_segment_centers_are_backbone_carbons():
    row=matrix_rows(load_config('configs/initial_builder.yaml'),'tiny')[1]; topo=build_python_topology(row)
    assert sum(a.is_segment_center for a in topo.atoms)==row['total_backbone_carbons_actual']
    assert all(a.is_backbone and a.element=='C' for a in topo.atoms if a.is_segment_center)
    assert all(not a.is_segment_center for a in topo.atoms if a.atom_type=='PP_CH3_SIDE' or a.element=='H')
