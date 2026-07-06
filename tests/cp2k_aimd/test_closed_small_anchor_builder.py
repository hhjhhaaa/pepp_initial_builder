import numpy as np

from pepp_initial_builder.cp2k_aimd.closed_small_anchor_builder import _cap_silica, _min_heavy_heavy_distance


def test_multi_oh_capping_avoids_close_oxygen_pairs():
    elems = ["Si", "O"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.62, 0.0, 0.0],
        ],
        dtype=float,
    )

    capped_elems, capped_coords, meta = _cap_silica(elems, coords)

    assert meta["undercoordinated_Si_after_capping"] == 0
    assert meta["uncapped_O_after_capping"] == 0
    assert _min_heavy_heavy_distance(capped_elems, capped_coords) >= 1.35
