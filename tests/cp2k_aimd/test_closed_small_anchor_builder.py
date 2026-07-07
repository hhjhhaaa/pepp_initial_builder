import numpy as np

from pepp_initial_builder.cp2k_aimd import closed_small_anchor_builder
from pepp_initial_builder.cp2k_aimd.closed_small_anchor_builder import _cap_silica, _min_heavy_heavy_distance, _min_oxygen_oxygen_distance


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
    assert _min_oxygen_oxygen_distance(capped_elems, capped_coords) >= 2.05


def test_packmol_surface_input_uses_fixed_silica_and_surface_box(tmp_path):
    silica = (["Si", "O"], np.array([[2.0, 2.0, 1.0], [3.6, 2.0, 1.0]], dtype=float))
    frag = (["C", "H"], np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=float))
    inp = closed_small_anchor_builder._write_surface_packmol_input(
        tmp_path / "packmol",
        silica,
        [("PE", frag)],
        {
            "packmol_tolerance_A": 1.85,
            "packmol_lateral_padding_A": 7.0,
            "packmol_surface_gap_A": 2.1,
            "packmol_surface_layer_thickness_A": 7.5,
            "packmol_maxit": 5000,
        },
        seed=9001,
    )

    text = inp.read_text(encoding="utf-8")
    assert "fixed 0.0 0.0 0.0 0.0 0.0 0.0" in text
    assert "inside box" in text
    assert "packed_surface.xyz" in text
    assert not hasattr(closed_small_anchor_builder, "_lateral_slots")
