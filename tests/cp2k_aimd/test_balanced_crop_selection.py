import numpy as np

from pepp_initial_builder.cp2k_aimd.local_environment_selector import LocalEnvironment, surface_site_metadata
from pepp_initial_builder.cp2k_aimd.seed_patch_builder import select_balanced_candidates


def _env(family):
    return LocalEnvironment(
        selection_reason=family,
        what_local_environment_it_teaches=family,
        center_atom_index=0,
        center_atom_element="C",
        nearest_wall_distance_A=3.0,
        local_polymer_density=0.01,
        local_PE_fraction=1.0,
        local_PP_fraction=0.0,
        surface_class="silanol_rich",
    )


def test_balanced_selection_prioritizes_family_minimums():
    candidates = [
        ({}, _env("PE_HDPE_CH2_silanol_contact"), [], None, (12.0, 12.0, 12.0)),
        ({}, _env("PE_HDPE_CH2_silanol_contact"), [], None, (12.0, 12.0, 12.0)),
        ({}, _env("silica_only_wall_baseline"), [], None, (12.0, 12.0, 12.0)),
        ({}, _env("PE_PP_mixed_near_wall"), [], None, (12.0, 12.0, 12.0)),
    ]
    config = {"cp2k_crop": {"balanced_family_min_counts": {"PE_PP_mixed_near_wall": 1, "silica_only_wall_baseline": 1}}}
    selected = select_balanced_candidates(candidates, config, 3)
    assert [item[1].selection_reason for item in selected] == [
        "PE_PP_mixed_near_wall",
        "silica_only_wall_baseline",
        "PE_HDPE_CH2_silanol_contact",
    ]


def test_surface_site_metadata_records_silanol_and_siloxane_counts():
    elems = ["C", "O", "H", "O", "Si"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.9, 0.0, 0.0],
            [4.0, 2.0, 0.0],
            [5.6, 2.0, 0.0],
        ]
    )
    meta = surface_site_metadata(elems, coords, 0)
    assert meta["nearest_site_type"] == "silanol_OH"
    assert meta["n_silanol_OH_within_5A"] == 1
    assert meta["n_siloxane_O_within_5A"] == 1
    assert meta["nearest_OH_distance_A"] == 3.0
