from __future__ import annotations

AMU_TO_G = 1.66053906660e-24
MASS = {"C": 12.011, "H": 1.008}
ATOM_TYPE_IDS = {"PE_C": 1, "PP_CH2": 2, "PP_CH": 3, "PP_CH3_SIDE": 4, "H": 5}
ATOM_TYPE_MASS = {
    1: ("PE_C", MASS["C"]),
    2: ("PP_CH2", MASS["C"]),
    3: ("PP_CH", MASS["C"]),
    4: ("PP_CH3_SIDE", MASS["C"]),
    5: ("H", MASS["H"]),
}
