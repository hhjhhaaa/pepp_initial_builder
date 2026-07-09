from pathlib import Path

from pepp_initial_builder.mlff_seed.full_pore_seed import _write_packmol_full_pore_input
from pepp_initial_builder.pore.porems_builder import atoms_from_elements


def test_full_pore_packmol_input_uses_cylinder_constraint(tmp_path):
    pore_atoms = atoms_from_elements(["Si", "O"], [[10, 10, 10], [12, 10, 10]], 0)
    template = tmp_path / "pe_chain.pdb"
    template.write_text("END\n", encoding="utf-8")
    inp = _write_packmol_full_pore_input(tmp_path, pore_atoms, [template], (30.0, 30.0, 30.0), 10.0, 3.0, 3.0, 1, {"packing": {"tolerance_A": 2.0, "maxit": 2000}})
    text = Path(inp).read_text(encoding="utf-8")
    assert "fixed_silica_pore.pdb" in text
    assert "inside cylinder 15.000000 15.000000 3.000000 0.0 0.0 1.0 7.000000 24.000000" in text



def test_polymer_compositions_support_ps_and_keep_pe_pp_labels():
    from pepp_initial_builder.mlff_seed.full_pore_seed import _composition_label_from_specs
    from pepp_initial_builder.mlff_seed.full_pore_seed import _composition_specs

    legacy = _composition_specs({"pe_pp_compositions": [[1.0, 0.0]]})
    assert _composition_label_from_specs(legacy[0]) == "PE_HDPE100_PP00"

    modern = _composition_specs({"polymer_compositions": [{"PE": 0.0, "PP": 0.0, "PS": 1.0}]})
    assert modern == [[{"component": "PS", "fraction": 1.0}]]
    assert _composition_label_from_specs(modern[0]) == "PE00_PP00_PS100"



def test_component_chain_counts_gives_exact_ps_doped_chain_numbers():
    from pepp_initial_builder.mlff_seed.full_pore_seed import _component_chain_counts

    cfg = {"full_pore_seed": {"loading_chain_multiplier": {"baseline_4chains": 4}}}
    counts = _component_chain_counts(
        cfg,
        [{"component": "PE", "fraction": 0.75}, {"component": "PS", "fraction": 0.25}],
        "baseline_4chains",
    )
    assert counts == {"PE": 3, "PS": 1}
