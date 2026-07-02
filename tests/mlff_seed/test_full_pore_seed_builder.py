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
