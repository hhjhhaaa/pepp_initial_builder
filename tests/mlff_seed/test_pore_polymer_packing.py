from pepp_initial_builder.mlff_seed.full_pore_seed import _packmol_executable


def test_packmol_executable_missing_is_not_replaced(monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert _packmol_executable({"tools": {}}) is None
