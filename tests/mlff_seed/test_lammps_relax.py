from pathlib import Path

from pepp_initial_builder.mlff_seed.lammps_relax import _write_lammps_input


def test_lammps_relax_input_has_staged_pre_equilibration(tmp_path):
    inp = tmp_path / "in.relax"
    _write_lammps_input(
        inp,
        200000,
        400000,
        200000,
        1600000,
        {
            "initial_temperature_K": 300.0,
            "temperature_K": 523.0,
            "high_temperature_K": 650.0,
            "timestep_fs": 0.5,
            "thermo_every": 500,
            "dump_every": 1000,
        },
    )
    text = Path(inp).read_text(encoding="utf-8")
    assert "minimize" in text
    assert "fix int polymer nvt temp 300.000 523.000" in text
    assert "run 200000" in text
    assert "fix int polymer nvt temp 650.000 650.000" in text
    assert "run 400000" in text
    assert "fix int polymer nvt temp 650.000 523.000" in text
    assert "fix int polymer nvt temp 523.000 523.000" in text
    assert "run 1600000" in text
    assert "write_dump all custom relaxed_snapshot.dump" in text
