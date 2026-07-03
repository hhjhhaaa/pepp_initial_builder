from pathlib import Path

from pepp_initial_builder.mlff_seed.lammps_relax import _write_lammps_input
from pepp_initial_builder.mlff_seed.lammps_relax import write_lammps_relax_array_script


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


def test_lammps_relax_array_activates_project_env(tmp_path):
    structures = tmp_path / "structures"
    structures.mkdir()
    (structures / "full_pore_seed_manifest.csv").write_text("status\navailable\n", encoding="utf-8")
    cfg = {
        "paths": {
            "root": str(tmp_path),
            "jobs_dir": "jobs",
            "full_pore_seed_structures_dir": "structures",
        },
        "tools": {"known_lammps_executable": "/path/to/lmp"},
    }
    script = write_lammps_relax_array_script(cfg, "pilot")
    text = script.read_text(encoding="utf-8")
    collect = (tmp_path / "jobs" / "collect_lammps_relax_manifests.sh").read_text(encoding="utf-8")
    assert "conda activate peppmixure" in text
    assert "conda activate peppmixure" in collect
    assert 'export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"' in text
    assert 'export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"' in collect
    assert "PYTHON_CMD=${PYTHON_CMD:-python}" in text
