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
    assert "PYTHON_CMD=${PYTHON_CMD:-/public/home/jinhao.hu/.conda/envs/peppmixure/bin/python}" in text
    assert "PYTHON_CMD=${PYTHON_CMD:-/public/home/jinhao.hu/.conda/envs/peppmixure/bin/python}" in collect


def test_relax_metrics_uses_final_target_hold_window(tmp_path):
    from pepp_initial_builder.mlff_seed.lammps_relax import _relax_metrics

    seed_dir = tmp_path / "seed"
    relax_dir = seed_dir / "lammps_relax"
    relax_dir.mkdir(parents=True)
    relaxed = seed_dir / "relaxed.extxyz"
    relaxed.write_text(
        "2\n"
        "Lattice=\"60 0 0 0 60 0 0 0 60\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n"
        "Si 30 30 30\n"
        "C 33 30 30\n",
        encoding="utf-8",
    )
    (relax_dir / "lammps_relax.log").write_text(
        "Step c_tpoly PotEng KinEng TotEng Press Atoms\n"
        "0 0 -10 0 -10 0 2\n"
        "1000 650 -5 1 -4 0 2\n"
        "2000 650 -4 1 -3 0 2\n"
        "3000 400 -3 1 -2 0 2\n"
        "4663 500 -2 1 -1 0 2\n"
        "5663 520 -1 1 0 0 2\n",
        encoding="utf-8",
    )
    metrics = _relax_metrics(
        {
            "full_pore_seed_matrix": {"pore_diameter_nm": [4.0]},
            "lammps_full_pore_relax": {"temperature_K": 510.0},
            "relax_quality_gate": {"target_temperature_K": 510.0},
        },
        {"full_pore_seed_id": "seed"},
        seed_dir,
        str(relaxed),
        1,
        500,
        1000,
        500,
        1000,
    )
    assert metrics["hold_temperature_mean_K"] == 510.0


def test_dump_to_extxyz_uses_authoritative_atom_role_elements(tmp_path):
    from pepp_initial_builder.mlff_seed.lammps_relax import _authoritative_elements
    from pepp_initial_builder.mlff_seed.lammps_relax import _dump_to_extxyz

    seed = tmp_path / "seed.extxyz"
    seed.write_text(
        "4\n"
        "Lattice=\"20 0 0 0 20 0 0 0 20\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n"
        "Si 0 0 0\n"
        "O 1 0 0\n"
        "Si 2 0 0\n"
        "Si 3 0 0\n",
        encoding="utf-8",
    )
    roles = tmp_path / "atom_roles.csv"
    roles.write_text("atom_id,element,polymer_type,template_index\n3,C,PP,1\n4,H,PP,1\n", encoding="utf-8")
    dump = tmp_path / "relaxed_snapshot.dump"
    dump.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n4\nITEM: BOX BOUNDS pp pp pp\n0 20\n0 20\n0 20\n"
        "ITEM: ATOMS id mol type q x y z\n"
        "1 0 3 0 0 0 0\n"
        "2 0 4 0 1 0 0\n"
        "3 1 3 0 2 0 0\n"
        "4 1 4 0 3 0 0\n",
        encoding="utf-8",
    )
    labels = _authoritative_elements(seed, 2, 4, roles)
    out = tmp_path / "relaxed.extxyz"
    _dump_to_extxyz(dump, out, (20.0, 20.0, 20.0), labels)
    species = [line.split()[0] for line in out.read_text(encoding="utf-8").splitlines()[2:]]
    assert species == ["Si", "O", "C", "H"]


def test_relax_metrics_corrects_polymer_elements_from_roles(tmp_path):
    from pepp_initial_builder.mlff_seed.lammps_relax import _relax_metrics

    seed_dir = tmp_path / "seed"
    relax_dir = seed_dir / "lammps_relax"
    relax_dir.mkdir(parents=True)
    relaxed = seed_dir / "relaxed.extxyz"
    relaxed.write_text(
        "4\n"
        "Lattice=\"20 0 0 0 20 0 0 0 20\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n"
        "Si 0 0 0\n"
        "O 1 0 0\n"
        "Si 3 0 0\n"
        "Si 4 0 0\n",
        encoding="utf-8",
    )
    (seed_dir / "atom_roles.csv").write_text("atom_id,element,polymer_type,template_index\n3,C,PP,1\n4,H,PP,1\n", encoding="utf-8")
    (relax_dir / "lammps_relax.log").write_text(
        "Step c_tpoly PotEng KinEng TotEng Press Atoms\n"
        "0 520 -2 1 -1 0 4\n"
        "1000 522 -1 1 0 0 4\n",
        encoding="utf-8",
    )
    metrics = _relax_metrics(
        {
            "full_pore_seed_matrix": {"pore_diameter_nm": [4.0]},
            "lammps_full_pore_relax": {"temperature_K": 523.0},
            "relax_quality_gate": {"target_temperature_K": 523.0, "min_contact_count_3p5A": 0, "min_contact_count_5p0A": 0},
        },
        {"full_pore_seed_id": "seed"},
        seed_dir,
        str(relaxed),
        2,
        0,
        0,
        0,
        1000,
    )
    assert metrics["min_polymer_silica_distance_A"] == 2.0
    assert metrics["polymer_inside_pore_fraction"] == 1.0



def test_combined_lammps_data_offsets_silica_types_after_polymer_types(tmp_path):
    from pepp_initial_builder.mlff_seed.lammps_relax import _write_combined_lammps_data

    seed = tmp_path / "seed.extxyz"
    seed.write_text(
        "5\n"
        "Lattice=\"20 0 0 0 20 0 0 0 20\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n"
        "Si 0 0 0\n"
        "O 1 0 0\n"
        "C 2 0 0\n"
        "C 3 0 0\n"
        "H 4 0 0\n",
        encoding="utf-8",
    )
    template = tmp_path / "template"
    template.mkdir()
    (template / "polymer.data").write_text(
        "LAMMPS data\n\n"
        "3 atoms\n0 bonds\n0 angles\n0 dihedrals\n0 impropers\n\n"
        "4 atom types\n\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Masses\n\n"
        "1 12.011 # c\n"
        "2 12.011 # c1\n"
        "3 12.011 # cp\n"
        "4 1.008 # hc\n\n"
        "Atoms\n\n"
        "1 1 1 0.0 2 0 0\n"
        "2 1 3 0.0 3 0 0\n"
        "3 1 4 0.0 4 0 0\n\n"
        "Bonds\n\nAngles\n\nDihedrals\n\nImpropers\n\n",
        encoding="utf-8",
    )
    out = tmp_path / "full_pore_relax.data"
    n_silica, n_polymer, silica_type = _write_combined_lammps_data(seed, [template], out)
    text = out.read_text(encoding="utf-8")
    assert n_silica == 2
    assert n_polymer == 3
    assert silica_type == {"Si": 5, "O": 6, "H": 7}
    assert "7 atom types" in text
    assert "5 28.085500 # Si" in text
    assert "6 15.999400 # O" in text
    assert "1 0 5 0.000000" in text
    assert "2 0 6 0.000000" in text



def _section_text(path, section):
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index(section) + 2
    rows = []
    for line in lines[start:]:
        if not line.strip():
            if rows:
                break
            continue
        if not line.split()[0].isdigit():
            if rows:
                break
            continue
        rows.append(line)
    return rows


def test_combined_lammps_data_remaps_mixed_atom_and_topology_labels(tmp_path):
    from pepp_initial_builder.mlff_seed.lammps_relax import _write_combined_lammps_data

    seed = tmp_path / "seed.extxyz"
    seed.write_text(
        "9\n"
        "Lattice=\"30 0 0 0 30 0 0 0 30\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n"
        "Si 0 0 0\n"
        "O 1 0 0\n"
        "C 2 0 0\n"
        "H 3 0 0\n"
        "H 4 0 0\n"
        "C 5 0 0\n"
        "H 6 0 0\n"
        "H 7 0 0\n"
        "H 8 0 0\n",
        encoding="utf-8",
    )
    t1 = tmp_path / "pe"
    t2 = tmp_path / "ps"
    t1.mkdir()
    t2.mkdir()
    (t1 / "polymer.data").write_text(
        "LAMMPS data\n\n"
        "3 atoms\n1 bonds\n1 angles\n0 dihedrals\n0 impropers\n\n"
        "2 atom types\n1 bond types\n1 angle types\n\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Masses\n\n"
        "1 12.011 # c\n"
        "2 1.008 # h\n\n"
        "Atoms\n\n"
        "1 1 1 0.0 0 0 0\n"
        "2 1 2 0.0 1 0 0\n"
        "3 1 2 0.0 2 0 0\n\n"
        "Bonds\n\n"
        "1 1 1 2 # c,h\n\n"
        "Angles\n\n"
        "1 1 2 1 3 # h,c,h\n\n"
        "Dihedrals\n\n"
        "Impropers\n\n",
        encoding="utf-8",
    )
    (t2 / "polymer.data").write_text(
        "LAMMPS data\n\n"
        "4 atoms\n1 bonds\n1 angles\n0 dihedrals\n1 impropers\n\n"
        "2 atom types\n1 bond types\n1 angle types\n1 improper types\n\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Masses\n\n"
        "1 12.011 # cp\n"
        "2 1.008 # hc\n\n"
        "Atoms\n\n"
        "1 1 1 0.0 0 0 0\n"
        "2 1 2 0.0 1 0 0\n"
        "3 1 2 0.0 2 0 0\n"
        "4 1 2 0.0 3 0 0\n\n"
        "Bonds\n\n"
        "1 1 1 2 # cp,hc\n\n"
        "Angles\n\n"
        "1 1 2 1 3 # hc,cp,hc\n\n"
        "Dihedrals\n\n"
        "Impropers\n\n"
        "1 1 2 1 3 4 # hc,cp,hc,hc\n\n",
        encoding="utf-8",
    )
    params = tmp_path / "polymer.params"
    params.write_text(
        "mass 1 1.008 # h\n"
        "mass 2 12.011 # c\n"
        "mass 3 1.008 # hc\n"
        "mass 4 12.011 # cp\n"
        "bond_coeff 1 1.0 1.0 # cp,hc\n"
        "bond_coeff 2 1.0 1.0 # c,h\n"
        "angle_coeff 1 1.0 1.0 # hc,cp,hc\n"
        "angle_coeff 2 1.0 1.0 # h,c,h\n"
        "angle_coeff 99 bb 1.0 1.0 # cross-term-not-data-type\n"
        "improper_coeff 7 0.0 0.0 # cp,hc,hc,hc\n",
        encoding="utf-8",
    )
    out = tmp_path / "full_pore_relax.data"
    n_silica, n_polymer, silica_type = _write_combined_lammps_data(seed, [t1, t2], out, params)
    text = out.read_text(encoding="utf-8")
    assert n_silica == 2
    assert n_polymer == 7
    assert silica_type == {"Si": 5, "O": 6, "H": 7}
    assert "7 atom types" in text
    assert "2 bond types" in text
    assert "2 angle types" in text
    assert "7 improper types" in text
    atom_rows = _section_text(out, "Atoms")
    assert atom_rows[2].split()[2] == "2"
    assert atom_rows[5].split()[2] == "4"
    assert _section_text(out, "Bonds") == ["1 2 3 4", "2 1 6 7"]
    assert _section_text(out, "Angles") == ["1 2 4 3 5", "2 1 7 6 8"]
    assert _section_text(out, "Impropers") == ["1 7 7 6 8 9"]


def test_lammps_relax_input_uses_dynamic_silica_types(tmp_path):
    inp = tmp_path / "in.relax"
    _write_lammps_input(
        inp,
        1,
        1,
        1,
        1,
        {"temperature_K": 523.0},
        {"Si": 5, "O": 6, "H": 7},
    )
    text = inp.read_text(encoding="utf-8")
    assert "mass 5 28.085500" in text
    assert "pair_coeff 6 6 0.054000 3.470000" in text
    assert "group silica type 5 6 7" in text
    assert "group silica type 3 4 5" not in text
