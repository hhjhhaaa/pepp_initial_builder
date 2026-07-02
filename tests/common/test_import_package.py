import importlib


def test_new_architecture_packages_import():
    modules = [
        "pepp_initial_builder.common.config",
        "pepp_initial_builder.polymer.matrix",
        "pepp_initial_builder.pore.porems_discovery",
        "pepp_initial_builder.mlff_seed.pore_polymer_packing",
        "pepp_initial_builder.cp2k_aimd.input_writer",
        "pepp_initial_builder.reuse.lmp_proj_discovery",
        "pepp_initial_builder.export.all_manifests",
    ]
    for name in modules:
        assert importlib.import_module(name)
