from pepp_initial_builder.core import export_manifest, load_config

def test_manifest_export_has_files():
    c,j=export_manifest(load_config('configs/initial_builder.yaml'))
    assert c.exists() and j.exists()
