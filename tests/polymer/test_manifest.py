from pepp_initial_builder.common.config import load_config
from pepp_initial_builder.polymer.manifest import export_manifest

def test_manifest_export_has_files():
    c,j=export_manifest(load_config('configs/polymer.yaml'))
    assert c.exists() and j.exists()
