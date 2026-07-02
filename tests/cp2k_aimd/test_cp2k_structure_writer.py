from pathlib import Path
import pandas as pd

from tests.pore.test_silica_patch_crop import make_manual_pore
from pepp_initial_builder.cp2k_aimd.structure_writer import write_cp2k_structure_inputs
from pepp_initial_builder.cp2k_aimd.seed_patch_builder import build_aimd_local_structures
from pepp_initial_builder.mlff_seed.full_pore_seed import build_full_pore_seed_structures


def test_cp2k_structure_writer_writes_xyz_only(tmp_path):
    cfg = make_manual_pore(tmp_path, ("configs/mlff_seed.yaml", "configs/cp2k_aimd.yaml"))
    build_full_pore_seed_structures(cfg, "tiny")
    build_aimd_local_structures(cfg, "tiny")
    manifest = write_cp2k_structure_inputs(cfg)
    df = pd.read_csv(manifest)
    assert Path(df.iloc[0]["cp2k_xyz_path"]).exists()
    assert df.iloc[0]["status"] == "structure_input_written_no_cp2k_run"
