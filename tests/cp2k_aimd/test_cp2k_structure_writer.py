from pathlib import Path
import pandas as pd

from tests.pore.test_silica_patch_crop import make_manual_pore
from pepp_initial_builder.cp2k_aimd.structure_writer import write_cp2k_structure_inputs
from pepp_initial_builder.cp2k_aimd.seed_patch_builder import build_aimd_local_structures
from tests.cp2k_aimd.helpers import write_relaxed_full_pore_source


def test_cp2k_structure_writer_writes_xyz_only(tmp_path):
    cfg = make_manual_pore(tmp_path, ("configs/mlff_seed.yaml", "configs/cp2k_aimd.yaml"))
    write_relaxed_full_pore_source(cfg, tmp_path)
    build_aimd_local_structures(cfg, "tiny")
    manifest = write_cp2k_structure_inputs(cfg)
    df = pd.read_csv(manifest)
    assert Path(df.iloc[0]["cp2k_xyz_path"]).exists()
    assert df.iloc[0]["status"] == "structure_input_written_no_cp2k_run"
