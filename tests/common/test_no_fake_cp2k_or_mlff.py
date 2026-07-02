from pepp_initial_builder.common.safety import assert_no_fake_cp2k_or_mlff


def test_no_fake_cp2k_or_mlff(tmp_path):
    assert assert_no_fake_cp2k_or_mlff(tmp_path) == []
