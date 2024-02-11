from pypoe.cli import _parse_colors


def test_single_parse():
    assert _parse_colors("1R2B") == [1, 0, 2]
