import pytest

from birdclef_kaggle.row_id import KNOWN_SITES, parse, site_to_index


def test_parse_canonical():
    p = parse("BC2026_Test_0001_S08_20240101_120000_5")
    assert p.file_id == "0001"
    assert p.site == "S08"
    assert p.date == "20240101"
    assert p.time == "120000"
    assert p.end_seconds == 5
    assert p.window_idx == 1
    assert p.soundscape_id == "BC2026_Test_0001_S08_20240101_120000"
    assert p.is_known_site is True


def test_parse_last_window():
    p = parse("BC2026_Test_0001_S08_20240101_120000_60")
    assert p.end_seconds == 60
    assert p.window_idx == 12


def test_parse_ood_site():
    p = parse("BC2026_Test_0001_S99_20240101_120000_30")
    assert p.site == "S99"
    assert p.is_known_site is False


def test_parse_malformed():
    with pytest.raises(ValueError):
        parse("not_a_row_id")
    with pytest.raises(ValueError):
        parse("BC2026_Test_0001_S08_20240101_120000")  # missing end_seconds
    with pytest.raises(ValueError):
        parse("BC2026_Test_0001_X08_20240101_120000_5")  # site must start with S


def test_parse_invalid_end_seconds():
    with pytest.raises(ValueError):
        parse("BC2026_Test_0001_S08_20240101_120000_7")  # not a multiple of 5
    with pytest.raises(ValueError):
        parse("BC2026_Test_0001_S08_20240101_120000_0")  # must be positive


def test_known_sites_range():
    assert KNOWN_SITES == frozenset(f"S{i:02d}" for i in range(1, 24))
    assert len(KNOWN_SITES) == 23


def test_site_to_index_known():
    # Sorted order: S01, S02, ..., S23 → index 0, 1, ..., 22
    assert site_to_index("S01") == 0
    assert site_to_index("S23") == 22


def test_site_to_index_unknown():
    assert site_to_index("S99") is None
    assert site_to_index("S00") is None
