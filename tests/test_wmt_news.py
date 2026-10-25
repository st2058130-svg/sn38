"""Smoke tests for WMT news year parsing / packing helpers."""

from train.data.wmt_news import _parse_years


def test_parse_years_single(monkeypatch):
    import train.config as cfg
    import train.data.wmt_news as wn

    monkeypatch.setattr(cfg, "CUTOFF_YEAR", 2011)
    monkeypatch.setattr(wn, "CUTOFF_YEAR", 2011)
    assert _parse_years("2011") == [2011]


def test_parse_years_range_clamped(monkeypatch):
    import train.config as cfg
    import train.data.wmt_news as wn

    monkeypatch.setattr(cfg, "CUTOFF_YEAR", 2011)
    monkeypatch.setattr(wn, "CUTOFF_YEAR", 2011)
    assert _parse_years("2008-2011") == [2008, 2009, 2010, 2011]
    assert _parse_years("2010-2015") == [2010, 2011]
