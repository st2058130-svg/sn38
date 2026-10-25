from train.data.pes2o import _year_from_row


def test_year_from_flat():
    assert _year_from_row({"year": 2010}) == 2010
    assert _year_from_row({"created": "2011-06-01T00:00:00Z"}) == 2011


def test_year_from_nested_metadata():
    assert _year_from_row({"text": "x", "metadata": {"year": 2009}}) == 2009


def test_year_missing_no_recursion():
    # Empty / missing metadata must return None (not RecursionError)
    assert _year_from_row({}) is None
    assert _year_from_row({"text": "hello", "metadata": {}}) is None
    assert _year_from_row({"text": "hello", "metadata": {"title": "t"}}) is None


def test_year_after_cutoff_still_parsed():
    assert _year_from_row({"year": 2015}) == 2015
