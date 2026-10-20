"""Tests for Wikipedia-2012 dump extraction (offline fixture)."""

from pathlib import Path

from train.data.wikitext import is_redirect, wikitext_to_text
from train.data.wikipedia_2012 import (
    extract_dump,
    iter_dump_pages,
    write_tiny_fixture_dump,
)


def test_wikitext_basic():
    raw = "'''Bold''' and [[Earth|world]] with {{cite web|url=x}} ref.<ref>x</ref>"
    text = wikitext_to_text(raw)
    assert "Bold" in text
    assert "world" in text
    assert "[[" not in text
    assert "<ref" not in text


def test_redirect_detect():
    assert is_redirect("#REDIRECT [[Foo]]")
    assert not is_redirect("Normal article text here.")


def test_fixture_picks_latest_pre_cutoff(tmp_path: Path):
    dump = write_tiny_fixture_dump(tmp_path / "tiny.xml")
    hits = list(iter_dump_pages(dump))
    assert len(hits) == 1
    assert hits[0].title == "Test Article 2012"
    assert hits[0].revision.rev_id == "11"
    assert hits[0].revision.timestamp.year == 2012


def test_extract_dump_jsonl(tmp_path: Path):
    dump = write_tiny_fixture_dump(tmp_path / "tiny.xml")
    out = tmp_path / "out.jsonl"
    stats = extract_dump(dump, out, min_chars=50)
    assert stats["kept"] == 1
    line = out.read_text(encoding="utf-8").strip()
    assert "Test Article 2012" in line
    assert "2012-06-15" in line
    assert "wikipedia_2012" in line
    assert "2014" not in line  # post-cutoff revision text must not appear
