"""Tests for the cutoff-year paragraph selector (train/qa/build_2012_qa.py)."""

import json

from train.config import QA_LOSS_WEIGHT
from train.qa.build_2012_qa import build, mentions_cutoff_year, paragraphs_from_article

P2012 = (
    "In November 2012, Barack Obama was re-elected President of the United States, defeating "
    "Mitt Romney. Turnout was lower than in the previous election. The campaign focused heavily "
    "on the economy and on health care reform."
)
P_OLD = (
    "The city was founded in the eighteenth century as a trading post. It grew quickly during the "
    "industrial era. Today it is known for its universities and its historic harbour district."
)
LIST = "* 2012 – item one with some words\n* 2012 – item two with more words\n* 2012 – item three" + " x" * 80


def test_year_detection():
    assert mentions_cutoff_year("The event happened in 2012.")
    assert mentions_cutoff_year("On 4 July 2012 CERN announced a new particle.")
    assert not mentions_cutoff_year("The population was 20125 in the census.")
    assert not mentions_cutoff_year("ISBN 978-0-12-2012 and 1.2012 ratio.")


def test_selects_only_cutoff_year_prose_with_title():
    recs = paragraphs_from_article("\n\n".join([P_OLD, P2012, LIST]), "United States presidential election, 2012")
    assert len(recs) == 1
    rec = recs[0]
    assert rec["text"] == f"United States presidential election, 2012\n\n{P2012}"
    assert rec["loss_weight"] == QA_LOSS_WEIGHT
    # no templates or unused fields
    assert "In 2012, regarding" not in rec["text"]
    assert "prompt" not in rec and "completion" not in rec


def test_build_dedupes_and_caps(tmp_path):
    src = tmp_path / "wiki.jsonl"
    rows = [{"title": "A", "text": P2012}, {"title": "B", "text": P2012}, {"title": "C", "text": P2012 + " Extra sentence here."}]
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "qa.jsonl"
    stats = build(src, out)
    assert stats["paragraphs"] == 2 and stats["duplicates"] == 1
    stats = build(src, out, max_tokens=1)
    assert stats["paragraphs"] == 1
