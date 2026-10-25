"""Tests for the streaming weighted mix builder (train/data/mix.py)."""

import json
from collections import Counter

import pytest

from train.data.mix import build_mix, estimate_tokens, plan_mix, split_long


def _write(path, n_docs, chars, prefix, weight=None):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_docs):
            doc = {"text": f"{prefix}{i:05d} " + "x" * (chars - 7)}
            if weight is not None:
                doc["loss_weight"] = weight
            f.write(json.dumps(doc) + "\n")
    return path


def test_plan_water_filling_caps_and_redistributes():
    unique = {"wiki": 100, "books": 1000, "news": 10}
    shares = {"wiki": 0.5, "books": 0.3, "news": 0.2}
    plan = plan_mix(unique, 1000, shares, {"wiki": 3, "books": 3, "news": 2})
    src = plan["sources"]
    # wiki wanted 500 but cap is 300; news wanted 200 but cap is 20; books absorbs the rest
    assert src["wiki"]["planned_tokens"] == 300 and src["wiki"]["capped"]
    assert src["news"]["planned_tokens"] == 20 and src["news"]["capped"]
    assert src["books"]["planned_tokens"] == 680
    assert plan["planned_total_tokens"] == 1000 and plan["short_by_tokens"] == 0


def test_plan_short_when_everything_capped_and_unlisted_once():
    plan = plan_mix({"a": 10, "smoke": 7}, 1000, {"a": 1.0}, {"a": 2})
    assert plan["sources"]["smoke"]["planned_tokens"] == 7  # unlisted → exactly once
    assert plan["sources"]["a"]["planned_tokens"] == 20
    assert plan["short_by_tokens"] == 1000 - 27


def test_plan_without_total_is_one_average_pass():
    plan = plan_mix({"a": 100, "b": 300}, None, {"a": 0.5, "b": 0.5}, {"a": 5, "b": 5})
    assert plan["planned_total_tokens"] == 400
    assert plan["sources"]["a"]["planned_tokens"] == 200 and plan["sources"]["a"]["repeat"] == 2.0


def test_split_long_respects_paragraphs_and_limit():
    text = "\n\n".join(["p" * 40] * 10)
    pieces = split_long(text, 100)
    assert all(len(p) <= 100 for p in pieces)
    assert "".join(pieces).replace("\n", "") == "p" * 400
    assert split_long("short", 100) == ["short"]
    hard = split_long("word " * 100, 60)
    assert all(len(p) <= 60 for p in hard)


def test_estimate_tokens_exact_for_small_file(tmp_path):
    p = _write(tmp_path / "a.jsonl", 10, 400, "a")
    assert estimate_tokens(p) == 10 * 400 // 4


def test_build_mix_interleaves_and_repeats(tmp_path):
    wiki = _write(tmp_path / "wikipedia_2012.jsonl", 50, 400, "w")  # 5,000 tokens
    books = _write(tmp_path / "books.jsonl", 200, 400, "b")  # 20,000 tokens
    out = tmp_path / "mix.jsonl"
    stats = build_mix(
        [(wiki, "wikipedia_2012", 1.0), (books, "books", 1.0)],
        out,
        total_tokens=20_000,
        shares={"wikipedia_2012": 0.5, "books": 0.5},
        max_epochs={"wikipedia_2012": 2, "books": 1},
        buffer_chars=2_000,
    )
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    srcs = [r["source"] for r in rows]
    # wiki: 10k planned = 2 full passes → every wiki doc exactly twice
    wiki_ids = Counter(r["text"][:6] for r in rows if r["source"] == "wikipedia_2012")
    assert set(wiki_ids.values()) == {2} and len(wiki_ids) == 50
    assert abs(stats["share_by_source"]["wikipedia_2012"] - 0.5) < 0.05
    # interleaved: both sources present in every quarter of the file
    q = len(srcs) // 4
    for k in range(4):
        part = srcs[k * q : (k + 1) * q]
        assert 0.25 < part.count("wikipedia_2012") / len(part) < 0.75
    assert (tmp_path / "mix_plan.json").exists()


def test_build_mix_keeps_doc_loss_weight_and_dry_run(tmp_path):
    news = _write(tmp_path / "news_2012.jsonl", 5, 400, "n", weight=1.5)
    out = tmp_path / "mix.jsonl"
    res = build_mix([(news, "news_2012", 1.0)], out, total_tokens=None, dry_run=True)
    assert res["dry_run"] and not out.exists()
    build_mix([(news, "news_2012", 1.0)], out, total_tokens=None)
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    assert rows and all(r["loss_weight"] == 1.5 for r in rows)


def test_duplicate_source_names_rejected(tmp_path):
    a = _write(tmp_path / "a.jsonl", 2, 400, "a")
    b = _write(tmp_path / "b.jsonl", 2, 400, "b")
    with pytest.raises(ValueError):
        build_mix([(a, "books", 1.0), (b, "books", 1.0)], tmp_path / "m.jsonl")
