"""Unit tests for the 2012 temporal filter (Phase 2)."""

from train.filter.pipeline import decide
from train.filter.year_scan import has_any_post_cutoff_year, has_hard_year_leak, is_past_event_leak
from train.canary.evaluate import evaluate_generations, load_suite


def test_keep_dated_2012_doc():
    d = decide(
        {
            "text": "Published: March 5, 2012. CERN continued the Higgs search.",
            "url": "https://example.com/2012/03/05/cern",
            "metadata": {"publication_date": "2012-03-05"},
        },
        source="fineweb_edu",
    )
    assert d.keep
    assert d.reason == "ok"


def test_discard_after_cutoff_metadata():
    d = decide(
        {
            "text": "Something.",
            "metadata": {"publication_date": "2013-01-01"},
        }
    )
    assert not d.keep
    assert d.reason in {"date_after_cutoff", "best_date_after_cutoff"}


def test_discard_no_date():
    d = decide({"text": "No dates here.", "metadata": {}}, source="fineweb_edu")
    assert not d.keep
    assert d.reason == "no_date_found"


def test_discard_past_tense_future_year():
    d = decide(
        {
            "text": "In 2014, the company announced a major restructuring of its board.",
            "metadata": {"publication_date": "2011-06-01"},
        }
    )
    assert not d.keep
    assert d.reason == "post_cutoff_past_tense_year"


def test_keep_speculative_future_year():
    assert not is_past_event_leak("By 2020, we expect quantum computers to improve.")
    assert not has_hard_year_leak("By 2020, we expect quantum computers to improve.")
    assert has_any_post_cutoff_year("By 2020, we expect quantum computers to improve.", cutoff_year=2011)
    d = decide(
        {
            "text": "By 2020, we expect quantum computers to improve.",
            "metadata": {"publication_date": "2010-05-01"},
        }
    )
    assert d.keep


def test_hard_drops_bare_year_soft_keeps():
    soft_bare = "See clause ISO 2014 for details."
    assert not is_past_event_leak(soft_bare, cutoff_year=2011)
    assert has_hard_year_leak(soft_bare, cutoff_year=2011)
    assert has_any_post_cutoff_year(soft_bare, cutoff_year=2011)


def test_timestamped_requires_real_date():
    from datetime import datetime, timezone

    from train.data.build_timestamped_mix import has_real_timestamp

    cutoff = datetime(2011, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    ok, _, reason = has_real_timestamp(
        {"text": "x", "publication_date": "2011-06-01", "source": "news_2012"},
        source="news_2012",
        cutoff=cutoff,
    )
    assert ok and reason == "ok"
    bad, _, reason = has_real_timestamp(
        {"text": "x", "trusted_pre_cutoff": True, "source": "usenet", "metadata": {}},
        source="usenet",
        cutoff=cutoff,
    )
    assert not bad and reason == "no_date_found"


def test_trusted_usenet_without_parseable_extra():
    d = decide(
        {
            "text": "Discussion about kernels.",
            "metadata": {"publication_date": "2012-11-20"},
            "trusted_pre_cutoff": True,
            "source": "usenet",
        },
        source="usenet",
    )
    assert d.keep


def test_canary_detects_leak():
    suite = load_suite()
    gens = {item["id"]: "..." for item in suite["items"]}
    gens["us_pres_2016"] = " Donald Trump, who defeated Hillary Clinton."
    report = evaluate_generations(suite, gens)
    assert report["n_leaked"] >= 1
    assert any(x["id"] == "us_pres_2016" for x in report["leaked"])


def test_canary_clean_near_zero():
    suite = load_suite()
    gens = {item["id"]: " unknown at this time." for item in suite["items"]}
    report = evaluate_generations(suite, gens)
    assert report["n_leaked"] == 0
    assert not report["alert"]
