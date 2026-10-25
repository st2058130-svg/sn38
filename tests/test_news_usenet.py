"""Offline tests for the 2012 news and Usenet extractors (no network)."""

import mailbox
from datetime import date

from train.config import CUTOFF_YEAR_DOC_WEIGHT, DEFAULT_LOSS_WEIGHT
from train.data import news_2012, usenet

BODY = (
    "The European Central Bank kept interest rates unchanged on Thursday as officials "
    "weighed fresh signs of weakness across the eurozone economy. Analysts said the "
    "decision was widely expected and markets reacted calmly to the announcement. "
) * 2


def test_guardian_keeps_2012_article():
    rec = news_2012.guardian_result_to_record(
        {
            "id": "business/2012/nov/08/ecb-rates",
            "type": "article",
            "webPublicationDate": "2012-11-08T13:45:00Z",
            "webUrl": "https://www.theguardian.com/business/2012/nov/08/ecb-rates",
            "fields": {"headline": "ECB holds rates", "bodyText": BODY},
        }
    )
    assert rec is not None
    assert rec["publication_date"] == "2012-11-08T13:45:00Z"
    assert rec["loss_weight"] == CUTOFF_YEAR_DOC_WEIGHT
    assert rec["text"].startswith("ECB holds rates")
    assert news_2012._finish(rec) is not None


def test_guardian_drops_other_years_and_liveblogs():
    base = {"type": "article", "fields": {"bodyText": BODY}}
    assert news_2012.guardian_result_to_record({**base, "webPublicationDate": "2013-01-01T00:00:01Z"}) is None
    assert news_2012.guardian_result_to_record({**base, "webPublicationDate": "2011-12-31T23:00:00Z"}) is None
    assert (
        news_2012.guardian_result_to_record({**base, "type": "liveblog", "webPublicationDate": "2012-05-01T10:00:00Z"})
        is None
    )


def test_wikinews_category_name():
    assert news_2012.wikinews_category(date(2012, 3, 5)) == "Category:March 5, 2012"


def test_wikinews_uses_cutoff_revision_and_creation_year():
    first = {"timestamp": "2012-06-01T10:00:00Z"}
    rev = {"timestamp": "2012-06-02T10:00:00Z", "revid": 1, "slots": {"main": {"*": BODY}}}
    rec = news_2012.wikinews_to_record("ECB holds rates", first, rev)
    assert rec is not None and rec["publication_date"] == "2012-06-01T10:00:00Z"
    # created after the cutoff year → dropped
    assert news_2012.wikinews_to_record("x", {"timestamp": "2013-02-01T00:00:00Z"}, rev) is None
    # created before 2012 → not a 2012 news article
    assert news_2012.wikinews_to_record("x", {"timestamp": "2011-02-01T00:00:00Z"}, rev) is None


def test_usenet_date_parsing():
    assert usenet.parse_post_date("Tue, 20 Nov 2012 10:00:00 +0100").isoformat() == "2012-11-20T09:00:00+00:00"
    assert usenet.parse_post_date("2012-11-20T10:00:00Z").year == 2012
    assert usenet.parse_post_date("garbage") is None
    assert usenet.parse_post_date("Mon, 1 Jan 1970 00:00:00 +0000") is None


def test_usenet_body_cleaning():
    body = "John Doe wrote:\n> quoted line\nMy actual reply.\n-- \nJohn\nsig line"
    assert usenet.clean_post_body(body) == "My actual reply."
    assert usenet.clean_post_body("begin 644 file.bin\nM9F]O\nend") == ""


def test_usenet_groups_and_weights():
    kw = dict(body=BODY, subject="ECB rates", hierarchies=usenet.DEFAULT_HIERARCHIES)
    r2012 = usenet.post_to_record(date_value="Tue, 20 Nov 2012 10:00:00 +0000", groups="sci.econ", **kw)
    r2005 = usenet.post_to_record(date_value="Tue, 20 Nov 2005 10:00:00 +0000", groups="comp.lang.python", **kw)
    assert r2012["loss_weight"] == CUTOFF_YEAR_DOC_WEIGHT and r2012["year"] == 2012
    assert r2005["loss_weight"] == DEFAULT_LOSS_WEIGHT
    assert usenet.post_to_record(date_value="Wed, 2 Jan 2013 10:00:00 +0000", groups="sci.econ", **kw) is None
    assert usenet.post_to_record(date_value="Tue, 20 Nov 2012 10:00:00 +0000", groups="alt.binaries.x", **kw) is None
    assert usenet.post_to_record(date_value="Tue, 20 Nov 2012 10:00:00 +0000", groups="talk.politics", **kw) is None
    assert (
        usenet.post_to_record(date_value="Tue, 20 Nov 2005 10:00:00 +0000", groups="comp.os", min_year=2012, **{**kw})
        is None
    )


def test_usenet_mbox_end_to_end(tmp_path):
    box_path = tmp_path / "sci.econ.mbox"
    box = mailbox.mbox(str(box_path))
    for mid, d in [("<a@x>", "Tue, 20 Nov 2012 10:00:00 +0000"), ("<a@x>", "Tue, 20 Nov 2012 10:00:00 +0000"), ("<b@x>", "Wed, 2 Jan 2013 10:00:00 +0000")]:
        msg = mailbox.mboxMessage()
        msg["Date"], msg["Newsgroups"], msg["Subject"], msg["Message-ID"] = d, "sci.econ", "ECB rates", mid
        msg.set_payload(BODY)
        box.add(msg)
    box.close()
    out = tmp_path / "usenet.jsonl"
    stats = usenet.stream_usenet(usenet.iter_mbox_posts([tmp_path]), out)
    # one kept, one duplicate Message-ID, one after cutoff
    assert stats["kept"] == 1 and stats["by_year"] == {2012: 1}
