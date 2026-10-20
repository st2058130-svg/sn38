"""Per-shard filter: raw extract → trainable JSONL."""

import json
from pathlib import Path

from train.data.wiki_history_pipeline import filter_raw_to_trainable


def test_filter_raw_to_trainable(tmp_path: Path):
    raw = tmp_path / "raw.jsonl"
    rows = [
        {
            "text": (
                "Published: March 5, 2012. The Higgs boson search continued at CERN with new data. "
                "Physicists analyzed collision events from the Large Hadron Collider throughout the year "
                "and prepared papers summarizing the experimental status for the particle physics community. "
                "This paragraph is long enough to pass the minimum document length filter used in training."
            ),
            "title": "Higgs boson",
            "source": "wikipedia_2012",
            "publication_date": "2012-03-05T00:00:00Z",
            "trusted_pre_cutoff": True,
        },
        {
            "text": (
                "In 2014, the company announced a major restructuring of its board of directors after "
                "several quarters of declining revenue. Executives described the reorganization as "
                "necessary for long-term growth. This text is intentionally long enough to pass length "
                "checks but must be discarded for post-cutoff past-tense leakage."
            ),
            "title": "Leak",
            "source": "wikipedia_2012",
            "publication_date": "2011-06-01T00:00:00Z",
            "trusted_pre_cutoff": True,
        },
        {
            "text": "x",  # too short → quality discard
            "title": "Tiny",
            "source": "wikipedia_2012",
            "publication_date": "2010-01-01T00:00:00Z",
            "trusted_pre_cutoff": True,
        },
    ]
    raw.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out = tmp_path / "filtered.jsonl"
    stats = filter_raw_to_trainable(raw, out)
    assert stats["kept"] == 1
    assert stats["discarded"] == 2
    kept = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert kept[0]["title"] == "Higgs boson"
