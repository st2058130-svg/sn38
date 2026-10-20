import json
from datetime import datetime, timezone
from pathlib import Path

from train.data.refilter_cutoff import refilter_file


def test_refilter_drops_2012_keeps_2011(tmp_path: Path):
    inp = tmp_path / "wiki.jsonl"
    rows = [
        {
            "text": (
                "In early 2011 scientists published results from the collider run. "
                "The paper summarized detector performance and calibration procedures "
                "for the collaboration and was widely discussed in the community of "
                "experimental particle physicists during that conference season."
            ),
            "source": "wikipedia_2012",
            "publication_date": "2011-06-01T00:00:00Z",
            "trusted_pre_cutoff": True,
        },
        {
            "text": (
                "In 2012 the experiment announced a major discovery to the press. "
                "Officials described the result as historic for particle physics and "
                "shared plots from the summer conference with journalists worldwide."
            ),
            "source": "wikipedia_2012",
            "publication_date": "2012-07-04T00:00:00Z",
            "trusted_pre_cutoff": True,
        },
    ]
    inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    cutoff = datetime(2011, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    stats = refilter_file(inp, out, cutoff, source="wikipedia_2012")
    assert stats["kept"] == 1
    assert stats["discarded"] == 1
    kept = json.loads(out.read_text(encoding="utf-8").strip())
    assert "2011" in kept["text"]
