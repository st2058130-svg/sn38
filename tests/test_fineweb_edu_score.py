"""Unit tests for FineWeb-edu int_score + JSONL keep/discard (mocked model)."""

import json
from pathlib import Path

from train.data.fineweb_edu_score import int_edu_score, score_jsonl


def test_int_edu_score_clamp_round():
    assert int_edu_score(1.4) == 1
    assert int_edu_score(1.5) == 2
    assert int_edu_score(2.0) == 2
    assert int_edu_score(-1) == 0
    assert int_edu_score(9) == 5


class _FakeClf:
    batch_size = 8

    def score_texts(self, texts: list[str]) -> list[float]:
        # "good" in text → high edu score
        return [3.2 if "good" in t else 0.4 for t in texts]


def test_score_jsonl_keeps_above_threshold(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    rows = [
        {"text": "good educational paragraph about science and history for students " * 5},
        {"text": "spam buy now click here cheap junk " * 5},
        {
            "text": "already scored low",
            "metadata": {"score": 1.0},
            "edu_int_score": 1,
        },
        {
            "text": "already scored high enough for FineWeb-edu-score-2",
            "metadata": {"score": 2.0},
            "edu_int_score": 2,
        },
    ]
    inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    stats = score_jsonl(inp, out, classifier=_FakeClf(), min_int_score=2, reuse_existing_score=True)
    assert stats["kept"] == 2
    assert stats["discarded"] == 2
    kept = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = " ".join(k["text"] for k in kept)
    assert "good educational" in texts
    assert "already scored high" in texts
