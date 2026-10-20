from pathlib import Path

from train.data.wiki_history_pipeline import _merge_shard_jsonls


def test_merge_shard_jsonls(tmp_path: Path):
    parts = tmp_path / "parts"
    parts.mkdir()
    (parts / "a.jsonl").write_text('{"id":1}\n{"id":2}\n', encoding="utf-8")
    (parts / "b.jsonl").write_text('{"id":3}\n', encoding="utf-8")
    out = tmp_path / "out.jsonl"
    n = _merge_shard_jsonls(parts, out)
    assert n == 3
    assert out.read_text(encoding="utf-8").count("\n") == 3
