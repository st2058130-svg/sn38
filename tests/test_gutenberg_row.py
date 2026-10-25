from train.data.gutenberg import _chunk_text, _row_meta, _row_text


def test_uppercase_text_field():
    row = {"TEXT": "Hello world " * 50, "SOURCE": "gutenberg", "METADATA": "{}"}
    assert _row_text(row).startswith("Hello world")


def test_metadata_json_string():
    row = {
        "TEXT": "x",
        "METADATA": '{"title": "Alice", "text_id": 11, "issued": "1979-12-01 00:00:00"}',
    }
    meta = _row_meta(row)
    assert meta["title"] == "Alice"
    assert meta["text_id"] == 11


def test_chunk_long_book():
    # Must exceed MAX_DOC_CHARS (500_000) to force chunking
    text = ("Paragraph about science and history. " * 50 + "\n\n") * 400
    assert len(text) > 500_000
    chunks = _chunk_text(text)
    assert len(chunks) >= 2
    assert all(len(c) >= 200 for c in chunks)
