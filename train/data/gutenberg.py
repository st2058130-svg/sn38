"""Phase 1.4 — Project Gutenberg (public domain → safe for 2012 cutoff).

Handles sedthh/gutenberg_english schema: TEXT / SOURCE / METADATA (JSON string).

  python -m train.data.gutenberg --full-scale --finalize
  python -m train.data.gutenberg --max-docs 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from train.config import FILTERED_DIR, MAX_DOC_CHARS, MIN_DOC_CHARS, RAW_DIR
from train.data.clean import clean_document, clean_text
from train.data.token_budget import TARGET_TOKENS, est_tokens, hit_target
from train.filter.pipeline import decide, filter_jsonl

logger = logging.getLogger(__name__)

# sedthh works with modern datasets; manu often needs a fallback
_CANDIDATES = [
    os.environ.get("GUTENBERG_DATASET", "sedthh/gutenberg_english"),
    "manu/project_gutenberg",
]
SOURCE = "books"
# Chunk oversized books so they pass MAX_DOC_CHARS instead of all being skipped
_CHUNK_CHARS = min(MAX_DOC_CHARS - 1000, 400_000)


def _load_gutenberg():
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("pip install datasets") from e

    last_err = None
    for repo in _CANDIDATES:
        if not repo:
            continue
        try:
            logger.info("Loading Gutenberg dataset %s", repo)
            ds = load_dataset(repo, split="train", streaming=True)
            return ds, repo
        except Exception as e:
            last_err = e
            logger.warning("Failed %s: %s", repo, e)
            try:
                ds = load_dataset(repo, name="default", split="train", streaming=True)
                return ds, repo
            except Exception as e2:
                last_err = e2
                logger.warning("Failed %s name=default: %s", repo, e2)
    raise SystemExit(
        f"Could not load any Gutenberg dataset from {_CANDIDATES}. Last error: {last_err}"
    )


def _row_text(row: dict) -> str:
    for key in ("TEXT", "text", "content", "CONTENT"):
        if row.get(key):
            return str(row[key])
    return ""


def _row_meta(row: dict) -> dict:
    raw = row.get("METADATA") or row.get("metadata") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _issued_date(meta: dict) -> str:
    """Gutenberg catalogue 'issued' is often a PG release date (1970s+), not authorship.

    Do NOT use that as publication_date for temporal filtering — it falsely looks
    post-cutoff. Prefer a trusted pre-cutoff placeholder; store issued in metadata.
    """
    return "1910-01-01"


def _chunk_text(text: str, chunk_chars: int = _CHUNK_CHARS) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= MAX_DOC_CHARS:
        return [text] if len(text) >= MIN_DOC_CHARS else []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        if end < n:
            # break on paragraph if possible
            cut = text.rfind("\n\n", start + chunk_chars // 2, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if len(piece) >= MIN_DOC_CHARS:
            chunks.append(piece)
        if end >= n:
            break
        start = end
    return chunks


def stream_gutenberg(
    out_path: Path,
    *,
    max_docs: int | None = None,
    target_tokens: int | None = None,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds, repo = _load_gutenberg()

    kept = skipped = 0
    n_tokens = 0
    books_seen = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in ds:
            books_seen += 1
            raw_text = _row_text(row)
            meta = _row_meta(row)
            title = meta.get("title") or row.get("title") or meta.get("text_id") or ""
            chunks = _chunk_text(raw_text)
            if not chunks:
                skipped += 1
                if skipped <= 5:
                    logger.warning(
                        "skip book keys=%s text_len=%d title=%r",
                        list(row.keys()),
                        len(raw_text),
                        str(title)[:80],
                    )
                continue

            book_kept = 0
            for i, chunk in enumerate(chunks):
                doc = {
                    "text": chunk,
                    "title": str(title).replace("\r", " ").strip(),
                    "source": SOURCE,
                    "publication_date": _issued_date(meta),
                    "trusted_pre_cutoff": True,
                    "metadata": {
                        "dataset": repo,
                        "text_id": meta.get("text_id"),
                        "authors": meta.get("authors"),
                        "issued_catalog": meta.get("issued"),
                        "chunk": i,
                        "n_chunks": len(chunks),
                    },
                }
                cleaned = clean_document(doc)
                if cleaned is None:
                    continue
                decision = decide(cleaned, source=SOURCE)
                if not decision.keep:
                    continue
                out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                kept += 1
                book_kept += 1
                n_tokens += est_tokens(cleaned["text"])
                if max_docs is not None and kept >= max_docs:
                    break
                if hit_target(n_tokens, target_tokens):
                    break

            if book_kept == 0:
                skipped += 1
            if kept % 200 == 0 and kept:
                logger.info(
                    "gutenberg books_seen=%d kept_chunks=%d skipped_books=%d est_tokens=%d",
                    books_seen,
                    kept,
                    skipped,
                    n_tokens,
                )
            if max_docs is not None and kept >= max_docs:
                break
            if hit_target(n_tokens, target_tokens):
                break

    stats = {
        "source": SOURCE,
        "dataset": repo,
        "books_seen": books_seen,
        "kept": kept,
        "skipped": skipped,
        "est_tokens": n_tokens,
        "target_tokens": target_tokens,
        "out": str(out_path),
    }
    logger.info("%s", stats)
    if kept == 0:
        logger.error(
            "kept=0 — schema mismatch or all filtered. Expected TEXT/METADATA columns "
            "on sedthh/gutenberg_english."
        )
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RAW_DIR / "books" / "gutenberg.jsonl"))
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--full-scale", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    target = TARGET_TOKENS["books"] if args.full_scale else args.target_tokens
    stats = stream_gutenberg(Path(args.out), max_docs=args.max_docs, target_tokens=target)
    if args.finalize:
        stats["filter"] = filter_jsonl(Path(args.out), FILTERED_DIR / "books.jsonl", source=SOURCE)
    print(json.dumps(stats, indent=2))
    if stats["kept"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
