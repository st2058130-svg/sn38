"""Phase 1.4 — Project Gutenberg (public domain → safe for 2012 cutoff).

Uses HuggingFace `manu/project_gutenberg` or local text files.

  python -m train.data.gutenberg --full-scale --finalize
  python -m train.data.gutenberg --max-docs 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from train.config import FILTERED_DIR, RAW_DIR
from train.data.clean import clean_document
from train.data.token_budget import TARGET_TOKENS, est_tokens, hit_target
from train.filter.pipeline import decide, filter_jsonl

logger = logging.getLogger(__name__)

HF_DATASET = os.environ.get("GUTENBERG_DATASET", "manu/project_gutenberg")
SOURCE = "books"


def stream_gutenberg(
    out_path: Path,
    *,
    max_docs: int | None = None,
    target_tokens: int | None = None,
) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("pip install datasets") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Dataset layouts vary; try common configs
    try:
        ds = load_dataset(HF_DATASET, split="train", streaming=True)
    except Exception:
        ds = load_dataset(HF_DATASET, name="en", split="train", streaming=True)

    kept = skipped = 0
    n_tokens = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in ds:
            text = row.get("text") or row.get("content") or ""
            title = row.get("title") or row.get("id") or ""
            doc = {
                "text": text,
                "title": str(title),
                "source": SOURCE,
                "publication_date": "1910-01-01",  # PD books; pre-cutoff by definition
                "trusted_pre_cutoff": True,
                "metadata": {"dataset": HF_DATASET, "id": row.get("id")},
            }
            cleaned = clean_document(doc)
            if cleaned is None:
                skipped += 1
                continue
            decision = decide(cleaned, source=SOURCE)
            if not decision.keep:
                skipped += 1
                continue
            out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1
            n_tokens += est_tokens(cleaned["text"])
            if kept % 200 == 0:
                logger.info("gutenberg kept=%d est_tokens=%d", kept, n_tokens)
            if max_docs is not None and kept >= max_docs:
                break
            if hit_target(n_tokens, target_tokens):
                break
    stats = {
        "source": SOURCE,
        "dataset": HF_DATASET,
        "kept": kept,
        "skipped": skipped,
        "est_tokens": n_tokens,
        "target_tokens": target_tokens,
        "out": str(out_path),
    }
    logger.info("%s", stats)
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


if __name__ == "__main__":
    main()
