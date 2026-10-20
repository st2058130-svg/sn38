"""Phase 1.3 — stream FineWeb-edu, keep score > threshold, then Phase-2 date filter.

Expect heavy discard. Requires: pip install datasets

  export SN38_DATA_ROOT=/workspace/sn38-data
  python -m train.data.fineweb_edu --max-docs 5000
  python -m train.data.fineweb_edu --finalize
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from train.config import FILTERED_DIR, FINEWEB_EDU_THRESHOLD, RAW_DIR
from train.data.clean import clean_document
from train.data.token_budget import TARGET_TOKENS, est_tokens, hit_target
from train.filter.pipeline import decide, filter_jsonl

logger = logging.getLogger(__name__)

HF_DATASET = os.environ.get("FINEWEB_DATASET", "HuggingFaceFW/fineweb-edu")
SOURCE = "fineweb_edu"


def stream_fineweb(
    out_path: Path,
    *,
    max_docs: int | None = None,
    target_tokens: int | None = None,
    min_score: float = FINEWEB_EDU_THRESHOLD,
    name: str | None = None,
) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Install datasets: pip install datasets") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"path": HF_DATASET, "split": "train", "streaming": True}
    if name:
        kwargs["name"] = name
    ds = load_dataset(**kwargs)

    kept = skipped = 0
    n_tokens = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in ds:
            score = row.get("score") or row.get("int_score") or 0
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = 0.0
            if score_f < min_score:
                skipped += 1
                continue
            doc = {
                "text": row.get("text") or "",
                "url": row.get("url") or row.get("source") or "",
                "source": SOURCE,
                "metadata": {
                    "score": score_f,
                    "dump": row.get("dump"),
                    "date": row.get("date") or row.get("crawl_date"),
                },
                "publication_date": row.get("date") or row.get("crawl_date"),
                "warc_date": row.get("crawl_date"),
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
            if kept % 500 == 0:
                logger.info(
                    "fineweb kept=%d skipped=%d est_tokens=%d (expect high skip)",
                    kept,
                    skipped,
                    n_tokens,
                )
            if max_docs is not None and kept >= max_docs:
                break
            if hit_target(n_tokens, target_tokens):
                logger.info("fineweb hit target_tokens=%d", target_tokens)
                break
    stats = {
        "source": SOURCE,
        "dataset": HF_DATASET,
        "min_score": min_score,
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
    parser = argparse.ArgumentParser(description="FineWeb-edu pre-2013 filtered stream")
    parser.add_argument("--out", type=str, default=str(RAW_DIR / "fineweb_edu" / "fineweb_2012.jsonl"))
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument(
        "--full-scale",
        action="store_true",
        help=f"Use plan target {TARGET_TOKENS['fineweb_edu']} tokens",
    )
    parser.add_argument("--min-score", type=float, default=FINEWEB_EDU_THRESHOLD)
    parser.add_argument("--name", type=str, default=None, help="Optional HF config name")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    target = TARGET_TOKENS["fineweb_edu"] if args.full_scale else args.target_tokens
    stats = stream_fineweb(
        Path(args.out),
        max_docs=args.max_docs,
        target_tokens=target,
        min_score=args.min_score,
        name=args.name,
    )
    if args.finalize:
        filtered = FILTERED_DIR / "fineweb_edu.jsonl"
        stats["filter"] = filter_jsonl(Path(args.out), filtered, source=SOURCE)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
