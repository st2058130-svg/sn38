"""Phase 1.2 — stream peS2o from HuggingFace, keep publication year <= 2012.

Requires: pip install datasets

  export SN38_DATA_ROOT=/workspace/sn38-data
  export HF_TOKEN=hf_xxx   # if needed
  python -m train.data.pes2o --max-docs 10000          # smoke
  python -m train.data.pes2o --finalize                # full stream (long)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from train.config import CUTOFF_YEAR, FILTERED_DIR, RAW_DIR
from train.data.clean import clean_document
from train.data.token_budget import TARGET_TOKENS, est_tokens, hit_target
from train.filter.pipeline import decide, filter_jsonl

logger = logging.getLogger(__name__)

HF_DATASET = os.environ.get("PES2O_DATASET", "allenai/peS2o")
SOURCE = "pes2o"


def _year_from_row(row: dict) -> int | None:
    for key in ("year", "publication_year", "date", "publication_date"):
        if key not in row or row[key] is None:
            continue
        val = row[key]
        if isinstance(val, int):
            return val
        text = str(val)
        if len(text) >= 4 and text[:4].isdigit():
            y = int(text[:4])
            if 1800 <= y <= 2100:
                return y
    meta = row.get("metadata") or {}
    if isinstance(meta, dict):
        return _year_from_row(meta)
    return None


def _text_from_row(row: dict) -> str:
    for key in ("text", "content", "body", "abstract"):
        if row.get(key):
            return str(row[key])
    return ""


def stream_pes2o(
    out_path: Path,
    *,
    max_docs: int | None = None,
    target_tokens: int | None = None,
    split: str = "train",
) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Install datasets: pip install datasets") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(HF_DATASET, split=split, streaming=True)
    kept = skipped = 0
    n_tokens = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in ds:
            year = _year_from_row(row)
            if year is None or year > CUTOFF_YEAR:
                skipped += 1
                continue
            text = _text_from_row(row)
            doc = {
                "text": text,
                "source": SOURCE,
                "publication_date": f"{year}-12-31",
                "metadata": {"year": year, "id": row.get("id") or row.get("paper_id")},
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
            if kept % 1000 == 0:
                logger.info("pes2o kept=%d skipped=%d est_tokens=%d", kept, skipped, n_tokens)
            if max_docs is not None and kept >= max_docs:
                break
            if hit_target(n_tokens, target_tokens):
                logger.info("pes2o hit target_tokens=%d", target_tokens)
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
    parser = argparse.ArgumentParser(description="peS2o ≤2012 extractor")
    parser.add_argument("--out", type=str, default=str(RAW_DIR / "pes2o" / "pes2o_2012.jsonl"))
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help=f"Stop near this many est. tokens (plan mid={TARGET_TOKENS['pes2o']})",
    )
    parser.add_argument("--full-scale", action="store_true", help=f"Use plan target {TARGET_TOKENS['pes2o']} tokens")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--finalize", action="store_true", help="Also write filtered JSONL")
    args = parser.parse_args()

    target = TARGET_TOKENS["pes2o"] if args.full_scale else args.target_tokens
    stats = stream_pes2o(
        Path(args.out), max_docs=args.max_docs, target_tokens=target, split=args.split
    )
    if args.finalize:
        filtered = FILTERED_DIR / "pes2o.jsonl"
        stats["filter"] = filter_jsonl(Path(args.out), filtered, source=SOURCE)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
