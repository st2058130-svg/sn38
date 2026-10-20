"""Load Ti-Ma/wikipedia_2012 (HF) → filtered JSONL for 2012-cutoff training.

Dataset: https://huggingface.co/datasets/Ti-Ma/wikipedia_2012
Already snapshot to 2012-12-31; we drop non-mainspace junk and run Phase 2.

  export SN38_DATA_ROOT=/workspace/sn38-data
  python -m train.data.ti_ma_wikipedia --finalize
  python -m train.data.ti_ma_wikipedia --max-docs 10000   # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

from train.config import CUTOFF, CUTOFF_ISO, FILTERED_DIR, MIN_DOC_CHARS, RAW_DIR
from train.data.clean import clean_document
from train.data.token_budget import TARGET_TOKENS, hit_target
from train.data.wikipedia_2012 import title_to_url
from train.filter.dates import parse_metadata_date
from train.filter.pipeline import decide, filter_jsonl

logger = logging.getLogger(__name__)

HF_DATASET = os.environ.get("TI_MA_WIKI_DATASET", "Ti-Ma/wikipedia_2012")
SOURCE = "wikipedia_2012"

# Drop talk/user/meta namespaces that dominate HF previews
_SKIP_TITLE = re.compile(
    r"^(Talk|User|User talk|Wikipedia|Wikipedia talk|File|File talk|"
    r"MediaWiki|MediaWiki talk|Template|Template talk|Help|Help talk|"
    r"Category|Category talk|Portal|Portal talk|Draft|Draft talk|"
    r"TimedText|Module|Education Program|Gadget|Topic|Special):",
    re.I,
)
# IP / user-looking titles without namespace prefix (common in this dump)
_IP_TITLE = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}$|^Did you know nominations/|^Articles for deletion/",
    re.I,
)


def is_mainspace_title(title: str) -> bool:
    if not title or not title.strip():
        return False
    t = title.strip()
    if _SKIP_TITLE.match(t):
        return False
    if _IP_TITLE.match(t):
        return False
    # Heuristic: pure user talk leftovers
    if t.endswith(".js") or t.endswith(".css"):
        return False
    return True


def stream_ti_ma(
    out_path: Path,
    *,
    max_docs: int | None = None,
    target_tokens: int | None = None,
    min_chars: int = MIN_DOC_CHARS,
) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("pip install datasets") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    kept = skipped_ns = skipped_quality = skipped_date = 0
    total_chars = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in ds:
            title = (row.get("title") or "").strip()
            if not is_mainspace_title(title):
                skipped_ns += 1
                continue
            text = row.get("text") or ""
            ts = row.get("timestamp")
            pub = None
            if ts is not None:
                dt = parse_metadata_date(ts if not hasattr(ts, "isoformat") else ts.isoformat())
                if dt is None:
                    # pandas/pyarrow Timestamp
                    try:
                        pub = str(ts)[:19].replace(" ", "T") + "Z"
                        dt = parse_metadata_date(pub)
                    except Exception:
                        dt = None
                else:
                    pub = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                if dt is not None and dt > CUTOFF:
                    skipped_date += 1
                    continue
            if pub is None:
                pub = "2012-12-31T23:59:59Z"

            doc = {
                "text": text,
                "title": title,
                "source": SOURCE,
                "publication_date": pub,
                "url": title_to_url(title),
                "trusted_pre_cutoff": True,
                "metadata": {
                    "page_id": row.get("page_id"),
                    "rev_id": row.get("revision_id"),
                    "year": row.get("year"),
                    "cutoff": CUTOFF_ISO,
                    "dataset": HF_DATASET,
                },
            }
            cleaned = clean_document(doc)
            if cleaned is None or len(cleaned.get("text") or "") < min_chars:
                skipped_quality += 1
                continue
            decision = decide(cleaned, source=SOURCE)
            if not decision.keep:
                skipped_quality += 1
                continue
            out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1
            total_chars += len(cleaned["text"])
            if kept % 2000 == 0:
                logger.info(
                    "ti-ma kept=%d chars=%.2fGB skip_ns=%d skip_q=%d",
                    kept,
                    total_chars / 1e9,
                    skipped_ns,
                    skipped_quality,
                )
            if max_docs is not None and kept >= max_docs:
                break
            if hit_target(total_chars // 4, target_tokens):
                logger.info("ti-ma hit target_tokens=%d", target_tokens)
                break

    n_tok = total_chars // 4
    stats = {
        "dataset": HF_DATASET,
        "kept": kept,
        "skipped_namespace": skipped_ns,
        "skipped_quality_or_filter": skipped_quality,
        "skipped_after_cutoff": skipped_date,
        "total_chars": total_chars,
        "est_tokens": int(n_tok),
        "target_tokens": target_tokens,
        "out": str(out_path),
        "cutoff": CUTOFF_ISO,
    }
    logger.info("%s", stats)
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description="Ti-Ma wikipedia_2012 → JSONL")
    parser.add_argument("--out", type=str, default=str(RAW_DIR / "wikipedia_2012" / "ti_ma_wiki.jsonl"))
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument(
        "--full-scale",
        action="store_true",
        help=f"Aim for plan mid target {TARGET_TOKENS['wikipedia_2012']} tokens (Ti-Ma may be smaller — take all)",
    )
    parser.add_argument("--min-chars", type=int, default=MIN_DOC_CHARS)
    parser.add_argument("--finalize", action="store_true", help="Also write filtered/wikipedia_2012.jsonl")
    args = parser.parse_args()

    # full-scale: stream entire Ti-Ma (target only stops early if somehow huge)
    target = TARGET_TOKENS["wikipedia_2012"] if args.full_scale else args.target_tokens
    if args.full_scale:
        target = None  # take all Ti-Ma; supplement with Wikimedia later if under 2B
    stats = stream_ti_ma(
        Path(args.out), max_docs=args.max_docs, target_tokens=target, min_chars=args.min_chars
    )
    if args.finalize:
        filtered = FILTERED_DIR / "wikipedia_2012.jsonl"
        stats["filter"] = filter_jsonl(Path(args.out), filtered, source=SOURCE)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
