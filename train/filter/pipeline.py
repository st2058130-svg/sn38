"""Phase 2.2 — discard-on-doubt temporal filter."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from train.config import CUTOFF, CUTOFF_ISO, FILTERED_DIR, RAW_DIR
from train.filter.dates import collect_evidence
from train.filter.year_scan import is_past_event_leak

logger = logging.getLogger(__name__)

HIGH_CONFIDENCE_SOURCES = frozenset({"wikipedia_2012", "usenet", "pes2o", "gutenberg"})


@dataclass
class FilterDecision:
    keep: bool
    reason: str
    best_date: str | None = None
    source_hint: str | None = None


def decide(
    doc: dict[str, Any],
    *,
    source: str | None = None,
) -> FilterDecision:
    """Apply train_plan discard rules to one document dict.

    Expected keys (all optional except text for year scan):
      text, url, metadata (dict), warc_date, publication_date, source
    """
    source = source or doc.get("source")
    text = doc.get("text") or doc.get("content") or ""
    metadata = dict(doc.get("metadata") or {})
    if "publication_date" in doc and "publication_date" not in metadata:
        metadata["publication_date"] = doc["publication_date"]
    if "year" in doc and "year" not in metadata:
        metadata["year"] = doc["year"]

    evidence = collect_evidence(
        metadata=metadata,
        url=doc.get("url"),
        warc_date=doc.get("warc_date"),
        text=text,
    )

    if evidence.any_after_cutoff:
        return FilterDecision(False, "date_after_cutoff", _fmt(evidence.best.when if evidence.best else None), source)

    if evidence.has_conflict:
        return FilterDecision(False, "conflicting_date_signals", None, source)

    if evidence.best is None:
        # Highest-confidence sources may carry an explicit trusted flag
        if doc.get("trusted_pre_cutoff") and source in HIGH_CONFIDENCE_SOURCES:
            return FilterDecision(True, "trusted_source_no_date", None, source)
        return FilterDecision(False, "no_date_found", None, source)

    if evidence.best.when > CUTOFF:
        return FilterDecision(False, "best_date_after_cutoff", _fmt(evidence.best.when), source)

    if text and is_past_event_leak(text):
        return FilterDecision(False, "post_cutoff_past_tense_year", _fmt(evidence.best.when), source)

    return FilterDecision(True, "ok", _fmt(evidence.best.when), source)


def _fmt(dt) -> str | None:
    return dt.isoformat().replace("+00:00", "Z") if dt else None


def filter_records(records: Iterable[dict], source: str | None = None) -> tuple[list[dict], dict]:
    kept, stats = [], {"kept": 0, "discarded": 0, "reasons": {}}
    for doc in records:
        decision = decide(doc, source=source)
        stats["reasons"][decision.reason] = stats["reasons"].get(decision.reason, 0) + 1
        if decision.keep:
            out = dict(doc)
            out["_filter"] = asdict(decision)
            out["_cutoff"] = CUTOFF_ISO
            kept.append(out)
            stats["kept"] += 1
        else:
            stats["discarded"] += 1
    return kept, stats


def filter_jsonl(in_path: Path, out_path: Path, source: str | None = None) -> dict:
    records = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    kept, stats = filter_records(records, source=source)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in kept:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    stats["in_path"] = str(in_path)
    stats["out_path"] = str(out_path)
    logger.info("Filter %s → kept=%d discarded=%d", in_path.name, stats["kept"], stats["discarded"])
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description="Phase 2 discard-on-doubt date filter")
    parser.add_argument("--in", dest="in_path", type=str, required=True, help="Input JSONL")
    parser.add_argument("--out", dest="out_path", type=str, default=None)
    parser.add_argument("--source", type=str, default=None, help="Source tag (wikipedia_2012, usenet, ...)")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path) if args.out_path else FILTERED_DIR / in_path.name
    stats = filter_jsonl(in_path, out_path, source=args.source)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
