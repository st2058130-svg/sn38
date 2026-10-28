"""Build a ChronoGPT-style mix: only docs with a real parseable timestamp ≤ cutoff.

Unlike the default filter, this does NOT keep ``trusted_pre_cutoff`` rows without a date.
FineWeb is excluded by default (crawl/WARC dates ≠ publication dates).

  # 1) Audit / write timestamped shards + mix (cutoff 2011)
  python -m train.data.build_timestamped_mix \\
    --filtered-dir $SN38_DATA_ROOT/filtered \\
    --cutoff-year 2011 \\
    --year-mode hard \\
    --out-dir $SN38_DATA_ROOT/filtered_timestamped_2011 \\
    --out-mix $SN38_DATA_ROOT/mix/train_mix_timestamped_2011.jsonl

  # dry-run plan only (after shards exist, or with --shards-only false and --dry-run)
  python -m train.data.build_timestamped_mix ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import DEFAULT_LOSS_WEIGHT, MIX_DIR, QA_LOSS_WEIGHT  # noqa: E402
from train.data.mix import add_mix_args, build_mix, parse_kv  # noqa: E402
from train.filter.dates import collect_evidence  # noqa: E402
from train.filter.year_scan import is_year_leak  # noqa: E402

logger = logging.getLogger(__name__)

# Real publication / revision / post timestamps (ChronoGPT-like).
DEFAULT_SOURCES = (
    "wikipedia_2012",
    "news_2012",
    "usenet",
    "pes2o",
    "qa_2012",
)

# Optional; Gutenberg "issued" is often a PG upload date, not authorship.
OPTIONAL_WEAK_DATE_SOURCES = frozenset({"books", "gutenberg"})

# Crawl/WARC dated web — not real publication timestamps.
EXCLUDED_BY_DEFAULT = frozenset({"fineweb_edu", "fineweb"})


def _cutoff_dt(year: int) -> datetime:
    return datetime(int(year), 12, 31, 23, 59, 59, tzinfo=timezone.utc)


def _fmt(dt: datetime | None) -> str | None:
    return dt.isoformat().replace("+00:00", "Z") if dt else None


def has_real_timestamp(
    doc: dict[str, Any],
    *,
    source: str | None,
    cutoff: datetime,
) -> tuple[bool, str | None, str]:
    """Require a parseable date signal ≤ cutoff. No trusted-without-date bypass."""
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
    if any(s.when > cutoff for s in evidence.signals):
        return False, None, "date_after_cutoff"
    if evidence.has_conflict:
        return False, None, "conflicting_date_signals"
    if evidence.best is None:
        return False, None, "no_date_found"
    if evidence.best.when > cutoff:
        return False, None, "best_date_after_cutoff"
    return True, _fmt(evidence.best.when), "ok"


def filter_shard(
    inp: Path,
    out: Path,
    *,
    source: str,
    cutoff: datetime,
    year_mode: str,
    max_docs: int = 0,
) -> dict:
    reasons: Counter[str] = Counter()
    kept = discarded = scanned = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with inp.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            scanned += 1
            ok, best_date, reason = has_real_timestamp(doc, source=source, cutoff=cutoff)
            if not ok:
                discarded += 1
                reasons[reason] += 1
            elif year_mode != "off" and is_year_leak(
                doc.get("text") or "", mode=year_mode, cutoff_year=cutoff.year
            ):
                discarded += 1
                reasons[f"year_leak_{year_mode}"] += 1
            else:
                row = dict(doc)
                row["source"] = source
                row["publication_date"] = best_date or row.get("publication_date")
                row["_timestamped"] = True
                row["_cutoff_year"] = cutoff.year
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                reasons["ok"] += 1
            if scanned % 200_000 == 0:
                logger.info(
                    "%s scanned=%d kept=%d discarded=%d",
                    inp.name, scanned, kept, discarded,
                )
            if max_docs and scanned >= max_docs:
                break
    return {
        "in": str(inp),
        "out": str(out),
        "source": source,
        "scanned": scanned,
        "kept": kept,
        "discarded": discarded,
        "keep_rate": round(kept / max(scanned, 1), 6),
        "reasons": dict(reasons),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    p = argparse.ArgumentParser(description="Timestamped-only shards + training mix")
    p.add_argument("--filtered-dir", type=Path, required=True, help="Input filtered/ or filtered_edu/")
    p.add_argument("--out-dir", type=Path, required=True, help="Where to write timestamped JSONL shards")
    p.add_argument("--out-mix", type=Path, default=None, help="Mix JSONL path (default under mix/)")
    p.add_argument("--cutoff-year", type=int, default=2011)
    p.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        help=f"Stems to include (default: {' '.join(DEFAULT_SOURCES)})",
    )
    p.add_argument(
        "--include-books",
        action="store_true",
        help="Also include books/gutenberg (dates often weak)",
    )
    p.add_argument(
        "--year-mode",
        choices=("off", "soft", "hard", "any"),
        default="hard",
        help="Post-cutoff year text scan on kept docs (default hard)",
    )
    p.add_argument("--max-docs-per-source", type=int, default=0, help="Sample cap per shard (0=all)")
    p.add_argument("--shards-only", action="store_true", help="Only write shards; skip mix")
    add_mix_args(p)
    args = p.parse_args()

    cutoff = _cutoff_dt(args.cutoff_year)
    sources = list(args.sources)
    if args.include_books:
        for s in ("books", "gutenberg"):
            if s not in sources:
                sources.append(s)

    bad = [s for s in sources if s in EXCLUDED_BY_DEFAULT]
    if bad:
        raise SystemExit(f"Refusing FineWeb-like sources for timestamped mix: {bad}")

    if not args.filtered_dir.is_dir():
        raise SystemExit(f"missing filtered dir: {args.filtered_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    shards: list[tuple[Path, str, float]] = []

    for source in sources:
        inp = args.filtered_dir / f"{source}.jsonl"
        if not inp.is_file():
            # qa often lives under qa/
            alt = args.filtered_dir.parent / "qa" / f"{source}.jsonl"
            if alt.is_file():
                inp = alt
            else:
                logger.warning("skip missing source %s (%s)", source, inp)
                continue
        out = args.out_dir / f"{source}.jsonl"
        logger.info("=== timestamp filter %s → %s (cutoff=%d) ===", inp, out, args.cutoff_year)
        rep = filter_shard(
            inp,
            out,
            source=source,
            cutoff=cutoff,
            year_mode=args.year_mode,
            max_docs=args.max_docs_per_source,
        )
        reports.append(rep)
        logger.info(
            "%s keep_rate=%.2f%% kept=%d discarded=%d reasons=%s",
            source, 100 * rep["keep_rate"], rep["kept"], rep["discarded"], rep["reasons"],
        )
        if rep["kept"] > 0:
            weight = QA_LOSS_WEIGHT if "qa" in source else DEFAULT_LOSS_WEIGHT
            shards.append((out, source, weight))

    summary = {
        "cutoff_year": args.cutoff_year,
        "year_mode": args.year_mode,
        "sources": sources,
        "shards": reports,
    }
    summary_path = args.out_dir / "timestamped_filter_report.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("shard report → %s", summary_path)

    if args.shards_only or not shards:
        print(json.dumps(summary, indent=2))
        if not shards:
            raise SystemExit("No timestamped shards kept — nothing to mix")
        return

    out_mix = args.out_mix or (MIX_DIR / f"train_mix_timestamped_{args.cutoff_year}.jsonl")
    # Prefer timestamped-friendly shares if user did not override
    share_override = parse_kv(args.share)
    if not share_override:
        share_override = {
            "wikipedia_2012": 0.45,
            "pes2o": 0.20,
            "usenet": 0.15,
            "news_2012": 0.12,
            "qa_2012": 0.08,
        }
        if args.include_books:
            share_override["books"] = 0.08
            # renormalize roughly by leaving build_mix to renorm over present

    mix_stats = build_mix(
        shards,
        Path(out_mix),
        total_tokens=args.total_tokens,
        shares=share_override,
        max_epochs=parse_kv(args.max_epochs),
        dry_run=args.dry_run,
    )
    summary["mix"] = mix_stats
    summary["out_mix"] = str(out_mix)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "shards"}, indent=2, default=str))
    print("--- per-source keep ---")
    for r in reports:
        print(json.dumps({k: r[k] for k in ("source", "scanned", "kept", "discarded", "keep_rate")}, indent=2))


if __name__ == "__main__":
    main()
