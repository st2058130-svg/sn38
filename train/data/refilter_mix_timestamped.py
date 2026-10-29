"""Refilter an existing mix JSONL → keep only docs with a real timestamp ≤ cutoff.

Reuses ``has_real_timestamp`` (no trusted-without-date bypass). Optional hard/any
year text scan.

Important: older mixes often dropped ``publication_date``. If keep_rate ≈ 0%,
rebuild from ``filtered/`` with ``build_timestamped_mix`` instead — you cannot
recover dates that were never written into the mix.

  # Sample first
  python -m train.data.refilter_mix_timestamped \\
    --in $SN38_DATA_ROOT/mix/train_mix_timestamped_2011.jsonl \\
    --cutoff-year 2011 --year-mode hard --max-docs 200000

  # Full rewrite
  python -m train.data.refilter_mix_timestamped \\
    --in $SN38_DATA_ROOT/mix/train_mix_timestamped_2011.jsonl \\
    --cutoff-year 2011 --year-mode hard \\
    --out $SN38_DATA_ROOT/mix/train_mix_timestamped_2011_strict.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.data.build_timestamped_mix import has_real_timestamp  # noqa: E402
from train.filter.year_scan import find_post_cutoff_years, is_year_leak  # noqa: E402

logger = logging.getLogger(__name__)


def _cutoff_dt(year: int) -> datetime:
    return datetime(int(year), 12, 31, 23, 59, 59, tzinfo=timezone.utc)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    p = argparse.ArgumentParser(description="Keep only timestamped rows in a mix JSONL")
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="If set, write kept rows; else audit only")
    p.add_argument("--cutoff-year", type=int, default=2011)
    p.add_argument(
        "--year-mode",
        choices=("off", "soft", "hard", "any"),
        default="hard",
        help="Also drop post-cutoff year text leaks (default hard)",
    )
    p.add_argument(
        "--require-date-field",
        action="store_true",
        help="Require publication_date or year field on the row (skip content/URL date mining)",
    )
    p.add_argument("--max-docs", type=int, default=0, help="0=all; >0 sample first N")
    p.add_argument("--examples", type=int, default=15)
    args = p.parse_args()

    if not args.inp.is_file():
        raise SystemExit(f"missing input: {args.inp}")

    cutoff = _cutoff_dt(args.cutoff_year)
    fout = None
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fout = args.out.open("w", encoding="utf-8")

    scanned = kept = discarded = 0
    reasons: Counter[str] = Counter()
    by_src_kept: Counter[str] = Counter()
    by_src_drop: Counter[str] = Counter()
    examples: list[dict] = []
    has_pub = has_year = 0

    try:
        with args.inp.open(encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                scanned += 1
                src = str(row.get("source") or "")
                if row.get("publication_date"):
                    has_pub += 1
                if row.get("year") is not None:
                    has_year += 1

                if args.require_date_field:
                    if not row.get("publication_date") and row.get("year") is None:
                        discarded += 1
                        reasons["no_date_field"] += 1
                        by_src_drop[src] += 1
                        ok, best_date, reason = False, None, "no_date_field"
                    else:
                        ok, best_date, reason = has_real_timestamp(row, source=src, cutoff=cutoff)
                else:
                    ok, best_date, reason = has_real_timestamp(row, source=src, cutoff=cutoff)

                if ok and args.year_mode != "off":
                    if is_year_leak(row.get("text") or "", mode=args.year_mode, cutoff_year=args.cutoff_year):
                        ok = False
                        reason = f"year_leak_{args.year_mode}"

                if not ok:
                    discarded += 1
                    reasons[reason] += 1
                    by_src_drop[src] += 1
                    if len(examples) < args.examples:
                        years = [
                            y
                            for y, _, _ in find_post_cutoff_years(
                                row.get("text") or "", cutoff_year=args.cutoff_year
                            )
                        ]
                        examples.append(
                            {
                                "reason": reason,
                                "source": src,
                                "years": sorted(set(years))[:8],
                                "publication_date": row.get("publication_date"),
                                "year": row.get("year"),
                                "snippet": (row.get("text") or "")[:200].replace("\n", " "),
                            }
                        )
                else:
                    kept += 1
                    by_src_kept[src] += 1
                    reasons["ok"] += 1
                    if fout is not None:
                        out_row = dict(row)
                        if best_date and not out_row.get("publication_date"):
                            out_row["publication_date"] = best_date
                        out_row["_timestamped"] = True
                        out_row["_cutoff_year"] = args.cutoff_year
                        fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

                if scanned % 200_000 == 0:
                    logger.info(
                        "scanned=%d kept=%d (%.2f%%) discarded=%d",
                        scanned, kept, 100.0 * kept / max(scanned, 1), discarded,
                    )
                if args.max_docs and scanned >= args.max_docs:
                    break
    finally:
        if fout is not None:
            fout.close()

    report = {
        "in": str(args.inp),
        "out": str(args.out) if args.out else None,
        "cutoff_year": args.cutoff_year,
        "year_mode": args.year_mode,
        "require_date_field": args.require_date_field,
        "scanned": scanned,
        "kept": kept,
        "discarded": discarded,
        "keep_rate": round(kept / max(scanned, 1), 6),
        "rows_with_publication_date": has_pub,
        "rows_with_year": has_year,
        "date_field_rate": round(has_pub / max(scanned, 1), 6),
        "reasons": dict(reasons),
        "kept_by_source": dict(by_src_kept),
        "dropped_by_source": dict(by_src_drop),
        "examples": examples,
    }
    if args.out is None:
        rep_path = args.inp.with_name(args.inp.stem + ".refilter_ts_audit.json")
    else:
        rep_path = args.out.with_suffix(args.out.suffix + ".refilter_ts_report.json")
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "cutoff=%d scanned=%d keep_rate=%.2f%% date_field_rate=%.2f%% → %s",
        args.cutoff_year,
        scanned,
        100.0 * report["keep_rate"],
        100.0 * report["date_field_rate"],
        rep_path,
    )
    if report["date_field_rate"] < 0.01 and report["keep_rate"] < 0.05:
        logger.warning(
            "Almost no dates on mix rows and almost nothing kept. "
            "Rebuild from filtered/ with: python -m train.data.build_timestamped_mix ..."
        )

    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    if examples:
        print("--- drop examples ---")
        for ex in examples[:10]:
            print(json.dumps(ex, ensure_ascii=False))


if __name__ == "__main__":
    main()
