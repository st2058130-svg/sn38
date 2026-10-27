"""Scan a mix JSONL for post-cutoff year + past-tense leaks (streaming).

Uses train.filter.year_scan.is_past_event_leak. Optionally writes a cleaned mix
that drops leaking rows.

Example (2011 cutoff, sample then full clean):
  python -m train.data.audit_mix_years \\
    --in $SN38_DATA_ROOT/mix/train_mix_nofineweb_nobooks.jsonl \\
    --cutoff-year 2011 --max-docs 100000

  python -m train.data.audit_mix_years \\
    --in $SN38_DATA_ROOT/mix/train_mix_nofineweb_nobooks.jsonl \\
    --cutoff-year 2011 \\
    --out $SN38_DATA_ROOT/mix/train_mix_clean_2011.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.filter.year_scan import find_post_cutoff_years, is_past_event_leak  # noqa: E402

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    p = argparse.ArgumentParser(description="Audit / strip post-cutoff year leaks in a mix")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None, help="If set, write rows that pass the year scan")
    p.add_argument("--cutoff-year", type=int, default=2011)
    p.add_argument("--max-docs", type=int, default=0, help="0 = all docs; >0 = sample first N for a quick audit")
    p.add_argument("--examples", type=int, default=20, help="How many leak snippets to print")
    args = p.parse_args()

    if not args.inp.is_file():
        raise SystemExit(f"missing input: {args.inp}")

    fout = None
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fout = args.out.open("w", encoding="utf-8")

    scanned = leaks = kept = 0
    leak_by_src: Counter[str] = Counter()
    ok_by_src: Counter[str] = Counter()
    examples: list[dict] = []

    try:
        with args.inp.open(encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                scanned += 1
                text = row.get("text") or ""
                src = str(row.get("source") or "")
                bad = is_past_event_leak(text, cutoff_year=args.cutoff_year)
                if bad:
                    leaks += 1
                    leak_by_src[src] += 1
                    if len(examples) < args.examples:
                        years = [y for y, _, _ in find_post_cutoff_years(text, cutoff_year=args.cutoff_year)]
                        examples.append(
                            {
                                "source": src,
                                "years": sorted(set(years))[:12],
                                "snippet": text[:240].replace("\n", " "),
                            }
                        )
                else:
                    kept += 1
                    ok_by_src[src] += 1
                    if fout is not None:
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")

                if scanned % 200_000 == 0:
                    logger.info(
                        "scanned=%d leaks=%d (%.2f%%) kept=%d",
                        scanned, leaks, 100.0 * leaks / max(scanned, 1), kept,
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
        "scanned": scanned,
        "leaks": leaks,
        "leak_rate": round(leaks / max(scanned, 1), 6),
        "kept": kept,
        "leak_by_source": dict(leak_by_src),
        "ok_by_source": dict(ok_by_src),
        "examples": examples,
    }
    rep_path = (args.out or args.inp).with_suffix(".year_audit.json")
    if args.out is None:
        rep_path = args.inp.with_name(args.inp.stem + ".year_audit.json")
    else:
        rep_path = args.out.with_suffix(args.out.suffix + ".year_audit.json")
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "cutoff=%d scanned=%d leak_rate=%.2f%% → report %s",
        args.cutoff_year, scanned, 100.0 * report["leak_rate"], rep_path,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    if examples:
        print("--- leak examples ---")
        for ex in examples:
            print(json.dumps(ex, ensure_ascii=False))


if __name__ == "__main__":
    main()
