"""Scan a mix JSONL for post-cutoff year leaks (streaming).

Modes (see train.filter.year_scan):
  soft — past-tense near post-cutoff year (default; keeps bare years)
  hard — any post-cutoff year unless window is purely speculative
  any  — any post-cutoff year token (strictest)

Example (hard filter on current clean mix):
  python -m train.data.audit_mix_years \\
    --in $SN38_DATA_ROOT/mix/train_mix_clean_2011.jsonl \\
    --cutoff-year 2011 --mode hard --max-docs 200000

  python -m train.data.audit_mix_years \\
    --in $SN38_DATA_ROOT/mix/train_mix_clean_2011.jsonl \\
    --cutoff-year 2011 --mode hard \\
    --out $SN38_DATA_ROOT/mix/train_mix_hard_2011.jsonl
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

from train.filter.year_scan import find_post_cutoff_years, is_year_leak  # noqa: E402

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    p = argparse.ArgumentParser(description="Audit / strip post-cutoff year leaks in a mix")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None, help="If set, write rows that pass the year scan")
    p.add_argument("--cutoff-year", type=int, default=2011)
    p.add_argument(
        "--mode",
        choices=("soft", "hard", "any"),
        default="soft",
        help="soft=past-tense only; hard=bare years too (keep speculative); any=drop every post-cutoff year",
    )
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
                bad = is_year_leak(text, mode=args.mode, cutoff_year=args.cutoff_year)
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
                        "mode=%s scanned=%d leaks=%d (%.2f%%) kept=%d",
                        args.mode, scanned, leaks, 100.0 * leaks / max(scanned, 1), kept,
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
        "mode": args.mode,
        "scanned": scanned,
        "leaks": leaks,
        "leak_rate": round(leaks / max(scanned, 1), 6),
        "kept": kept,
        "leak_by_source": dict(leak_by_src),
        "ok_by_source": dict(ok_by_src),
        "examples": examples,
    }
    if args.out is None:
        rep_path = args.inp.with_name(args.inp.stem + f".year_audit_{args.mode}.json")
    else:
        rep_path = args.out.with_suffix(args.out.suffix + f".year_audit_{args.mode}.json")
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "mode=%s cutoff=%d scanned=%d leak_rate=%.2f%% → report %s",
        args.mode, args.cutoff_year, scanned, 100.0 * report["leak_rate"], rep_path,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    if examples:
        print("--- leak examples ---")
        for ex in examples:
            print(json.dumps(ex, ensure_ascii=False))


if __name__ == "__main__":
    main()
