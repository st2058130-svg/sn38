"""Drop selected sources from an existing train_mix.jsonl (streaming).

Example (drop news from timestamped mix — keep wiki ± pes2o):
  python -m train.data.strip_mix_sources \\
    --in $SN38_DATA_ROOT/mix/train_mix_timestamped_2011_strict.jsonl \\
    --out $SN38_DATA_ROOT/mix/train_mix_wiki_pes2o_2011.jsonl \\
    --drop news_2012

  # wiki only:
  python -m train.data.strip_mix_sources \\
    --in .../train_mix_timestamped_2011_strict.jsonl \\
    --out .../train_mix_wiki_only_2011.jsonl \\
    --drop news_2012 pes2o

  # see exact source names first:
  python -m train.data.strip_mix_sources --in MIX.jsonl --list-sources
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

logger = logging.getLogger(__name__)


def list_sources(path: Path, max_docs: int = 0) -> Counter[str]:
    counts: Counter[str] = Counter()
    n = 0
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            counts[str(row.get("source") or "")] += 1
            n += 1
            if max_docs and n >= max_docs:
                break
            if n % 500_000 == 0:
                logger.info("listed %d rows...", n)
    return counts


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    p = argparse.ArgumentParser(description="Remove sources from a mix JSONL")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None, help="Required unless --list-sources")
    p.add_argument(
        "--drop",
        nargs="+",
        default=None,
        help="Source names to remove (exact match on row['source']), e.g. news_2012 pes2o",
    )
    p.add_argument(
        "--list-sources",
        action="store_true",
        help="Print source counts and exit (no rewrite)",
    )
    p.add_argument("--max-docs", type=int, default=0, help="With --list-sources: sample first N")
    args = p.parse_args()
    if not args.inp.is_file():
        raise SystemExit(f"missing input: {args.inp}")

    if args.list_sources:
        counts = list_sources(args.inp, max_docs=args.max_docs)
        report = {"in": str(args.inp), "by_source": dict(counts), "total": sum(counts.values())}
        print(json.dumps(report, indent=2))
        return

    if args.out is None or not args.drop:
        raise SystemExit("Pass --out and --drop ..., or use --list-sources")

    drop = set(args.drop)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    by_src: Counter[str] = Counter()
    drop_counts: Counter[str] = Counter()

    with args.inp.open(encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src = str(row.get("source") or "")
            if src in drop:
                dropped += 1
                drop_counts[src] += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            by_src[src] += 1
            if (kept + dropped) % 500_000 == 0:
                logger.info("progress rows=%d kept=%d dropped=%d", kept + dropped, kept, dropped)

    report = {
        "in": str(args.inp),
        "out": str(args.out),
        "drop": sorted(drop),
        "kept": kept,
        "dropped": dropped,
        "dropped_by_source": dict(drop_counts),
        "kept_by_source": dict(by_src),
    }
    rep_path = args.out.with_suffix(args.out.suffix + ".strip_report.json")
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("kept=%d dropped=%d → %s", kept, dropped, args.out)
    logger.info("report → %s", rep_path)
    if dropped == 0:
        logger.warning(
            "dropped=0 — source name may not match. Run --list-sources "
            "(mix usually uses news_2012, not news_2011)."
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
