"""Drop selected sources from an existing train_mix.jsonl (streaming).

Example (drop FineWeb + books for a cleaner unknown test):
  python -m train.data.strip_mix_sources \\
    --in $SN38_DATA_ROOT/mix/train_mix.jsonl \\
    --out $SN38_DATA_ROOT/mix/train_mix_nofineweb_nobooks.jsonl \\
    --drop fineweb_edu books
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    p = argparse.ArgumentParser(description="Remove sources from a mix JSONL")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--drop",
        nargs="+",
        required=True,
        help="Source names to remove (exact match on row['source']), e.g. fineweb_edu books",
    )
    args = p.parse_args()
    drop = set(args.drop)
    if not args.inp.is_file():
        raise SystemExit(f"missing input: {args.inp}")

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
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
