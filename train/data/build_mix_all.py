"""Finalize: FineWeb-edu score all filtered → QA → training mix.

  export SN38_DATA_ROOT=/workspace/sn38-data
  python -m train.data.build_mix_all
  python -m train.data.build_mix_all --skip-edu   # if filtered_edu/ already built
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from train.config import (
    DEFAULT_LOSS_WEIGHT,
    EDU_DIR,
    FILTERED_DIR,
    MIX_DIR,
    QA_DIR,
    QA_LOSS_WEIGHT,
)
from train.data.mix import build_mix
from train.qa.build_2012_qa import build as build_qa

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(
        description="Edu-classify filtered sources, build QA, write train_mix.jsonl"
    )
    parser.add_argument("--filtered-dir", type=str, default=str(FILTERED_DIR))
    parser.add_argument("--edu-dir", type=str, default=str(EDU_DIR))
    parser.add_argument("--qa-limit", type=int, default=None)
    parser.add_argument("--out", type=str, default=str(MIX_DIR / "train_mix.jsonl"))
    parser.add_argument(
        "--skip-edu",
        action="store_true",
        help="Skip classifier; mix from --edu-dir (or filtered if edu empty)",
    )
    parser.add_argument("--min-int-score", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    filtered = Path(args.filtered_dir)
    edu_dir = Path(args.edu_dir)
    report: dict = {}

    if not args.skip_edu:
        from train.data.fineweb_edu_score import score_all_filtered

        logger.info("=== FineWeb-edu classifier pass over all filtered sources ===")
        report["edu"] = score_all_filtered(
            filtered,
            edu_dir,
            min_int_score=args.min_int_score,
            device=args.device,
        )
    else:
        logger.info("Skipping edu pass (--skip-edu)")

    mix_root = edu_dir if any(edu_dir.glob("*.jsonl")) else filtered
    if mix_root == filtered:
        logger.warning("No filtered_edu JSONL found — mixing from %s", filtered)
    else:
        logger.info("Mixing from edu-filtered dir %s", mix_root)

    wiki = mix_root / "wikipedia_2012.jsonl"
    qa_path = QA_DIR / "qa_2012.jsonl"

    if wiki.exists():
        report["qa"] = build_qa(wiki, qa_path, limit=args.qa_limit)
    else:
        logger.warning("No %s — skipping QA build", wiki)

    shards = []
    for path in sorted(mix_root.glob("*.jsonl")):
        source = path.stem
        weight = QA_LOSS_WEIGHT if "qa" in source else DEFAULT_LOSS_WEIGHT
        shards.append((path, source, weight))
    if qa_path.exists():
        shards.append((qa_path, "qa_2012", QA_LOSS_WEIGHT))

    report["mix"] = build_mix(shards, Path(args.out))
    report["mix_root"] = str(mix_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
