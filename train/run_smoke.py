"""End-to-end smoke: sample data → filter → mix → (optional) tiny train.

  python -m train.run_smoke
  python -m train.run_smoke --train --steps 20
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from train.config import FILTERED_DIR, MIX_DIR, TRAIN_ROOT
from train.data.sources import write_smoke_samples
from train.filter.pipeline import filter_jsonl
from train.data.mix import build_mix
from train.config import DEFAULT_LOSS_WEIGHT

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    raw = write_smoke_samples()
    filtered = FILTERED_DIR / "smoke_filtered.jsonl"
    stats = filter_jsonl(raw, filtered, source="fineweb_edu")
    logger.info("filter stats: %s", stats)

    mix_path = MIX_DIR / "smoke_mix.jsonl"
    mix_stats = build_mix([(filtered, "smoke", DEFAULT_LOSS_WEIGHT)], mix_path)
    logger.info("mix stats: %s", mix_stats)

    # canary self-check without a model
    from train.canary.evaluate import evaluate_generations, load_suite

    suite = load_suite()
    clean = {i["id"]: " unknown." for i in suite["items"]}
    report = evaluate_generations(suite, clean)
    assert not report["alert"], report
    logger.info("Phase 0 canary harness OK (%d items)", report["n_items"])

    print(json.dumps({"filter": stats, "mix": mix_stats, "canary_items": report["n_items"]}, indent=2))

    if args.train:
        cmd = [
            sys.executable, "-m", "train.train_clm",
            "--mix", str(mix_path),
            "--arch", "nanochrono",
            "--init-new",
            "--tokenizer", "gpt2",  # smoke-only; real runs need a 32k tokenizer
            "--steps", str(args.steps),
            "--batch-size", "1",
            "--grad-accum", "1",
        ]
        logger.info("Running: %s", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(TRAIN_ROOT.parent))


if __name__ == "__main__":
    main()
