"""Phase 5 — build a weighted training mix JSONL from filtered sources."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from train.config import (
    DEFAULT_LOSS_WEIGHT,
    MIX_DIR,
    QA_LOSS_WEIGHT,
    QA_UPSAMPLE,
    SEED,
)

logger = logging.getLogger(__name__)


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_mix(
    shards: list[tuple[Path, str, float]],
    out_path: Path,
    qa_upsample: int = QA_UPSAMPLE,
) -> dict:
    """shards: list of (path, source_name, base_loss_weight)."""
    random.seed(SEED)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for path, source, weight in shards:
            if not path.exists():
                logger.warning("Missing shard %s — skip", path)
                continue
            rows = list(iter_jsonl(path))
            repeats = qa_upsample if source == "qa_2012" else 1
            for _ in range(repeats):
                for doc in rows:
                    text = doc.get("text") or ""
                    if not text:
                        continue
                    rec = {
                        "text": text,
                        "source": source,
                        "loss_weight": float(doc.get("loss_weight", weight)),
                        "year": doc.get("year"),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    counts[source] = counts.get(source, 0) + 1
    stats = {"records": n, "by_source": counts, "out": str(out_path)}
    logger.info("Mix written: %s", stats)
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description="Build weighted training mix")
    parser.add_argument("--filtered-dir", type=str, required=True)
    parser.add_argument("--qa", type=str, default=None, help="Optional 2012 QA JSONL")
    parser.add_argument("--out", type=str, default=str(MIX_DIR / "train_mix.jsonl"))
    args = parser.parse_args()

    filtered = Path(args.filtered_dir)
    shards = []
    for path in sorted(filtered.glob("*.jsonl")):
        source = path.stem.replace("_filtered", "")
        weight = QA_LOSS_WEIGHT if "qa" in source else DEFAULT_LOSS_WEIGHT
        shards.append((path, source, weight))
    if args.qa:
        shards.append((Path(args.qa), "qa_2012", QA_LOSS_WEIGHT))

    stats = build_mix(shards, Path(args.out))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
