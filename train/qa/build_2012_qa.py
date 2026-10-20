"""Phase 4 — build 2012-weighted QA pairs from filtered Wikipedia JSONL.

Input records need: {"text": "...", "title": "..." (optional)}
Output: JSONL with loss_weight=5.0 style fields for the mix builder.

This generates simple cloze / continuation pairs — expand with LLM extraction
for the full 500k–1M target in train_plan.txt.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from train.config import QA_DIR, QA_LOSS_WEIGHT

logger = logging.getLogger(__name__)

_SENT = re.compile(r"(?<=[.!?])\s+")


def pairs_from_article(text: str, title: str | None = None) -> list[dict]:
    sents = [s.strip() for s in _SENT.split(text) if len(s.strip()) > 40]
    out = []
    for sent in sents[:20]:
        # continuation-style (matches SN38 completion eval better than chat QA)
        words = sent.split()
        if len(words) < 8:
            continue
        split = max(4, len(words) // 2)
        prompt = " ".join(words[:split])
        completion = " ".join(words[split:])
        out.append(
            {
                "text": prompt + " " + completion,
                "prompt": prompt,
                "completion": completion,
                "year": 2012,
                "weight": 5.0,
                "loss_weight": QA_LOSS_WEIGHT,
                "source": "wikipedia_2012",
                "title": title,
            }
        )
        if title and "2012" in sent:
            out.append(
                {
                    "text": f"In 2012, regarding {title}: {sent}",
                    "year": 2012,
                    "weight": 5.0,
                    "loss_weight": QA_LOSS_WEIGHT,
                    "source": "wikipedia_2012",
                    "title": title,
                }
            )
    return out


def build(in_path: Path, out_path: Path, limit: int | None = None) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if limit is not None and n_out >= limit:
                break
            doc = json.loads(line)
            n_in += 1
            for pair in pairs_from_article(doc.get("text", ""), doc.get("title")):
                fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
                n_out += 1
                if limit is not None and n_out >= limit:
                    break
    stats = {"articles": n_in, "pairs": n_out, "out": str(out_path)}
    logger.info("%s", stats)
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", default=str(QA_DIR / "qa_2012.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(build(Path(args.in_path), Path(args.out), args.limit), indent=2))


if __name__ == "__main__":
    main()
