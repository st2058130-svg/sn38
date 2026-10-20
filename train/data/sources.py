"""Phase 1 — data acquisition helpers (download + normalize to JSONL).

Each source writer emits JSONL records:
  {"text": "...", "url": "...", "publication_date": "...", "source": "...", "metadata": {...}}

Heavy downloads are intentionally manual / gated; these functions document the
exact commands and write small smoke samples for pipeline testing.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from train.config import RAW_DIR

logger = logging.getLogger(__name__)


SOURCE_NOTES = {
    "wikipedia_2012": {
        "priority": "highest",
        "howto": (
            "Download English Wikipedia pages-meta-history near 2012-12 from "
            "https://dumps.wikimedia.org/enwiki/ and filter revisions with "
            "timestamp <= 2012-12-31. Or reconstruct via MediaWiki API "
            "prop=revisions&rvstart=2012-12-31T23:59:59Z."
        ),
        "hf": None,
    },
    "pes2o": {
        "priority": "high",
        "howto": "Load peS2o from AllenAI / HuggingFace; keep year <= 2012 only.",
        "hf": "allenai/peS2o",
    },
    "fineweb_edu": {
        "priority": "medium",
        "howto": (
            "Stream HuggingFaceFW/fineweb-edu, keep classifier score > 2.0, "
            "then run Phase 2 date filter (expect heavy discard)."
        ),
        "hf": "HuggingFaceFW/fineweb-edu",
    },
    "books": {
        "priority": "medium",
        "howto": "Project Gutenberg (all PD) + Internet Archive books with publication_date <= 2012.",
        "hf": None,
    },
    "usenet": {
        "priority": "optional",
        "howto": "HuggingFace Usenet Corpus 1980–2013 (gated). Prefer comp.* and sci.*; date <= cutoff.",
        "hf": None,
    },
}


def write_smoke_samples(out_dir: Path | None = None) -> Path:
    """Write tiny dated samples so filter/train can be smoke-tested without full dumps."""
    out_dir = out_dir or (RAW_DIR / "smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "smoke_docs.jsonl"
    samples = [
        {
            "text": "Published: March 5, 2012. The Higgs boson search continued at CERN.",
            "url": "https://example.com/2012/03/05/cern",
            "source": "fineweb_edu",
            "metadata": {"publication_date": "2012-03-05"},
        },
        {
            "text": "In 2014, the company announced a major restructuring.",
            "url": "https://example.com/2011/01/01/old",
            "source": "fineweb_edu",
            "metadata": {"publication_date": "2011-06-01"},
        },
        {
            "text": "Usenet discussion about Python 3 migration.",
            "source": "usenet",
            "metadata": {"publication_date": "2012-11-20T10:00:00Z"},
            "trusted_pre_cutoff": True,
        },
        {
            "text": "By 2020, we expect quantum computers to improve.",
            "url": "https://example.com/2010/05/speculative",
            "source": "books",
            "metadata": {"publication_date": "2010-05-01"},
        },
        {
            "text": "No date here at all, should be discarded.",
            "source": "fineweb_edu",
            "metadata": {},
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote smoke samples → %s", path)
    return path


def print_acquisition_plan():
    print("Phase 1 acquisition plan (see train_plan.txt for full detail)\n")
    for name, info in SOURCE_NOTES.items():
        print(f"## {name} [{info['priority']}]")
        print(f"  {info['howto']}")
        if info["hf"]:
            print(f"  HF: {info['hf']}")
        print(f"  raw dir: {RAW_DIR / name}")
        print()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description="Phase 1 data acquisition helpers")
    parser.add_argument("--plan", action="store_true", help="Print source acquisition plan")
    parser.add_argument("--smoke", action="store_true", help="Write smoke JSONL samples")
    args = parser.parse_args()
    if args.plan:
        print_acquisition_plan()
    if args.smoke:
        write_smoke_samples()
    if not args.plan and not args.smoke:
        parser.print_help()


if __name__ == "__main__":
    main()
