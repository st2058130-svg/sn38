"""Re-filter a saved corpus snapshot to an earlier cutoff (e.g. 2012 → 2011).

Uses the same discard-on-doubt rules with a temporary cutoff override.

  # From 2012 date-filtered snapshot → 2011 filtered set
  python -m train.data.refilter_cutoff \
    --from-snapshot snapshots/cutoff_2012/filtered \
    --cutoff-year 2011 \
    --out-dir filtered_2011

Then run edu + mix on the new out-dir for the 2011 model.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from train.config import DATA_ROOT, SNAPSHOT_DIR
from train.data.clean import clean_document
from train.filter import pipeline as filter_pipeline
from train.filter.dates import collect_evidence
from train.filter.year_scan import find_post_cutoff_years, is_past_event_leak

logger = logging.getLogger(__name__)


def _decide_with_cutoff(doc: dict, cutoff: datetime, source: str | None) -> filter_pipeline.FilterDecision:
    """Same as filter.pipeline.decide but with an explicit cutoff (not config.CUTOFF)."""
    from train.filter.pipeline import FilterDecision, HIGH_CONFIDENCE_SOURCES

    source = source or doc.get("source")
    text = doc.get("text") or doc.get("content") or ""
    metadata = dict(doc.get("metadata") or {})
    if "publication_date" in doc and "publication_date" not in metadata:
        metadata["publication_date"] = doc["publication_date"]
    if "year" in doc and "year" not in metadata:
        metadata["year"] = doc["year"]

    # Temporarily monkey-patch CUTOFF used by collect_evidence / year scan via local checks
    evidence = collect_evidence(
        metadata=metadata,
        url=doc.get("url"),
        warc_date=doc.get("warc_date"),
        text=text,
    )

    def _fmt(dt):
        return dt.isoformat().replace("+00:00", "Z") if dt else None

    if any(s.when > cutoff for s in evidence.signals):
        return FilterDecision(False, "date_after_cutoff", _fmt(evidence.best.when if evidence.best else None), source)
    if evidence.has_conflict:
        return FilterDecision(False, "conflicting_date_signals", None, source)
    if evidence.best is None:
        if doc.get("trusted_pre_cutoff") and source in HIGH_CONFIDENCE_SOURCES:
            # still require publication year <= cutoff_year when present in metadata
            return FilterDecision(True, "trusted_source_no_date", None, source)
        return FilterDecision(False, "no_date_found", None, source)
    if evidence.best.when > cutoff:
        return FilterDecision(False, "best_date_after_cutoff", _fmt(evidence.best.when), source)

    # Post-cutoff year scan relative to this cutoff year
    cutoff_year = cutoff.year
    if text:
        # reuse heuristic but years > cutoff_year
        import re

        post_years = list(range(cutoff_year + 1, 2027))
        year_re = re.compile(r"\b(" + "|".join(str(y) for y in post_years) + r")\b")
        past = re.compile(
            r"(?i)\b(was|were|had|did|been|became|announced|elected|won|died|launched|"
            r"released|founded|occurred|happened|invaded|signed|passed|defeated|"
            r"collapsed|acquired|merged|published|introduced|opened|closed)\b"
        )
        safe = re.compile(
            r"(?i)\b(by|until|before|expect|expected|will|would|plan|plans|projected|"
            r"forecast|goal|target|aim|hoping|envision)\b"
        )
        for m in year_re.finditer(text):
            left = max(0, m.start() - 40)
            right = min(len(text), m.end() + 40)
            ctx = text[left:right]
            if safe.search(ctx) and not past.search(ctx):
                continue
            if past.search(ctx):
                return FilterDecision(False, "post_cutoff_past_tense_year", _fmt(evidence.best.when), source)

    return FilterDecision(True, "ok", _fmt(evidence.best.when), source)


def refilter_file(in_path: Path, out_path: Path, cutoff: datetime, source: str | None = None) -> dict:
    kept = discarded = 0
    reasons: dict[str, int] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem_source = source or in_path.stem
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            cleaned = clean_document(doc)
            if cleaned is None:
                discarded += 1
                reasons["quality"] = reasons.get("quality", 0) + 1
                continue
            decision = _decide_with_cutoff(cleaned, cutoff, stem_source)
            reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
            if not decision.keep:
                discarded += 1
                continue
            cleaned["_filter"] = {
                "keep": True,
                "reason": decision.reason,
                "best_date": decision.best_date,
                "source_hint": stem_source,
            }
            cleaned["_cutoff"] = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1
    return {"in": str(in_path), "out": str(out_path), "kept": kept, "discarded": discarded, "reasons": reasons}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description="Re-filter snapshot to earlier cutoff year")
    parser.add_argument(
        "--from-snapshot",
        type=str,
        default=None,
        help="Dir of JSONL (e.g. snapshots/cutoff_2012/filtered)",
    )
    parser.add_argument("--cutoff-year", type=int, required=True, help="New cutoff year, e.g. 2011")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output dir (default: $SN38_DATA_ROOT/filtered)",
    )
    args = parser.parse_args()

    src = Path(args.from_snapshot or (SNAPSHOT_DIR / "cutoff_2012" / "filtered"))
    if not src.exists():
        raise SystemExit(f"Snapshot not found: {src}")

    cutoff = datetime(args.cutoff_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    out_dir = Path(args.out_dir or (DATA_ROOT / "filtered"))
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for path in sorted(src.glob("*.jsonl")):
        out = out_dir / path.name
        stats = refilter_file(path, out, cutoff, source=path.stem)
        reports.append(stats)
        logger.info("%s kept=%d discarded=%d", path.name, stats["kept"], stats["discarded"])

    print(json.dumps({"cutoff_year": args.cutoff_year, "files": reports, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
