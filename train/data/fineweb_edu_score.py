"""Post-datagen FineWeb-edu classifier pass over ALL filtered sources.

After Phase-1 date/quality JSONL exists under filtered/, score every document with
HuggingFaceFW/fineweb-edu-classifier and keep int_score >= threshold (default 2).

  export SN38_DATA_ROOT=/workspace/sn38-data
  python -m train.data.fineweb_edu_score --all
  python -m train.data.fineweb_edu_score --in filtered/wikipedia_2012.jsonl

Output: filtered_edu/<same stem>.jsonl  (trainable corpus for mix)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from train.config import (
    EDU_DIR,
    FILTERED_DIR,
    FINEWEB_EDU_BATCH_SIZE,
    FINEWEB_EDU_CLASSIFIER,
    FINEWEB_EDU_MAX_LENGTH,
    FINEWEB_EDU_SKIP_STEMS,
    FINEWEB_EDU_THRESHOLD,
)

logger = logging.getLogger(__name__)


def int_edu_score(score: float) -> int:
    """FineWeb-edu convention: round continuous logit to 0–5."""
    return int(round(max(0.0, min(float(score), 5.0))))


class FineWebEduClassifier:
    """Thin wrapper around HuggingFaceFW/fineweb-edu-classifier."""

    def __init__(
        self,
        model_id: str = FINEWEB_EDU_CLASSIFIER,
        device: str | None = None,
        max_length: int = FINEWEB_EDU_MAX_LENGTH,
        batch_size: int = FINEWEB_EDU_BATCH_SIZE,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info("Loading FineWeb-edu classifier %s on %s", model_id, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    def score_texts(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores: list[float] = []
        torch = self.torch
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits.squeeze(-1).float()
                if logits.ndim == 0:
                    scores.append(float(logits.item()))
                else:
                    scores.extend(float(x) for x in logits.detach().cpu().tolist())
        return scores


def score_jsonl(
    in_path: Path,
    out_path: Path,
    *,
    classifier: FineWebEduClassifier | None = None,
    min_int_score: int | None = None,
    reuse_existing_score: bool = True,
    limit: int | None = None,
) -> dict:
    """Score one JSONL; write keepers with edu_score / edu_int_score fields."""
    min_int = int(FINEWEB_EDU_THRESHOLD) if min_int_score is None else min_int_score
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: load rows (streaming write would need classifier mid-stream — batch in chunks)
    kept = discarded = scored_n = reused = 0
    pending_docs: list[dict] = []
    pending_texts: list[str] = []

    def flush(clf: FineWebEduClassifier, fout) -> None:
        nonlocal kept, discarded, scored_n
        if not pending_docs:
            return
        scores = clf.score_texts(pending_texts)
        scored_n += len(scores)
        for doc, score in zip(pending_docs, scores):
            i_score = int_edu_score(score)
            meta = dict(doc.get("metadata") or {})
            meta["edu_score"] = score
            meta["edu_int_score"] = i_score
            meta["edu_classifier"] = FINEWEB_EDU_CLASSIFIER
            doc = dict(doc)
            doc["metadata"] = meta
            doc["edu_score"] = score
            doc["edu_int_score"] = i_score
            if i_score >= min_int:
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1
            else:
                discarded += 1
        pending_docs.clear()
        pending_texts.clear()

    clf = classifier
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line_i, line in enumerate(fin):
            if limit is not None and (kept + discarded) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            text = doc.get("text") or ""
            if not text.strip():
                discarded += 1
                continue

            # Reuse FineWeb-edu dataset scores when already present and trusted
            if reuse_existing_score:
                existing = doc.get("edu_int_score")
                if existing is None:
                    meta = doc.get("metadata") or {}
                    existing = meta.get("edu_int_score")
                    if existing is None and meta.get("score") is not None:
                        try:
                            existing = int_edu_score(float(meta["score"]))
                        except (TypeError, ValueError):
                            existing = None
                if existing is not None:
                    i_score = int(existing)
                    reused += 1
                    if i_score >= min_int:
                        doc = dict(doc)
                        doc["edu_int_score"] = i_score
                        fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        kept += 1
                    else:
                        discarded += 1
                    continue

            if clf is None:
                clf = FineWebEduClassifier()
            pending_docs.append(doc)
            pending_texts.append(text)
            if len(pending_docs) >= (clf.batch_size if clf else FINEWEB_EDU_BATCH_SIZE):
                flush(clf, fout)
            if (line_i + 1) % 2000 == 0:
                logger.info(
                    "%s progress lines=%d kept=%d discarded=%d",
                    in_path.name,
                    line_i + 1,
                    kept,
                    discarded,
                )
        if pending_docs:
            if clf is None:
                clf = FineWebEduClassifier()
            flush(clf, fout)

    stats = {
        "in": str(in_path),
        "out": str(out_path),
        "kept": kept,
        "discarded": discarded,
        "scored_with_model": scored_n,
        "reused_existing_score": reused,
        "min_int_score": min_int,
        "classifier": FINEWEB_EDU_CLASSIFIER,
    }
    logger.info("%s", stats)
    return stats


def score_all_filtered(
    filtered_dir: Path | None = None,
    edu_dir: Path | None = None,
    *,
    min_int_score: int | None = None,
    skip_stems: frozenset[str] | None = None,
    limit_per_file: int | None = None,
    device: str | None = None,
) -> dict:
    filtered_dir = filtered_dir or FILTERED_DIR
    edu_dir = edu_dir or EDU_DIR
    skip = skip_stems if skip_stems is not None else FINEWEB_EDU_SKIP_STEMS
    edu_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in filtered_dir.glob("*.jsonl") if p.stem not in skip)
    if not paths:
        logger.warning("No JSONL in %s to score", filtered_dir)
        return {"files": [], "edu_dir": str(edu_dir)}

    clf = FineWebEduClassifier(device=device)
    reports = []
    for path in paths:
        out = edu_dir / path.name
        reports.append(
            score_jsonl(
                path,
                out,
                classifier=clf,
                min_int_score=min_int_score,
                limit=limit_per_file,
            )
        )
    return {"files": reports, "edu_dir": str(edu_dir), "n_files": len(reports)}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(
        description="Apply FineWeb-edu classifier to filtered JSONL (post-datagen)"
    )
    parser.add_argument("--all", action="store_true", help="Score every filtered/*.jsonl → filtered_edu/")
    parser.add_argument("--in", dest="in_path", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--filtered-dir", type=str, default=str(FILTERED_DIR))
    parser.add_argument("--edu-dir", type=str, default=str(EDU_DIR))
    parser.add_argument(
        "--min-int-score",
        type=int,
        default=int(FINEWEB_EDU_THRESHOLD),
        help="Keep docs with int_score >= this (FineWeb-Edu-score-2 uses 2)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max docs per file (smoke)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Ignore existing metadata scores; always run the classifier",
    )
    args = parser.parse_args()

    reuse = not args.rescore

    if args.all:
        filtered_dir = Path(args.filtered_dir)
        edu_dir = Path(args.edu_dir)
        edu_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(p for p in filtered_dir.glob("*.jsonl") if p.stem not in FINEWEB_EDU_SKIP_STEMS)
        clf = FineWebEduClassifier(device=args.device) if paths else None
        reports = []
        for path in paths:
            assert clf is not None
            reports.append(
                score_jsonl(
                    path,
                    edu_dir / path.name,
                    classifier=clf,
                    min_int_score=args.min_int_score,
                    reuse_existing_score=reuse,
                    limit=args.limit,
                )
            )
        print(json.dumps({"files": reports, "edu_dir": str(edu_dir), "n_files": len(reports)}, indent=2))
        return

    if not args.in_path:
        parser.error("Provide --all or --in")
    in_path = Path(args.in_path)
    out_path = Path(args.out) if args.out else Path(args.edu_dir) / in_path.name
    clf = FineWebEduClassifier(device=args.device)
    stats = score_jsonl(
        in_path,
        out_path,
        classifier=clf,
        min_int_score=args.min_int_score,
        reuse_existing_score=reuse,
        limit=args.limit,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
