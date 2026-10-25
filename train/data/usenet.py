"""Phase 1.5 — Usenet posts dated <= CUTOFF (every post carries a Date header).

Two backends:

  hf    — stream a Hugging Face Usenet dataset (gated corpora need HF_TOKEN).
          The dataset id is configurable because mirrors differ; field names
          (text/body, date, newsgroup) are auto-detected per row.
            export USENET_DATASET=<owner/name>   [USENET_CONFIG, USENET_DATA_FILES]
  mbox  — local mbox / .mbox.gz / .mbox.zip files, e.g. the Internet Archive
          "usenet-*" collections (archive.org/details/usenethistorical, giganews
          dumps). Pass files or directories with --mbox.

Post-processing per message: strip headers, quoted replies ("> ..."), signatures
("-- " line) and binaries; keep preferred hierarchies (comp., sci., ... by default).
Posts from the cutoff year get CUTOFF_YEAR_DOC_WEIGHT.

  python -m train.data.usenet --backend mbox --mbox /workspace/usenet_raw --max-docs 1000
  python -m train.data.usenet --backend hf --full-scale --finalize
  python -m train.data.usenet --backend hf --min-year 2012 --finalize   # 2012-only slice
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import mailbox
import os
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable, Iterator

from train.config import CUTOFF, CUTOFF_YEAR, CUTOFF_YEAR_DOC_WEIGHT, DEFAULT_LOSS_WEIGHT, FILTERED_DIR, RAW_DIR
from train.data.clean import clean_document
from train.data.token_budget import TARGET_TOKENS, est_tokens, hit_target
from train.filter.pipeline import decide, filter_jsonl

logger = logging.getLogger(__name__)

SOURCE = "usenet"
HF_DATASET = os.environ.get("USENET_DATASET", "")
HF_CONFIG = os.environ.get("USENET_CONFIG") or None
HF_DATA_FILES = os.environ.get("USENET_DATA_FILES") or None

# train_plan: comp.* and sci.* are the cleanest; a few others add general-knowledge discussion
DEFAULT_HIERARCHIES = ("comp.", "sci.", "news.", "rec.", "soc.", "misc.", "humanities.")
SKIP_GROUP_PATTERNS = re.compile(r"(^alt\.binaries|\.binaries\.|\.test$|^control\.|^junk$)")

_QUOTE = re.compile(r"^\s*(>|\|>|:)")
_ATTRIBUTION = re.compile(r"^.{0,120}\b(wrote|writes|said)\s*:\s*$", re.I)
_UUENCODE = re.compile(r"^begin [0-7]{3} ", re.M)
_TEXT_KEYS = ("text", "body", "content", "message")
_DATE_KEYS = ("date", "Date", "timestamp", "created_at", "posted", "datetime")
_GROUP_KEYS = ("newsgroup", "newsgroups", "group", "Newsgroups")
_SUBJECT_KEYS = ("subject", "Subject", "title")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_post_date(value) -> datetime | None:
    """RFC 2822 Date header, ISO string, or epoch → aware UTC datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    # Usenet starts 1980; anything else is a broken header
    return dt if 1980 <= dt.year <= 2100 else None


def clean_post_body(body: str) -> str:
    """Drop quoted replies, attributions, signatures and uuencoded binaries."""
    if not body or _UUENCODE.search(body):
        return ""
    lines = []
    for line in body.replace("\r\n", "\n").split("\n"):
        if line.rstrip() == "--":  # "-- " signature delimiter
            break
        if _QUOTE.match(line) or _ATTRIBUTION.match(line):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def group_allowed(groups: str, hierarchies: tuple[str, ...] | None) -> bool:
    names = [g.strip() for g in re.split(r"[,\s]+", groups or "") if g.strip()]
    if not names:
        return hierarchies is None
    if any(SKIP_GROUP_PATTERNS.search(g) for g in names):
        return False
    if hierarchies is None:
        return True
    return any(g.startswith(h) for g in names for h in hierarchies)


def post_to_record(
    *,
    body: str,
    date_value,
    groups: str,
    subject: str = "",
    message_id: str | None = None,
    hierarchies: tuple[str, ...] | None = DEFAULT_HIERARCHIES,
    min_year: int | None = None,
) -> dict | None:
    when = parse_post_date(date_value)
    if when is None or when > CUTOFF:
        return None
    if min_year is not None and when.year < min_year:
        return None
    if not group_allowed(groups, hierarchies):
        return None
    text = clean_post_body(body)
    if not text:
        return None
    subject = (subject or "").strip()
    if subject:
        text = f"{subject}\n\n{text}"
    pub = when.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "text": text,
        "source": SOURCE,
        "publication_date": pub,
        "trusted_pre_cutoff": True,
        "loss_weight": CUTOFF_YEAR_DOC_WEIGHT if when.year == CUTOFF_YEAR else DEFAULT_LOSS_WEIGHT,
        "year": when.year,
        "metadata": {
            "newsgroups": groups,
            "message_id": message_id,
            "publication_date": pub,
        },
    }


def _first(row: dict, keys: Iterable[str]):
    for k in keys:
        if row.get(k) not in (None, ""):
            return row[k]
    return None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def iter_hf_posts(**kw) -> Iterator[dict]:
    if not HF_DATASET and not HF_DATA_FILES:
        raise SystemExit("Set USENET_DATASET=<owner/name> (or USENET_DATA_FILES) for the hf backend")
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Install datasets: pip install datasets") from e
    if HF_DATA_FILES:
        fmt = "parquet" if ".parquet" in HF_DATA_FILES else "json"
        ds = load_dataset(fmt, data_files=HF_DATA_FILES, split="train", streaming=True)
    else:
        ds = load_dataset(HF_DATASET, name=HF_CONFIG, split="train", streaming=True)
    logger.info("Streaming Usenet from %s", HF_DATA_FILES or HF_DATASET)
    for row in ds:
        rec = post_to_record(
            body=str(_first(row, _TEXT_KEYS) or ""),
            date_value=_first(row, _DATE_KEYS),
            groups=str(_first(row, _GROUP_KEYS) or ""),
            subject=str(_first(row, _SUBJECT_KEYS) or ""),
            message_id=row.get("message_id") or row.get("id"),
            **kw,
        )
        yield rec


def _message_body(msg) -> str:
    if msg.is_multipart():
        parts = [p for p in msg.walk() if p.get_content_type() == "text/plain"]
        payload = parts[0].get_payload(decode=True) if parts else None
    else:
        payload = msg.get_payload(decode=True)
    if payload is None:
        return ""
    charset = msg.get_content_charset() or "latin-1"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("latin-1", errors="replace")


def _mbox_files(paths: list[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_dir():
            yield from sorted(x for x in p.rglob("*") if x.is_file() and (".mbox" in x.name or x.suffix in {".mbox", ".gz", ".zip"}))
        elif p.exists():
            yield p


def _open_mbox(path: Path) -> Iterator[mailbox.mbox]:
    """Yield mailbox objects; compressed inputs are expanded to a temp file first."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                with zf.open(name) as src, tempfile.NamedTemporaryFile(delete=False, suffix=".mbox") as tmp:
                    tmp.write(src.read())
                yield from _mbox_from_tmp(Path(tmp.name))
    elif path.suffix == ".gz":
        with gzip.open(path, "rb") as src, tempfile.NamedTemporaryFile(delete=False, suffix=".mbox") as tmp:
            while chunk := src.read(1 << 20):
                tmp.write(chunk)
        yield from _mbox_from_tmp(Path(tmp.name))
    else:
        yield mailbox.mbox(str(path), create=False)


def _mbox_from_tmp(tmp: Path) -> Iterator[mailbox.mbox]:
    try:
        box = mailbox.mbox(str(tmp), create=False)
        yield box
        box.close()
    finally:
        tmp.unlink(missing_ok=True)


def iter_mbox_posts(paths: list[Path], **kw) -> Iterator[dict | None]:
    for path in _mbox_files(paths):
        logger.info("mbox %s", path)
        for box in _open_mbox(path):
            for msg in box:
                yield post_to_record(
                    body=_message_body(msg),
                    date_value=msg.get("Date"),
                    groups=str(msg.get("Newsgroups") or ""),
                    subject=str(msg.get("Subject") or ""),
                    message_id=msg.get("Message-ID"),
                    **kw,
                )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def stream_usenet(
    records: Iterable[dict | None],
    out_path: Path,
    *,
    max_docs: int | None = None,
    target_tokens: int | None = None,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = n_tokens = 0
    by_year: dict[int, int] = {}
    seen_ids: set[str] = set()
    with open(out_path, "w", encoding="utf-8") as out:
        for rec in records:
            if rec is None:
                skipped += 1
                continue
            mid = rec["metadata"].get("message_id")
            # crossposts appear once per group in many archives
            if mid:
                if mid in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(mid)
            cleaned = clean_document(rec)
            if cleaned is None or not decide(cleaned, source=SOURCE).keep:
                skipped += 1
                continue
            out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1
            by_year[cleaned["year"]] = by_year.get(cleaned["year"], 0) + 1
            n_tokens += est_tokens(cleaned["text"])
            if kept % 10000 == 0:
                logger.info("usenet kept=%d skipped=%d est_tokens=%d", kept, skipped, n_tokens)
            if (max_docs is not None and kept >= max_docs) or hit_target(n_tokens, target_tokens):
                break
    stats = {
        "source": SOURCE,
        "kept": kept,
        "skipped": skipped,
        "est_tokens": n_tokens,
        "target_tokens": target_tokens,
        "by_year": dict(sorted(by_year.items())),
        "out": str(out_path),
    }
    logger.info("%s", stats)
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description="Usenet ≤cutoff extractor")
    parser.add_argument("--backend", choices=["hf", "mbox"], default="hf")
    parser.add_argument("--mbox", nargs="*", default=[], help="mbox files or directories (mbox backend)")
    parser.add_argument("--out", type=str, default=str(RAW_DIR / "usenet" / "usenet.jsonl"))
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--full-scale", action="store_true", help=f"Use plan target {TARGET_TOKENS['usenet']} tokens")
    parser.add_argument("--min-year", type=int, default=None, help=f"Only posts from this year on (e.g. {CUTOFF_YEAR})")
    parser.add_argument(
        "--hierarchies",
        type=str,
        default=",".join(DEFAULT_HIERARCHIES),
        help="Comma-separated newsgroup prefixes to keep; 'all' keeps every non-binary group",
    )
    parser.add_argument("--finalize", action="store_true", help="Also write filtered JSONL")
    args = parser.parse_args()

    hier = None if args.hierarchies == "all" else tuple(h.strip() for h in args.hierarchies.split(",") if h.strip())
    kw = {"hierarchies": hier, "min_year": args.min_year}
    if args.backend == "hf":
        records = iter_hf_posts(**kw)
    else:
        if not args.mbox:
            raise SystemExit("--mbox PATH [PATH ...] required for the mbox backend")
        records = iter_mbox_posts([Path(p) for p in args.mbox], **kw)

    target = TARGET_TOKENS["usenet"] if args.full_scale else args.target_tokens
    stats = stream_usenet(records, Path(args.out), max_docs=args.max_docs, target_tokens=target)
    if args.finalize:
        stats["filter"] = filter_jsonl(Path(args.out), FILTERED_DIR / f"{SOURCE}.jsonl", source=SOURCE)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
