"""WMT News Crawl (statmt) → ``news_2012`` JSONL for the mix news slot.

Source: https://data.statmt.org/news-crawl/en/news.YYYY.en.shuffled.deduped.gz
Year-tagged monolingual news lines (shuffled). We pack lines into short docs,
tag ``publication_date`` from the file year, and write streaming JSONL.

Mix key stays ``news_2012`` (see MIX_SHARES) even when CUTOFF_YEAR is 2011.

  export SN38_DATA_ROOT=/workspace/sn38-data
  # cutoff year only (fastest; fills the ~100–300M news budget easily)
  python -m train.data.wmt_news --years 2011 --finalize
  # fuller archive through cutoff
  python -m train.data.wmt_news --years 2008-2011 --finalize
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import urllib.request
from pathlib import Path

from train.config import (
    CUTOFF,
    CUTOFF_ISO,
    CUTOFF_YEAR,
    CUTOFF_YEAR_DOC_WEIGHT,
    FILTERED_DIR,
    MAX_DOC_CHARS,
    MIN_DOC_CHARS,
    RAW_DIR,
)
from train.data.clean import clean_document
from train.data.token_budget import TARGET_TOKENS, est_tokens, hit_target
from train.filter.pipeline import decide

logger = logging.getLogger(__name__)

SOURCE = "news_2012"
BASE_URL = os.environ.get(
    "WMT_NEWS_BASE_URL",
    "https://data.statmt.org/news-crawl/en",
)
# Pack shuffled sentences into docs so MIN_DOC_CHARS is reachable
_PACK_TARGET_CHARS = int(os.environ.get("WMT_NEWS_PACK_CHARS", "3000"))
_USER_AGENT = "sn38-train-wmt-news/0.1 (ChronoLLM miner dataset build)"


def _parse_years(spec: str) -> list[int]:
    """'2011' or '2008-2011' → sorted year list (must be ≤ CUTOFF_YEAR)."""
    spec = spec.strip()
    if "-" in spec and not spec.startswith("-"):
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
    else:
        lo = hi = int(spec)
    if lo > hi:
        lo, hi = hi, lo
    years = [y for y in range(lo, hi + 1) if y <= CUTOFF_YEAR]
    if not years:
        raise SystemExit(f"No years in {spec!r} with y <= CUTOFF_YEAR={CUTOFF_YEAR}")
    return years


def _gz_name(year: int) -> str:
    return f"news.{year}.en.shuffled.deduped.gz"


def _gz_url(year: int) -> str:
    return f"{BASE_URL.rstrip('/')}/{_gz_name(year)}"


def download_year(year: int, dest_dir: Path, *, force: bool = False) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / _gz_name(year)
    if path.exists() and path.stat().st_size > 1_000_000 and not force:
        logger.info("Using cached %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
        return path
    url = _gz_url(year)
    tmp = path.with_suffix(path.suffix + ".part")
    logger.info("Downloading %s → %s", url, path)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(path)
    logger.info("Saved %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    return path


def _flush_pack(
    parts: list[str],
    *,
    year: int,
    fout,
) -> tuple[int, int]:
    """Write one packed doc if it passes clean + decide. Returns (kept, tokens)."""
    if not parts:
        return 0, 0
    text = "\n".join(parts).strip()
    parts.clear()
    pub = f"{year}-12-31T23:59:59Z"
    weight = CUTOFF_YEAR_DOC_WEIGHT if year == CUTOFF_YEAR else 1.0
    doc = {
        "text": text,
        "source": SOURCE,
        "publication_date": pub,
        "loss_weight": weight,
        "year": year,
        "trusted_pre_cutoff": True,
        "metadata": {
            "publisher": "wmt_news_crawl",
            "year": year,
            "publication_date": pub,
            "cutoff": CUTOFF_ISO,
        },
    }
    cleaned = clean_document(doc)
    if cleaned is None:
        return 0, 0
    decision = decide(cleaned, source=SOURCE)
    if not decision.keep:
        return 0, 0
    cleaned["_filter"] = {
        "keep": True,
        "reason": decision.reason,
        "best_date": decision.best_date,
        "source_hint": SOURCE,
    }
    cleaned["_cutoff"] = CUTOFF.strftime("%Y-%m-%dT%H:%M:%SZ")
    fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
    return 1, est_tokens(cleaned["text"])


def stream_year_gz(
    gz_path: Path,
    year: int,
    fout,
    *,
    max_docs: int | None,
    target_tokens: int | None,
    kept_so_far: int,
    tokens_so_far: int,
) -> tuple[int, int, int]:
    """Pack lines from one year file. Returns (kept_delta, tokens_delta, skipped_lines)."""
    pack: list[str] = []
    pack_chars = 0
    kept = tokens = skipped = 0
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if len(line) > MAX_DOC_CHARS:
                skipped += 1
                continue
            # flush if adding this line would overshoot pack target (and we already have enough)
            if pack and pack_chars + len(line) + 1 > _PACK_TARGET_CHARS and pack_chars >= MIN_DOC_CHARS:
                k, t = _flush_pack(pack, year=year, fout=fout)
                kept += k
                tokens += t
                pack_chars = 0
                if max_docs is not None and kept_so_far + kept >= max_docs:
                    break
                if hit_target(tokens_so_far + tokens, target_tokens):
                    break
            pack.append(line)
            pack_chars += len(line) + 1
            if pack_chars >= _PACK_TARGET_CHARS:
                k, t = _flush_pack(pack, year=year, fout=fout)
                kept += k
                tokens += t
                pack_chars = 0
                if max_docs is not None and kept_so_far + kept >= max_docs:
                    break
                if hit_target(tokens_so_far + tokens, target_tokens):
                    break
        if pack and (max_docs is None or kept_so_far + kept < max_docs):
            if not hit_target(tokens_so_far + tokens, target_tokens):
                k, t = _flush_pack(pack, year=year, fout=fout)
                kept += k
                tokens += t
    return kept, tokens, skipped


def build_wmt_news(
    years: list[int],
    *,
    out_filtered: Path,
    cache_dir: Path,
    max_docs: int | None = None,
    target_tokens: int | None = None,
    force_download: bool = False,
) -> dict:
    out_filtered.parent.mkdir(parents=True, exist_ok=True)
    kept = tokens = skipped = 0
    per_year: dict[str, int] = {}
    with open(out_filtered, "w", encoding="utf-8") as fout:
        for year in years:
            gz = download_year(year, cache_dir, force=force_download)
            before = kept
            k, t, sk = stream_year_gz(
                gz,
                year,
                fout,
                max_docs=max_docs,
                target_tokens=target_tokens,
                kept_so_far=kept,
                tokens_so_far=tokens,
            )
            kept += k
            tokens += t
            skipped += sk
            per_year[str(year)] = kept - before
            logger.info(
                "year=%d +kept=%d total_kept=%d est_tokens=%d",
                year,
                kept - before,
                kept,
                tokens,
            )
            if max_docs is not None and kept >= max_docs:
                break
            if hit_target(tokens, target_tokens):
                logger.info("hit target_tokens=%d", target_tokens)
                break
    stats = {
        "source": SOURCE,
        "dataset": "WMT-news-crawl-en",
        "years": years,
        "kept": kept,
        "skipped_lines": skipped,
        "est_tokens": tokens,
        "target_tokens": target_tokens,
        "kept_by_year": per_year,
        "out": str(out_filtered),
        "cutoff": CUTOFF_ISO,
    }
    logger.info("%s", stats)
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    default_years = f"{max(2008, CUTOFF_YEAR - 3)}-{CUTOFF_YEAR}"
    parser = argparse.ArgumentParser(description="WMT News Crawl → news_2012 JSONL")
    parser.add_argument(
        "--years",
        type=str,
        default=str(CUTOFF_YEAR),
        help=f"Single year or range, e.g. 2011 or 2008-2011 (default: {CUTOFF_YEAR}; hint: {default_years})",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(RAW_DIR / "wmt_news"),
        help="Where to store .gz downloads",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(FILTERED_DIR / "news_2012.jsonl"),
        help="Output JSONL (default: filtered/news_2012.jsonl)",
    )
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument(
        "--full-scale",
        action="store_true",
        help=f"Stop around plan mid target {TARGET_TOKENS['news_2012']} tokens (or take all years if smaller)",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Write filtered output (streaming; recommended). Alias for normal write to --out.",
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    years = _parse_years(args.years)
    target = TARGET_TOKENS["news_2012"] if args.full_scale else args.target_tokens
    # full-scale: prefer taking all selected years; target only caps if enormous
    if args.full_scale:
        target = None

    out = Path(args.out)
    if not args.finalize and out == FILTERED_DIR / "news_2012.jsonl":
        # without --finalize, still write filtered path but warn
        logger.info("Writing %s (pass --finalize explicitly in scripts)", out)

    stats = build_wmt_news(
        years,
        out_filtered=out,
        cache_dir=Path(args.cache_dir),
        max_docs=args.max_docs,
        target_tokens=target,
        force_download=args.force_download,
    )
    print(json.dumps(stats, indent=2))
    if stats["kept"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
