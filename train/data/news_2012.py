"""Phase 1.6 — contemporaneous news text published in the cutoff year (2012).

Two API backends, both with reliable per-article publication timestamps:

  guardian  — The Guardian Open Platform content API (full article bodyText).
              Free developer key: https://open-platform.theguardian.com/access/
              export GUARDIAN_API_KEY=xxxx
  wikinews  — English Wikinews (CC-BY) via the MediaWiki API. Articles are found
              through the daily date categories ("Category:March 5, 2012") and the
              revision active at CUTOFF is used, so later edits never leak in.

  export SN38_DATA_ROOT=/workspace/sn38-data
  python -m train.data.news_2012 --backend wikinews --max-docs 50          # smoke, no key
  python -m train.data.news_2012 --backend guardian --max-docs 200
  python -m train.data.news_2012 --backend all --full-scale --finalize
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from train.config import CUTOFF, CUTOFF_ISO, CUTOFF_YEAR, CUTOFF_YEAR_DOC_WEIGHT, FILTERED_DIR, RAW_DIR
from train.data.clean import clean_document
from train.data.token_budget import TARGET_TOKENS, est_tokens, hit_target
from train.data.wikitext import is_redirect, wikitext_to_text
from train.filter.pipeline import decide, filter_jsonl

logger = logging.getLogger(__name__)

SOURCE = "news_2012"
USER_AGENT = "sn38-train-news-2012/0.1 (ChronoLLM miner dataset build)"

GUARDIAN_URL = os.environ.get("GUARDIAN_API_URL", "https://content.guardianapis.com/search")
# developer keys allow ~1 call/s; page-size max is 200
GUARDIAN_SLEEP = float(os.environ.get("GUARDIAN_SLEEP", "1.1"))
GUARDIAN_PAGE_SIZE = int(os.environ.get("GUARDIAN_PAGE_SIZE", "200"))
# Opinion / live blogs / galleries are noisy or mostly non-prose
GUARDIAN_SKIP_TYPES = frozenset({"liveblog", "gallery", "picture", "video", "audio", "crossword", "interactive"})

WIKINEWS_API = os.environ.get("WIKINEWS_API_URL", "https://en.wikinews.org/w/api.php")
WIKINEWS_SLEEP = float(os.environ.get("WIKINEWS_SLEEP", "0.2"))


def _http_json(url: str, params: dict, *, retries: int = 5) -> dict:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 429 = rate limit; back off. 4xx otherwise is a real error.
            if e.code != 429 and 400 <= e.code < 500:
                raise
            wait = 2**attempt * 5
            logger.warning("HTTP %s, retry in %ds", e.code, wait)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            wait = 2**attempt * 5
            logger.warning("network error %s, retry in %ds", e, wait)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url}")


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return None


def _in_cutoff_year(when: datetime | None) -> bool:
    return when is not None and when.year == CUTOFF_YEAR and when <= CUTOFF


def _days(year: int) -> Iterator[date]:
    d = date(year, 1, 1)
    while d.year == year:
        yield d
        d += timedelta(days=1)


# ---------------------------------------------------------------------------
# Guardian
# ---------------------------------------------------------------------------
def guardian_result_to_record(item: dict) -> dict | None:
    """Map one Guardian /search result (with show-fields=bodyText,headline) to a record."""
    if item.get("type") in GUARDIAN_SKIP_TYPES:
        return None
    when = _parse_iso(item.get("webPublicationDate", ""))
    if not _in_cutoff_year(when):
        return None
    fields = item.get("fields") or {}
    body = (fields.get("bodyText") or "").strip()
    if not body:
        return None
    headline = (fields.get("headline") or item.get("webTitle") or "").strip()
    pub = when.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "text": f"{headline}\n\n{pub[:10]}. {body}" if headline else body,
        "title": headline,
        "url": item.get("webUrl"),
        "source": SOURCE,
        "publication_date": pub,
        "loss_weight": CUTOFF_YEAR_DOC_WEIGHT,
        "year": CUTOFF_YEAR,
        "metadata": {
            "publisher": "guardian",
            "id": item.get("id"),
            "section": item.get("sectionName"),
            "type": item.get("type"),
            "publication_date": pub,
        },
    }


def iter_guardian(api_key: str, year: int = CUTOFF_YEAR) -> Iterator[dict]:
    """One day at a time so every query stays well under the API's page limits."""
    for day in _days(year):
        page, pages = 1, 1
        while page <= pages:
            data = _http_json(
                GUARDIAN_URL,
                {
                    "from-date": day.isoformat(),
                    "to-date": day.isoformat(),
                    "show-fields": "bodyText,headline",
                    "page-size": GUARDIAN_PAGE_SIZE,
                    "page": page,
                    "order-by": "oldest",
                    "use-date": "published",
                    "api-key": api_key,
                },
            )
            resp = data.get("response") or {}
            if resp.get("status") != "ok":
                raise RuntimeError(f"Guardian API error: {resp.get('message') or data}")
            pages = int(resp.get("pages") or 0)
            for item in resp.get("results") or []:
                yield item
            page += 1
            time.sleep(GUARDIAN_SLEEP)
        logger.info("guardian day %s done (%d pages)", day, pages)


# ---------------------------------------------------------------------------
# Wikinews
# ---------------------------------------------------------------------------
def wikinews_category(day: date) -> str:
    return f"Category:{day.strftime('%B')} {day.day}, {day.year}"


def iter_wikinews_titles(year: int = CUTOFF_YEAR) -> Iterator[tuple[str, date]]:
    for day in _days(year):
        cont: dict = {}
        while True:
            data = _http_json(
                WIKINEWS_API,
                {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": wikinews_category(day),
                    "cmnamespace": 0,
                    "cmtype": "page",
                    "cmlimit": 500,
                    "format": "json",
                    **cont,
                },
            )
            for m in data.get("query", {}).get("categorymembers", []):
                yield m["title"], day
            if "continue" not in data:
                break
            cont = data["continue"]
            time.sleep(WIKINEWS_SLEEP)


def _wikinews_revisions(title: str) -> tuple[dict | None, dict | None]:
    """(first revision, revision active at CUTOFF) for a Wikinews article."""
    common = {"action": "query", "prop": "revisions", "titles": title, "rvslots": "main", "format": "json"}

    def one(extra: dict) -> dict | None:
        data = _http_json(WIKINEWS_API, {**common, **extra})
        for page in data.get("query", {}).get("pages", {}).values():
            revs = page.get("revisions") or []
            return revs[0] if revs else None
        return None

    first = one({"rvprop": "ids|timestamp", "rvlimit": 1, "rvdir": "newer"})
    at_cutoff = one(
        {"rvprop": "ids|timestamp|content", "rvlimit": 1, "rvstart": CUTOFF_ISO, "rvdir": "older"}
    )
    return first, at_cutoff


def wikinews_to_record(title: str, first: dict | None, rev: dict | None) -> dict | None:
    if not first or not rev:
        return None
    created = _parse_iso(first.get("timestamp", ""))
    rev_ts = _parse_iso(rev.get("timestamp", ""))
    # article must be created in the cutoff year and the text taken from <= CUTOFF
    if not _in_cutoff_year(created) or rev_ts is None or rev_ts > CUTOFF:
        return None
    wikitext = rev["*"] if "*" in rev else ((rev.get("slots") or {}).get("main") or {}).get("*", "")
    if not wikitext or is_redirect(wikitext):
        return None
    body = wikitext_to_text(wikitext).strip()
    if not body:
        return None
    pub = created.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "text": f"{title}\n\n{pub[:10]}. {body}",
        "title": title,
        "url": f"https://en.wikinews.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
        "source": SOURCE,
        "publication_date": pub,
        "loss_weight": CUTOFF_YEAR_DOC_WEIGHT,
        "year": CUTOFF_YEAR,
        "metadata": {
            "publisher": "wikinews",
            "rev_id": rev.get("revid"),
            "revision_timestamp": rev.get("timestamp"),
            "publication_date": pub,
        },
    }


def iter_wikinews(year: int = CUTOFF_YEAR) -> Iterator[dict]:
    seen: set[str] = set()
    for title, _day in iter_wikinews_titles(year):
        if title in seen:
            continue
        seen.add(title)
        first, rev = _wikinews_revisions(title)
        rec = wikinews_to_record(title, first, rev)
        time.sleep(WIKINEWS_SLEEP)
        if rec is not None:
            yield rec


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _finish(rec: dict) -> dict | None:
    cleaned = clean_document(rec)
    if cleaned is None:
        return None
    if not decide(cleaned, source=SOURCE).keep:
        return None
    return cleaned


def stream_news(
    out_path: Path,
    backends: list[str],
    *,
    max_docs: int | None = None,
    target_tokens: int | None = None,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = n_tokens = 0
    per_backend: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as out:
        for backend in backends:
            if backend == "guardian":
                key = os.environ.get("GUARDIAN_API_KEY")
                if not key:
                    logger.warning("GUARDIAN_API_KEY not set — skipping guardian backend")
                    continue
                records = (guardian_result_to_record(it) for it in iter_guardian(key))
            elif backend == "wikinews":
                records = iter_wikinews()
            else:
                raise SystemExit(f"unknown backend {backend!r}")
            for rec in records:
                rec = _finish(rec) if rec is not None else None
                if rec is None:
                    skipped += 1
                    continue
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                per_backend[backend] = per_backend.get(backend, 0) + 1
                n_tokens += est_tokens(rec["text"])
                if kept % 500 == 0:
                    logger.info("news kept=%d skipped=%d est_tokens=%d", kept, skipped, n_tokens)
                if (max_docs is not None and kept >= max_docs) or hit_target(n_tokens, target_tokens):
                    break
            if (max_docs is not None and kept >= max_docs) or hit_target(n_tokens, target_tokens):
                break
    stats = {
        "source": SOURCE,
        "backends": per_backend,
        "kept": kept,
        "skipped": skipped,
        "est_tokens": n_tokens,
        "target_tokens": target_tokens,
        "out": str(out_path),
    }
    logger.info("%s", stats)
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description=f"{CUTOFF_YEAR} news extractor (Guardian API + Wikinews)")
    parser.add_argument("--backend", choices=["guardian", "wikinews", "all"], default="all")
    parser.add_argument("--out", type=str, default=str(RAW_DIR / "news_2012" / "news_2012.jsonl"))
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--full-scale", action="store_true", help=f"Use plan target {TARGET_TOKENS['news_2012']} tokens")
    parser.add_argument("--finalize", action="store_true", help="Also write filtered JSONL")
    args = parser.parse_args()

    backends = ["wikinews", "guardian"] if args.backend == "all" else [args.backend]
    target = TARGET_TOKENS["news_2012"] if args.full_scale else args.target_tokens
    stats = stream_news(Path(args.out), backends, max_docs=args.max_docs, target_tokens=target)
    if args.finalize:
        stats["filter"] = filter_jsonl(Path(args.out), FILTERED_DIR / f"{SOURCE}.jsonl", source=SOURCE)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
