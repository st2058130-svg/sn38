"""Phase 1.1 — extract English Wikipedia as-of end-2012 → JSONL.

Modes
-----
1) History dump (best purity):
   pages-meta-history XML; for each page keep the latest revision with
   timestamp <= 2012-12-31T23:59:59Z.

2) Articles dump (practical):
   pages-articles XML from a dump dated on/before early 2013. Uses each
   page's revision timestamp when present; otherwise requires --dump-date
   <= cutoff (or --allow-post-cutoff-dump-date with a warning).

3) API sample (smoke / small sets):
   MediaWiki Action API reconstructions for given titles at the cutoff.

Output record
-------------
{
  "text": "...",
  "title": "...",
  "source": "wikipedia_2012",
  "publication_date": "2012-06-01T12:00:00Z",
  "url": "https://en.wikipedia.org/wiki/Title",
  "trusted_pre_cutoff": true,
  "metadata": {"page_id": ..., "rev_id": ..., "namespace": 0}
}

Examples
--------
  # API smoke (no dump download)
  python -m train.data.wikipedia_2012 --api-sample --limit 20

  # History dump
  python -m train.data.wikipedia_2012 \\
    --dump enwiki-...-pages-meta-history.xml.bz2 --mode history

  # Articles dump near cutoff
  python -m train.data.wikipedia_2012 \\
    --dump enwiki-20130102-pages-articles.xml.bz2 --mode articles --dump-date 2013-01-02

Recommended dumps: https://dumps.wikimedia.org/enwiki/
Prefer a dump stamp on/just after 2012-12-31 (e.g. 20130102) for articles mode,
or any pages-meta-history dump for history mode.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import logging
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterator

from train.config import CUTOFF, CUTOFF_ISO, FILTERED_DIR, MIN_DOC_CHARS, RAW_DIR
from train.data.clean import clean_text, quality_ok
from train.data.wikitext import is_redirect, wikitext_to_text
from train.filter.pipeline import decide

logger = logging.getLogger(__name__)

SOURCE = "wikipedia_2012"
API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "sn38-train-wikipedia-2012/0.1 (ChronoLLM miner dataset build)"

# Sensible default titles for API smoke covering 2012-era facts
DEFAULT_API_TITLES = [
    "Barack Obama",
    "United States",
    "European Union",
    "Python (programming language)",
    "Higgs boson",
    "Apple Inc.",
    "Facebook",
    "Olympic Games",
    "2012 Summer Olympics",
    "Wikipedia",
    "World War II",
    "Albert Einstein",
    "Climate change",
    "NASA",
    "Microsoft",
    "Google",
    "London",
    "China",
    "India",
    "Association football",
]


@dataclass
class Revision:
    rev_id: str | None
    timestamp: datetime
    text: str


@dataclass
class PageHit:
    title: str
    page_id: str | None
    ns: int
    revision: Revision


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_mw_timestamp(value: str) -> datetime:
    # 2012-06-01T12:00:00Z
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_dump_date(value: str) -> datetime:
    value = value.strip()
    if re.fullmatch(r"\d{8}", value):
        return datetime(int(value[:4]), int(value[4:6]), int(value[6:8]), tzinfo=timezone.utc)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        y, m, d = value.split("-")
        return datetime(int(y), int(m), int(d), tzinfo=timezone.utc)
    return parse_mw_timestamp(value)


def title_to_url(title: str) -> str:
    slug = urllib.parse.quote(title.replace(" ", "_"), safe="()'")
    return f"https://en.wikipedia.org/wiki/{slug}"


def open_dump(path: Path):
    """Open a dump for streaming XML. Supports .xml, .gz, .bz2, .7z (via 7z -so)."""
    name = path.name.lower()
    if name.endswith(".7z"):
        import subprocess

        # Stream decompressed XML on stdout (needs p7zip/7z installed)
        proc = subprocess.Popen(
            ["7z", "x", "-so", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if proc.stdout is None:
            raise RuntimeError("7z failed to open stdout pipe — install p7zip-full")
        return _ProcStdout(proc)
    raw = open(path, "rb")
    if name.endswith(".bz2"):
        return bz2.BZ2File(raw)
    if name.endswith(".gz"):
        return gzip.GzipFile(fileobj=raw)
    return raw


class _ProcStdout:
    """File-like wrapper that closes the child process when the stream closes."""

    def __init__(self, proc):
        self.proc = proc
        self._fh = proc.stdout

    def read(self, n=-1):
        return self._fh.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        try:
            self._fh.close()
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait(timeout=30)


def _page_from_element(page_el: ET.Element) -> PageHit | None:
    title = page_id = None
    ns = 0
    best: Revision | None = None
    is_redir_el = False

    for child in page_el:
        tag = _local(child.tag)
        if tag == "title":
            title = (child.text or "").strip()
        elif tag == "id":
            page_id = (child.text or "").strip()
        elif tag == "ns":
            try:
                ns = int((child.text or "0").strip())
            except ValueError:
                ns = 0
        elif tag == "redirect":
            is_redir_el = True
        elif tag == "revision":
            rev = _revision_from_element(child)
            if rev is None:
                continue
            if rev.timestamp > CUTOFF:
                continue
            if best is None or rev.timestamp > best.timestamp:
                best = rev

    if is_redir_el or not title or best is None:
        return None
    if ns != 0:
        return None
    if is_redirect(best.text):
        return None
    return PageHit(title=title, page_id=page_id, ns=ns, revision=best)


def _revision_from_element(rev_el: ET.Element) -> Revision | None:
    rev_id = None
    timestamp = None
    text = ""
    for child in rev_el:
        tag = _local(child.tag)
        if tag == "id" and rev_id is None:
            rev_id = (child.text or "").strip()
        elif tag == "timestamp":
            try:
                timestamp = parse_mw_timestamp(child.text or "")
            except ValueError:
                return None
        elif tag == "text":
            # deleted revisions may have text missing / empty
            if child.get("{http://www.w3.org/2001/XMLSchema-instance}nil") == "true":
                return None
            text = child.text or ""
    if timestamp is None:
        return None
    return Revision(rev_id=rev_id, timestamp=timestamp, text=text)


def iter_dump_pages(path: Path) -> Iterator[PageHit]:
    """Stream MediaWiki XML; yield one PageHit per page (best pre-cutoff revision)."""
    with open_dump(path) as fh:
        context = ET.iterparse(fh, events=("end",))
        for event, elem in context:
            if _local(elem.tag) != "page":
                continue
            hit = _page_from_element(elem)
            elem.clear()
            if hit is not None:
                yield hit


def page_to_record(hit: PageHit, *, min_chars: int = MIN_DOC_CHARS) -> dict | None:
    plain = clean_text(wikitext_to_text(hit.revision.text))
    if not quality_ok(plain) or len(plain) < min_chars:
        return None
    pub = hit.revision.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    rec = {
        "text": plain,
        "title": hit.title,
        "source": SOURCE,
        "publication_date": pub,
        "url": title_to_url(hit.title),
        "trusted_pre_cutoff": True,
        "metadata": {
            "page_id": hit.page_id,
            "rev_id": hit.revision.rev_id,
            "namespace": hit.ns,
            "revision_timestamp": pub,
            "cutoff": CUTOFF_ISO,
        },
    }
    # Enforce Phase 2 rules even for trusted wiki
    decision = decide(rec, source=SOURCE)
    if not decision.keep:
        return None
    return rec


def extract_dump(
    dump_path: Path,
    out_path: Path,
    *,
    limit: int | None = None,
    min_chars: int = MIN_DOC_CHARS,
    append: bool = False,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_pages = n_kept = n_skip = 0
    t0 = time.time()
    mode = "a" if append else "w"
    with open(out_path, mode, encoding="utf-8") as out:
        for hit in iter_dump_pages(dump_path):
            n_pages += 1
            rec = page_to_record(hit, min_chars=min_chars)
            if rec is None:
                n_skip += 1
            else:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_kept += 1
            if n_pages % 1000 == 0:
                logger.info(
                    "progress pages=%d kept=%d skip=%d elapsed=%.0fs",
                    n_pages,
                    n_kept,
                    n_skip,
                    time.time() - t0,
                )
            if limit is not None and n_kept >= limit:
                break
    stats = {
        "mode": "dump",
        "dump": str(dump_path),
        "out": str(out_path),
        "append": append,
        "pages_seen_with_pre_cutoff_rev": n_pages,
        "kept": n_kept,
        "skipped_quality_or_filter": n_skip,
        "seconds": round(time.time() - t0, 1),
        "cutoff": CUTOFF_ISO,
    }
    logger.info("done: %s", stats)
    return stats


def _api_get(params: dict) -> dict:
    params = {**params, "format": "json"}
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{API_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_revision_at_cutoff(title: str) -> PageHit | None:
    """MediaWiki: revision active at CUTOFF (rvstart=cutoff, rvdir=older)."""
    data = _api_get(
        {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "rvlimit": 1,
            "rvstart": CUTOFF_ISO,
            "rvdir": "older",
            "redirects": 1,
        }
    )
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            return None
        revs = page.get("revisions") or []
        if not revs:
            return None
        rev = revs[0]
        ts = parse_mw_timestamp(rev["timestamp"])
        if ts > CUTOFF:
            return None
        # API shape: rev["*"] (legacy) or slots.main["*"] (current)
        if "*" in rev:
            text = rev["*"]
        else:
            text = ((rev.get("slots") or {}).get("main") or {}).get("*", "")
        if not text or is_redirect(text):
            return None
        return PageHit(
            title=page.get("title") or title,
            page_id=str(page.get("pageid")) if page.get("pageid") is not None else None,
            ns=int(page.get("ns") or 0),
            revision=Revision(rev_id=str(rev.get("revid")), timestamp=ts, text=text),
        )
    return None


def extract_api(
    titles: list[str],
    out_path: Path,
    *,
    limit: int | None = None,
    min_chars: int = MIN_DOC_CHARS,
    sleep_s: float = 0.2,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_kept = n_miss = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as out:
        for i, title in enumerate(titles):
            if limit is not None and n_kept >= limit:
                break
            try:
                hit = fetch_revision_at_cutoff(title)
            except Exception as e:
                logger.warning("API fail %s: %s", title, e)
                n_miss += 1
                time.sleep(sleep_s)
                continue
            if hit is None:
                n_miss += 1
                logger.info("no pre-cutoff revision: %s", title)
            else:
                rec = page_to_record(hit, min_chars=min_chars)
                if rec is None:
                    n_miss += 1
                else:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_kept += 1
                    logger.info("kept %s (%d chars) @ %s", hit.title, len(rec["text"]), rec["publication_date"])
            time.sleep(sleep_s)
            if (i + 1) % 10 == 0:
                logger.info("api progress %d/%d kept=%d", i + 1, len(titles), n_kept)
    stats = {
        "mode": "api",
        "out": str(out_path),
        "titles": len(titles),
        "kept": n_kept,
        "missed": n_miss,
        "seconds": round(time.time() - t0, 1),
        "cutoff": CUTOFF_ISO,
    }
    logger.info("done: %s", stats)
    return stats


def write_tiny_fixture_dump(path: Path) -> Path:
    """Write a tiny MediaWiki XML dump for offline tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    xml = """<?xml version="1.0" encoding="utf-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/">
  <page>
    <title>Test Article 2012</title>
    <ns>0</ns>
    <id>1</id>
    <revision>
      <id>10</id>
      <timestamp>2011-05-01T00:00:00Z</timestamp>
      <text xml:space="preserve">Old short.</text>
    </revision>
    <revision>
      <id>11</id>
      <timestamp>2012-06-15T12:00:00Z</timestamp>
      <text xml:space="preserve">'''Test Article 2012''' was a notable topic in 2012.

This paragraph is long enough to pass the minimum character quality gate used by the extractor pipeline for ChronoLLM dataset construction. It mentions CERN and the ongoing search related to particle physics without referencing later years in the past tense.</text>
    </revision>
    <revision>
      <id>12</id>
      <timestamp>2014-01-01T00:00:00Z</timestamp>
      <text xml:space="preserve">Post cutoff revision that must be ignored because it is after 2012.</text>
    </revision>
  </page>
  <page>
    <title>Redirect Page</title>
    <ns>0</ns>
    <id>2</id>
    <redirect title="Test Article 2012" />
    <revision>
      <id>20</id>
      <timestamp>2012-01-01T00:00:00Z</timestamp>
      <text xml:space="preserve">#REDIRECT [[Test Article 2012]]</text>
    </revision>
  </page>
  <page>
    <title>Talk:Something</title>
    <ns>1</ns>
    <id>3</id>
    <revision>
      <id>30</id>
      <timestamp>2012-01-01T00:00:00Z</timestamp>
      <text xml:space="preserve">Talk pages are skipped even when long enough for quality filters to accept them in main namespace.</text>
    </revision>
  </page>
  <page>
    <title>Only Post Cutoff</title>
    <ns>0</ns>
    <id>4</id>
    <revision>
      <id>40</id>
      <timestamp>2015-01-01T00:00:00Z</timestamp>
      <text xml:space="preserve">This page did not exist before the cutoff so it should not appear in the output JSONL file at all.</text>
    </revision>
  </page>
</mediawiki>
"""
    path.write_text(xml, encoding="utf-8")
    return path


def print_download_hints():
    print(
        f"""
Wikipedia 2012 download hints (cutoff {CUTOFF_ISO})
===================================================
Index: https://dumps.wikimedia.org/enwiki/

Articles mode (smaller, practical):
  Pick a dump stamp on/near 2013-01-02, e.g.:
    enwiki-YYYYMMDD-pages-articles.xml.bz2
  Then:
    python -m train.data.wikipedia_2012 --dump PATH --mode articles --dump-date YYYY-MM-DD

History mode (best temporal purity, much larger):
    enwiki-YYYYMMDD-pages-meta-history*.xml-*.bz2
  Then:
    python -m train.data.wikipedia_2012 --dump PATH --mode history

API smoke (no dump):
    python -m train.data.wikipedia_2012 --api-sample --limit 20

Output default:
    {RAW_DIR / "wikipedia_2012" / "wiki_2012.jsonl"}
""".strip()
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description="Extract Wikipedia-as-of-2012 → JSONL")
    parser.add_argument("--dump", type=str, default=None, help="Path to MediaWiki XML (.xml/.bz2/.gz)")
    parser.add_argument("--mode", choices=["history", "articles"], default="history")
    parser.add_argument("--dump-date", type=str, default=None, help="Dump stamp (articles mode)")
    parser.add_argument(
        "--allow-post-cutoff-dump-date",
        action="store_true",
        help="Allow articles dump dated after cutoff (NOT recommended)",
    )
    parser.add_argument("--api-sample", action="store_true", help="Fetch DEFAULT_API_TITLES at cutoff")
    parser.add_argument("--titles-file", type=str, default=None, help="Text file of titles (one per line)")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=MIN_DOC_CHARS)
    parser.add_argument("--hints", action="store_true", help="Print dump download hints")
    parser.add_argument("--write-fixture", type=str, default=None, help="Write tiny test dump XML to path")
    parser.add_argument(
        "--also-filter",
        action="store_true",
        help="Also write Phase-2 filtered copy under train/data/filtered/",
    )
    args = parser.parse_args()

    if args.hints:
        print_download_hints()
        return

    if args.write_fixture:
        p = write_tiny_fixture_dump(Path(args.write_fixture))
        print(json.dumps({"fixture": str(p)}, indent=2))
        return

    out = Path(args.out) if args.out else (RAW_DIR / "wikipedia_2012" / "wiki_2012.jsonl")

    if args.api_sample or args.titles_file:
        if args.titles_file:
            titles = [
                ln.strip() for ln in Path(args.titles_file).read_text(encoding="utf-8").splitlines() if ln.strip()
            ]
        else:
            titles = list(DEFAULT_API_TITLES)
        stats = extract_api(titles, out, limit=args.limit, min_chars=args.min_chars)
    elif args.dump:
        dump_path = Path(args.dump)
        if not dump_path.exists():
            raise SystemExit(f"Dump not found: {dump_path}")
        if args.mode == "articles":
            if not args.dump_date:
                raise SystemExit("articles mode requires --dump-date YYYY-MM-DD (or YYYYMMDD)")
            dump_dt = parse_dump_date(args.dump_date)
            if dump_dt > CUTOFF and not args.allow_post_cutoff_dump_date:
                raise SystemExit(
                    f"dump-date {dump_dt.date()} is after cutoff {CUTOFF.date()}. "
                    "Use a dump on/before cutoff, switch to --mode history, or pass "
                    "--allow-post-cutoff-dump-date (leak risk)."
                )
            if dump_dt > CUTOFF:
                logger.warning(
                    "articles dump-date %s is AFTER cutoff — revision timestamps still enforced, "
                    "but pages created after cutoff may appear if their first kept rev is wrong",
                    dump_dt.date(),
                )
        stats = extract_dump(dump_path, out, limit=args.limit, min_chars=args.min_chars)
        stats["declared_mode"] = args.mode
        if args.dump_date:
            stats["dump_date"] = args.dump_date
    else:
        parser.error("Provide --dump, --api-sample/--titles-file, --hints, or --write-fixture")

    if args.also_filter and Path(stats["out"]).exists():
        from train.filter.pipeline import filter_jsonl

        filtered = FILTERED_DIR / "wikipedia_2012.jsonl"
        fstats = filter_jsonl(Path(stats["out"]), filtered, source=SOURCE)
        stats["filter"] = fstats

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
