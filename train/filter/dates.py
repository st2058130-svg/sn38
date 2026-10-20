"""Phase 2.1 — multi-signal publication date extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from train.config import CUTOFF

# URL path date: /2012/03/... or /2012-03-15/
_URL_DATE = re.compile(
    r"/(?P<y>19\d{2}|20[0-2]\d)/(?P<m>0[1-9]|1[0-2])(?:/(?P<d>0[1-9]|[12]\d|3[01]))?"
)
_URL_DATE_DASH = re.compile(
    r"/(?P<y>19\d{2}|20[0-2]\d)-(?P<m>0[1-9]|1[0-2])-(?P<d>0[1-9]|[12]\d|3[01])"
)

# "Published: March 5, 2012" / "Published on 2012-03-05"
_CONTENT_DATE = re.compile(
    r"(?i)(?:published|posted|updated|date)\s*(?:on|:)?\s*"
    r"(?:"
    r"(?P<mon>January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(?P<d>\d{1,2}),?\s+(?P<y>19\d{2}|20[0-2]\d)"
    r"|"
    r"(?P<y2>19\d{2}|20[0-2]\d)-(?P<m2>0[1-9]|1[0-2])-(?P<d2>0[1-9]|[12]\d|3[01])"
    r")"
)

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


@dataclass
class DateSignal:
    source: str  # metadata | url | warc | content
    when: datetime
    confidence: float  # 0–1


@dataclass
class DateEvidence:
    signals: list[DateSignal] = field(default_factory=list)

    @property
    def best(self) -> DateSignal | None:
        if not self.signals:
            return None
        return max(self.signals, key=lambda s: s.confidence)

    @property
    def has_conflict(self) -> bool:
        """True if any two signals disagree on calendar year."""
        years = {s.when.year for s in self.signals}
        return len(years) > 1

    @property
    def any_after_cutoff(self) -> bool:
        return any(s.when > CUTOFF for s in self.signals)


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_metadata_date(value: Any) -> datetime | None:
    """Parse common metadata date fields into aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, (int, float)):
        year = int(value)
        if 1900 <= year <= 2100:
            return datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        return None
    text = str(value).strip()
    if not text:
        return None
    # year only
    if re.fullmatch(r"(19|20)\d{2}", text):
        return datetime(int(text), 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d %B %Y",
        "%B %d, %Y",
    ):
        try:
            dt = datetime.strptime(text.replace("+00:00", "+0000"), fmt.replace("%z", "%z"))
            return _as_utc(dt)
        except ValueError:
            continue
    # fallback: leading YYYY-MM-DD
    m = re.match(r"(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})", text)
    if m:
        return datetime(int(m["y"]), int(m["m"]), int(m["d"]), tzinfo=timezone.utc)
    return None


def date_from_url(url: str) -> datetime | None:
    if not url:
        return None
    path = urlparse(url).path
    m = _URL_DATE.search(path) or _URL_DATE_DASH.search(path)
    if not m:
        return None
    y = int(m.group("y"))
    month = int(m.group("m"))
    day = int(m.groupdict().get("d") or m.groupdict().get("d2") or 1)
    try:
        return datetime(y, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


def date_from_content_head(text: str, head_chars: int = 2000) -> datetime | None:
    if not text:
        return None
    head = text[:head_chars]
    m = _CONTENT_DATE.search(head)
    if not m:
        return None
    if m.group("mon"):
        y, month, d = int(m.group("y")), _MONTHS[m.group("mon").lower()], int(m.group("d"))
    else:
        y, month, d = int(m.group("y2")), int(m.group("m2")), int(m.group("d2"))
    try:
        return datetime(y, month, d, tzinfo=timezone.utc)
    except ValueError:
        return None


def collect_evidence(
    *,
    metadata: dict | None = None,
    url: str | None = None,
    warc_date: Any = None,
    text: str | None = None,
) -> DateEvidence:
    """Gather date signals in priority order from train_plan Phase 2.1."""
    evidence = DateEvidence()
    metadata = metadata or {}

    for key in (
        "publication_date", "datePublished", "published", "date", "year",
        "article:published_time", "crawl_date",
    ):
        if key in metadata:
            dt = parse_metadata_date(metadata[key])
            if dt:
                # crawl_date is only an upper-bound hint — lower confidence
                conf = 0.55 if key == "crawl_date" else 0.95
                evidence.signals.append(DateSignal("metadata", dt, conf))
                break

    if url:
        dt = date_from_url(url)
        if dt:
            evidence.signals.append(DateSignal("url", dt, 0.7))

    if warc_date is not None:
        dt = parse_metadata_date(warc_date)
        if dt:
            evidence.signals.append(DateSignal("warc", dt, 0.5))

    if text:
        dt = date_from_content_head(text)
        if dt:
            evidence.signals.append(DateSignal("content", dt, 0.65))

    return evidence
