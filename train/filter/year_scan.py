"""Phase 2.3 — internal year scanning for post-cutoff past-tense leakage."""

from __future__ import annotations

import re
from functools import lru_cache

from train.config import CUTOFF_YEAR

# Window around a year mention; past-tense / event verbs → likely leakage
_PAST_HINT = re.compile(
    r"(?i)\b("
    r"was|were|had|did|been|became|announced|elected|won|died|launched|"
    r"released|founded|occurred|happened|invaded|signed|passed|defeated|"
    r"collapsed|acquired|merged|published|introduced|opened|closed"
    r")\b"
)

# Speculative / future framing near the year → usually safe
_SAFE_HINT = re.compile(
    r"(?i)\b("
    r"by|until|before|expect|expected|will|would|plan|plans|projected|"
    r"forecast|goal|target|aim|hoping|envision"
    r")\b"
)


@lru_cache(maxsize=8)
def _year_re(cutoff_year: int) -> re.Pattern[str]:
    post = list(range(int(cutoff_year) + 1, 2027))
    if not post:
        return re.compile(r"(?!)")  # never matches
    return re.compile(r"\b(" + "|".join(str(y) for y in post) + r")\b")


def find_post_cutoff_years(text: str, cutoff_year: int | None = None) -> list[tuple[int, int, int]]:
    """Return list of (year, start, end) for post-cutoff year mentions."""
    cy = CUTOFF_YEAR if cutoff_year is None else int(cutoff_year)
    return [(int(m.group(1)), m.start(), m.end()) for m in _year_re(cy).finditer(text or "")]


def is_past_event_leak(text: str, window: int = 40, cutoff_year: int | None = None) -> bool:
    """Soft: True if a post-cutoff year appears near past-tense event language.

    Speculative framing ('By 2020 we expect...') is treated as safe.
    Bare years alone (e.g. 'ISO 2014') are kept.
    """
    if not text:
        return False
    cy = CUTOFF_YEAR if cutoff_year is None else int(cutoff_year)
    for year, start, end in find_post_cutoff_years(text, cutoff_year=cy):
        left = max(0, start - window)
        right = min(len(text), end + window)
        ctx = text[left:right]
        if _SAFE_HINT.search(ctx) and not _PAST_HINT.search(ctx):
            continue
        if _PAST_HINT.search(ctx):
            return True
        # Bare year with no safe framing — soft mode keeps these.
        if year >= cy + 1:
            continue
    return False


def has_hard_year_leak(text: str, window: int = 40, cutoff_year: int | None = None) -> bool:
    """Hard: drop any post-cutoff year unless the window is *only* speculative.

    Bare years (no safe framing) are leaks. Past-tense near the year is a leak.
    'By 2020 we expect...' with no past hint is kept.
    """
    if not text:
        return False
    cy = CUTOFF_YEAR if cutoff_year is None else int(cutoff_year)
    for _year, start, end in find_post_cutoff_years(text, cutoff_year=cy):
        left = max(0, start - window)
        right = min(len(text), end + window)
        ctx = text[left:right]
        if _SAFE_HINT.search(ctx) and not _PAST_HINT.search(ctx):
            continue
        return True
    return False


def has_any_post_cutoff_year(text: str, cutoff_year: int | None = None) -> bool:
    """Strictest: any post-cutoff calendar year token → leak (no safe exceptions)."""
    return bool(find_post_cutoff_years(text, cutoff_year=cutoff_year))


def is_year_leak(text: str, mode: str = "soft", cutoff_year: int | None = None, window: int = 40) -> bool:
    """Dispatch soft | hard | any."""
    m = (mode or "soft").lower()
    if m == "soft":
        return is_past_event_leak(text, window=window, cutoff_year=cutoff_year)
    if m == "hard":
        return has_hard_year_leak(text, window=window, cutoff_year=cutoff_year)
    if m == "any":
        return has_any_post_cutoff_year(text, cutoff_year=cutoff_year)
    raise ValueError(f"unknown year-leak mode {mode!r}; use soft|hard|any")
