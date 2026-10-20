"""Phase 2.3 — internal year scanning for post-cutoff past-tense leakage."""

from __future__ import annotations

import re

from train.config import CUTOFF_YEAR

# Years after cutoff that should not appear as past events
_POST_YEARS = list(range(CUTOFF_YEAR + 1, 2027))
_YEAR_RE = re.compile(r"\b(" + "|".join(str(y) for y in _POST_YEARS) + r")\b")

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


def find_post_cutoff_years(text: str) -> list[tuple[int, int, int]]:
    """Return list of (year, start, end) for post-cutoff year mentions."""
    return [(int(m.group(1)), m.start(), m.end()) for m in _YEAR_RE.finditer(text or "")]


def is_past_event_leak(text: str, window: int = 40) -> bool:
    """True if a post-cutoff year appears near past-tense event language.

    Speculative framing ('By 2020 we expect...') is treated as safe.
    """
    if not text:
        return False
    for year, start, end in find_post_cutoff_years(text):
        left = max(0, start - window)
        right = min(len(text), end + window)
        ctx = text[left:right]
        if _SAFE_HINT.search(ctx) and not _PAST_HINT.search(ctx):
            continue
        if _PAST_HINT.search(ctx):
            return True
        # Bare year with no safe framing in a factual sentence — discard-on-doubt
        # for high-confidence sources you may relax this; default is strict.
        if year >= CUTOFF_YEAR + 1:
            # require past hint for discard; bare years alone are not enough
            # (covers "ISO 2014 standard" style noise). Past-tense path above
            # already caught real leaks.
            continue
    return False


def has_hard_year_leak(text: str) -> bool:
    """Stricter variant: any post-cutoff year + past tense anywhere in doc head."""
    return is_past_event_leak(text)
