"""Phase 3.1 — heuristic cleaning for retained documents."""

from __future__ import annotations

import re

from train.config import MAX_DOC_CHARS, MIN_DOC_CHARS

_HTML = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _HTML.sub(" ", text)
    text = text.replace("\xa0", " ")
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def quality_ok(text: str) -> bool:
    if not text:
        return False
    n = len(text)
    if n < MIN_DOC_CHARS or n > MAX_DOC_CHARS:
        return False
    # Extreme special-character ratio
    special = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in ".,;:'\"!?()-"))
    if special / n > 0.3:
        return False
    return True


def clean_document(doc: dict) -> dict | None:
    text = clean_text(doc.get("text") or doc.get("content") or "")
    if not quality_ok(text):
        return None
    out = dict(doc)
    out["text"] = text
    return out
