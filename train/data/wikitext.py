"""Lightweight MediaWiki wikitext → plain text (no external deps)."""

from __future__ import annotations

import re

_NOWIKI = re.compile(r"<nowiki>.*?</nowiki>", re.I | re.S)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF = re.compile(r"<ref\b[^>]*>.*?</ref>", re.I | re.S)
_REF_SELF = re.compile(r"<ref\b[^/]*/>", re.I)
_TAG = re.compile(r"</?(?:div|span|poem|gallery|includeonly|noinclude|onlyinclude|syntaxhighlight|source|code|math|sup|sub|small|big|center|blockquote)[^>]*>", re.I)
_ANY_TAG = re.compile(r"<[^>]+>")
_FILE = re.compile(r"\[\[(?:File|Image|Media):[^\]\n]+\]\]", re.I)
_CATEGORY = re.compile(r"\[\[Category:[^\]]+\]\]", re.I)
_LINK_NAMED = re.compile(r"\[\[([^|\]]+)\|([^\]]+)\]\]")
_LINK_SIMPLE = re.compile(r"\[\[([^\]]+)\]\]")
_EXT_LINK = re.compile(r"\[https?://[^\]\s]+ ([^\]]+)\]")
_EXT_BARE = re.compile(r"\[https?://[^\]\s]+\]")
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_BOLD_ITAL = re.compile(r"'{2,}")
_HEADING = re.compile(r"^={1,6}\s*(.*?)\s*={1,6}\s*$", re.M)
_TABLE_ROW = re.compile(r"^\|.*$", re.M)
_TABLE_MARK = re.compile(r"^\{\|.*$|^\|\}.*$|^\|-.*$|^!.*$", re.M)
_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")
_REDIRECT = re.compile(r"^#\s*redirect\b", re.I)


def is_redirect(wikitext: str) -> bool:
    return bool(_REDIRECT.match((wikitext or "").lstrip()))


def _strip_templates(text: str, rounds: int = 8) -> str:
    """Remove nested {{templates}} iteratively (best-effort)."""
    prev = None
    for _ in range(rounds):
        if prev == text:
            break
        prev = text
        text = _TEMPLATE.sub(" ", text)
    # leftover braces
    text = text.replace("{{", " ").replace("}}", " ")
    return text


def wikitext_to_text(wikitext: str) -> str:
    if not wikitext:
        return ""
    text = wikitext.replace("\r\n", "\n")
    text = _NOWIKI.sub(" ", text)
    text = _COMMENT.sub(" ", text)
    text = _REF.sub(" ", text)
    text = _REF_SELF.sub(" ", text)
    text = _FILE.sub(" ", text)
    text = _CATEGORY.sub(" ", text)
    text = _strip_templates(text)
    text = _TAG.sub(" ", text)
    text = _ANY_TAG.sub(" ", text)
    text = _LINK_NAMED.sub(r"\2", text)
    text = _LINK_SIMPLE.sub(r"\1", text)
    text = _EXT_LINK.sub(r"\1", text)
    text = _EXT_BARE.sub(" ", text)
    text = _BOLD_ITAL.sub("", text)
    text = _HEADING.sub(r"\n\1\n", text)
    text = _TABLE_MARK.sub(" ", text)
    text = _TABLE_ROW.sub(" ", text)
    text = text.replace("'''", "").replace("''", "")
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()
