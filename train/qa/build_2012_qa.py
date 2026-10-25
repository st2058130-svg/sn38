"""Phase 4 — cutoff-year emphasis set from filtered Wikipedia JSONL.

Despite the historical name ("qa_2012"), this does NOT emit question/answer or
cloze templates. It selects whole paragraphs that talk about the cutoff year
(a bare "2012" year token or a dated phrase like "March 2012") and writes each
one with its article title as context, as natural prose:

    {title}\\n\\n{paragraph}

Why prose: SN38 leak probes score the log-prob of a completion phrase, and 70%
of the final score is GPT-judged completion quality — templated or fragmentary
text helps neither.

Guards:
  * only paragraphs about the cutoff year (not "the first N sentences")
  * list/table/reference-style lines are skipped
  * "remade in 2012"-style mentions of an older subject are not filtered out
    here, but no synthetic prefix is added, so nothing false is asserted
  * exact-duplicate paragraphs are written once
  * total size capped by --max-tokens (default QA_MAX_TOKENS, a few % of the mix)

Input records need: {"text": "...", "title": "..." (optional)}

  python -m train.qa.build_2012_qa --in $SN38_DATA_ROOT/filtered/wikipedia_2012.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

from train.config import CUTOFF_YEAR, QA_DIR, QA_LOSS_WEIGHT, QA_MAX_TOKENS
from train.data.token_budget import est_tokens

logger = logging.getLogger(__name__)

SOURCE = "qa_2012"
MIN_PARA_CHARS = 200
MAX_PARA_CHARS = 2500

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan\\.|Feb\\.|Mar\\.|Apr\\.|Jun\\.|Jul\\.|Aug\\.|Sep\\.|Sept\\.|Oct\\.|Nov\\.|Dec\\."
)
# standalone year token: not part of a longer number, ISBN, phone, "2012–13" season ranges are kept
_YEAR = re.compile(rf"(?<![\d.,/-]){CUTOFF_YEAR}(?![\d])")
_DATED = re.compile(rf"\b(?:{_MONTHS})\s+(?:\d{{1,2}},?\s+)?{CUTOFF_YEAR}\b|\b\d{{1,2}}\s+(?:{_MONTHS})\s+{CUTOFF_YEAR}\b")
_LISTY = re.compile(r"^\s*(?:[*#|!:;]|\{\||\d+\.\s)")
_PARA_SPLIT = re.compile(r"\n\s*\n")


def mentions_cutoff_year(paragraph: str) -> bool:
    return bool(_DATED.search(paragraph) or _YEAR.search(paragraph))


def is_prose(paragraph: str) -> bool:
    lines = [ln for ln in paragraph.split("\n") if ln.strip()]
    if not lines:
        return False
    listy = sum(1 for ln in lines if _LISTY.match(ln))
    if listy / len(lines) > 0.3:
        return False
    # prose has sentences
    return paragraph.count(". ") + paragraph.rstrip().endswith(".") >= 2


def _split_long(paragraph: str) -> list[str]:
    """Split an over-long paragraph on sentence ends into <= MAX_PARA_CHARS pieces."""
    if len(paragraph) <= MAX_PARA_CHARS:
        return [paragraph]
    pieces, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph):
        if cur and len(cur) + len(sent) + 1 > MAX_PARA_CHARS:
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def paragraphs_from_article(text: str, title: str | None = None) -> list[dict]:
    out = []
    for raw in _PARA_SPLIT.split(text or ""):
        para = raw.strip()
        if len(para) < MIN_PARA_CHARS or not is_prose(para):
            continue
        for piece in _split_long(para):
            if len(piece) < MIN_PARA_CHARS or not mentions_cutoff_year(piece):
                continue
            body = f"{title}\n\n{piece}" if title else piece
            out.append(
                {
                    "text": body,
                    "title": title,
                    "source": SOURCE,
                    "year": CUTOFF_YEAR,
                    "loss_weight": QA_LOSS_WEIGHT,
                    "metadata": {"origin": "wikipedia_2012"},
                }
            )
    return out


def build(
    in_path: Path,
    out_path: Path,
    limit: int | None = None,
    max_tokens: int | None = QA_MAX_TOKENS,
) -> dict:
    """``limit`` caps records, ``max_tokens`` caps estimated tokens (whichever hits first)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = n_dup = n_tokens = 0
    seen: set[bytes] = set()
    done = False
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if done:
                break
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            n_in += 1
            for rec in paragraphs_from_article(doc.get("text", ""), doc.get("title")):
                key = hashlib.blake2b(rec["text"].split("\n\n", 1)[-1].encode("utf-8"), digest_size=16).digest()
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
                n_tokens += est_tokens(rec["text"])
                if (limit is not None and n_out >= limit) or (max_tokens is not None and n_tokens >= max_tokens):
                    done = True
                    break
    stats = {
        "articles": n_in,
        "paragraphs": n_out,
        "duplicates": n_dup,
        "est_tokens": n_tokens,
        "max_tokens": max_tokens,
        "out": str(out_path),
    }
    logger.info("%s", stats)
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description=f"Select {CUTOFF_YEAR}-focused Wikipedia paragraphs as prose")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", default=str(QA_DIR / "qa_2012.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Max records")
    parser.add_argument("--max-tokens", type=int, default=QA_MAX_TOKENS, help="Max est. tokens (0 = no cap)")
    args = parser.parse_args()
    stats = build(Path(args.in_path), Path(args.out), args.limit, args.max_tokens or None)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
