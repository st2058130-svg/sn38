"""Phase 5 — build the training mix JSONL from filtered sources (streaming).

Planning:
  1. estimate each source's unique tokens (sampled bytes→chars ratio × file size)
  2. give each source ``share × total_tokens``, capped at ``max_epochs × unique``;
     capped sources hand their unused share to the uncapped ones (water-filling)
  3. write mix_plan.json next to the output (also printed with --dry-run)

Writing (constant memory, never loads a whole shard):
  * each source is streamed ``repeat = planned / unique`` times: full passes, then a
    final pass that keeps each doc with probability ``repeat - floor(repeat)``
  * a per-source shuffle buffer varies document order between passes
  * long docs are split at paragraph boundaries (MIX_MAX_DOC_CHARS)
  * sources are interleaved by sampling ∝ remaining planned tokens, so every part
    of the file has the planned source proportions (no source-by-source curriculum)

  python -m train.data.mix --filtered-dir $SN38_DATA_ROOT/filtered_edu --dry-run
  python -m train.data.mix --filtered-dir ... --total-tokens 35e9 --share books=0.10 --max-epochs wikipedia_2012=8
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from pathlib import Path
from typing import Iterator

from train.config import (
    DEFAULT_LOSS_WEIGHT,
    MIX_DIR,
    MIX_MAX_DOC_CHARS,
    MIX_MAX_EPOCHS,
    MIX_SHARES,
    MIX_SHUFFLE_BUFFER_CHARS,
    MIX_TOTAL_TOKENS,
    QA_LOSS_WEIGHT,
    SEED,
)
from train.data.token_budget import est_tokens

logger = logging.getLogger(__name__)

_PARA = re.compile(r"\n\s*\n")


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def estimate_tokens(path: Path, sample_lines: int = 20_000) -> int:
    """Unique est. tokens in a JSONL shard: exact for small files, sampled for large ones."""
    size = path.stat().st_size
    read = chars = n = 0
    with open(path, "rb") as f:
        for raw in f:
            read += len(raw)
            raw = raw.strip()
            if not raw:
                continue
            chars += len(json.loads(raw).get("text") or "")
            n += 1
            if n >= sample_lines:
                break
    if read == 0:
        return 0
    if read >= size:
        return chars // 4
    return int(chars / read * size) // 4


def plan_mix(
    unique: dict[str, int],
    total_tokens: float | None,
    shares: dict[str, float],
    max_epochs: dict[str, float],
) -> dict:
    """Water-filling allocation of output tokens per source.

    Unlisted sources are taken exactly once. ``total_tokens=None`` means "the sum of
    unique tokens" (one average pass).
    """
    present = {s: u for s, u in unique.items() if u > 0}
    listed = {s: shares[s] for s in present if s in shares and shares[s] > 0}
    fixed = {s: float(u) for s, u in present.items() if s not in listed}
    total = float(total_tokens) if total_tokens else float(sum(present.values()))

    planned: dict[str, float] = dict(fixed)
    remaining = max(0.0, total - sum(fixed.values()))
    active = dict(listed)
    while active and remaining > 0:
        share_sum = sum(active.values())
        alloc = {s: remaining * w / share_sum for s, w in active.items()}
        capped = {s for s, a in alloc.items() if a > max_epochs.get(s, 1) * present[s]}
        if not capped:
            planned.update(alloc)
            remaining = 0.0
            break
        for s in capped:
            planned[s] = max_epochs.get(s, 1) * present[s]
            remaining -= planned[s]
            del active[s]
    for s in listed:
        planned.setdefault(s, 0.0)

    out_total = sum(planned.values())
    share_norm = sum(listed.values()) or 1.0
    sources = {}
    for s in sorted(present):
        sources[s] = {
            "unique_tokens": present[s],
            "planned_tokens": int(planned[s]),
            "repeat": round(planned[s] / present[s], 3),
            "max_epochs": max_epochs.get(s, 1) if s in listed else 1,
            "target_share": round(listed[s] / share_norm * (1 - sum(fixed.values()) / max(total, 1)), 4)
            if s in listed
            else None,
            "final_share": round(planned[s] / max(out_total, 1), 4),
            "capped": s in listed and planned[s] >= max_epochs.get(s, 1) * present[s] - 1,
        }
    return {
        "requested_total_tokens": int(total),
        "planned_total_tokens": int(out_total),
        "short_by_tokens": int(max(0.0, total - out_total)),
        "missing_listed_sources": sorted(s for s in shares if s not in present),
        "sources": sources,
    }


def log_plan(plan: dict) -> None:
    logger.info(
        "Mix plan: requested=%.2fB planned=%.2fB",
        plan["requested_total_tokens"] / 1e9,
        plan["planned_total_tokens"] / 1e9,
    )
    for s, p in plan["sources"].items():
        target = f"{p['target_share']:.1%}" if p["target_share"] is not None else "fixed"
        logger.info(
            "  %-16s unique=%6.2fB planned=%6.2fB repeat=%5.2f share=%5.1f%% (target %s)%s",
            s,
            p["unique_tokens"] / 1e9,
            p["planned_tokens"] / 1e9,
            p["repeat"],
            100 * p["final_share"],
            target,
            " CAPPED" if p["capped"] else "",
        )
        if p["target_share"] and p["final_share"] > 1.5 * p["target_share"]:
            logger.warning(
                "  %s got %.1f%% vs target %.1f%% because other sources hit their epoch caps — "
                "raise their --max-epochs, lower --total-tokens or lower this source's share",
                s, 100 * p["final_share"], 100 * p["target_share"],
            )
    if plan["short_by_tokens"] > 0:
        logger.warning(
            "Mix is %.2fB tokens short of the requested total — every listed source is at its epoch cap",
            plan["short_by_tokens"] / 1e9,
        )
    if plan["missing_listed_sources"]:
        logger.info("Listed in MIX_SHARES but not present: %s", ", ".join(plan["missing_listed_sources"]))


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
def split_long(text: str, max_chars: int) -> list[str]:
    """Split at paragraph boundaries into pieces <= max_chars (hard-split oversize paragraphs)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    cur = ""
    for para in _PARA.split(text):
        para = para.strip()
        if not para:
            continue
        while len(para) > max_chars:
            cut = para.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(para[:cut].strip())
            para = para[cut:].strip()
        if cur and len(cur) + 2 + len(para) > max_chars:
            pieces.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        pieces.append(cur)
    return pieces


def stream_source(
    path: Path,
    source: str,
    base_weight: float,
    repeat: float,
    *,
    seed: int,
    max_doc_chars: int = MIX_MAX_DOC_CHARS,
    buffer_chars: int = MIX_SHUFFLE_BUFFER_CHARS,
) -> Iterator[dict]:
    """Yield mix records for one source over ceil(repeat) passes."""
    for pass_idx in range(math.ceil(repeat)):
        keep_p = min(1.0, repeat - pass_idx)
        rng = random.Random(f"{seed}:{source}:{pass_idx}")
        buf: list[dict] = []
        held = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                # decide before parsing so skipped docs in a partial pass cost nothing
                if keep_p < 1.0 and rng.random() >= keep_p:
                    continue
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                text = doc.get("text") or ""
                if not text:
                    continue
                weight = float(doc.get("loss_weight", base_weight))
                for piece in split_long(text, max_doc_chars):
                    buf.append({"text": piece, "source": source, "loss_weight": weight, "year": doc.get("year")})
                    held += len(piece)
                while held > buffer_chars and buf:
                    rec = buf.pop(rng.randrange(len(buf)))
                    held -= len(rec["text"])
                    yield rec
        rng.shuffle(buf)
        yield from buf


def build_mix(
    shards: list[tuple[Path, str, float]],
    out_path: Path,
    *,
    total_tokens: float | None = MIX_TOTAL_TOKENS,
    shares: dict[str, float] | None = None,
    max_epochs: dict[str, float] | None = None,
    max_doc_chars: int = MIX_MAX_DOC_CHARS,
    buffer_chars: int = MIX_SHUFFLE_BUFFER_CHARS,
    dry_run: bool = False,
    seed: int = SEED,
) -> dict:
    """shards: list of (path, source_name, base_loss_weight). Source names must be unique."""
    shares = {**MIX_SHARES, **(shares or {})}
    max_epochs = {**MIX_MAX_EPOCHS, **(max_epochs or {})}

    by_source: dict[str, tuple[Path, float]] = {}
    for path, source, weight in shards:
        if not path.exists():
            logger.warning("Missing shard %s — skip", path)
            continue
        if source in by_source:
            raise ValueError(f"Duplicate source name {source!r} ({by_source[source][0]} and {path})")
        by_source[source] = (path, weight)

    unique = {s: estimate_tokens(p) for s, (p, _) in by_source.items()}
    plan = plan_mix(unique, total_tokens, shares, max_epochs)
    log_plan(plan)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path = out_path.with_name(out_path.stem + "_plan.json")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if dry_run:
        return {"plan": plan, "plan_path": str(plan_path), "dry_run": True}

    rng = random.Random(seed)
    streams = []
    for s, info in plan["sources"].items():
        if info["planned_tokens"] <= 0:
            continue
        path, weight = by_source[s]
        it = stream_source(
            path, s, weight, info["repeat"], seed=seed, max_doc_chars=max_doc_chars, buffer_chars=buffer_chars
        )
        streams.append([s, it, float(info["planned_tokens"])])

    docs: dict[str, int] = {}
    toks: dict[str, int] = {}
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        while streams:
            idx = rng.choices(range(len(streams)), weights=[max(st[2], 1.0) for st in streams])[0]
            s, it, _left = streams[idx]
            rec = next(it, None)
            if rec is None:
                streams.pop(idx)
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            t = est_tokens(rec["text"])
            streams[idx][2] -= t
            docs[s] = docs.get(s, 0) + 1
            toks[s] = toks.get(s, 0) + t
            n += 1
            if streams[idx][2] <= 0:
                streams.pop(idx)
            if n % 1_000_000 == 0:
                logger.info("mix wrote %dM records, %.2fB tokens", n // 1_000_000, sum(toks.values()) / 1e9)

    total_written = sum(toks.values())
    stats = {
        "records": n,
        "by_source": docs,
        "tokens_by_source": toks,
        "share_by_source": {s: round(t / max(total_written, 1), 4) for s, t in toks.items()},
        "est_tokens": total_written,
        "plan_path": str(plan_path),
        "out": str(out_path),
    }
    logger.info("Mix written: %s", stats)
    return stats


def parse_kv(items: list[str] | None) -> dict[str, float]:
    """['books=0.1', 'pes2o=0.2'] → {'books': 0.1, 'pes2o': 0.2}"""
    out = {}
    for item in items or []:
        key, _, val = item.partition("=")
        if not key or not val:
            raise SystemExit(f"Expected name=value, got {item!r}")
        out[key.strip()] = float(val)
    return out


def add_mix_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--total-tokens", type=float, default=MIX_TOTAL_TOKENS, help="Output size; match train_clm --total-tokens"
    )
    parser.add_argument("--share", action="append", help="Override MIX_SHARES, e.g. --share books=0.1")
    parser.add_argument("--max-epochs", action="append", help="Override MIX_MAX_EPOCHS, e.g. --max-epochs pes2o=5")
    parser.add_argument("--dry-run", action="store_true", help="Only estimate sizes and print/write the plan")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description="Build streaming weighted training mix")
    parser.add_argument("--filtered-dir", type=str, required=True)
    parser.add_argument("--qa", type=str, default=None, help="Optional 2012 QA JSONL")
    parser.add_argument("--out", type=str, default=str(MIX_DIR / "train_mix.jsonl"))
    add_mix_args(parser)
    args = parser.parse_args()

    filtered = Path(args.filtered_dir)
    shards = []
    for path in sorted(filtered.glob("*.jsonl")):
        source = path.stem.replace("_filtered", "")
        weight = QA_LOSS_WEIGHT if "qa" in source else DEFAULT_LOSS_WEIGHT
        shards.append((path, source, weight))
    if args.qa:
        shards.append((Path(args.qa), "qa_2012", QA_LOSS_WEIGHT))

    stats = build_mix(
        shards,
        Path(args.out),
        total_tokens=args.total_tokens,
        shares=parse_kv(args.share),
        max_epochs=parse_kv(args.max_epochs),
        dry_run=args.dry_run,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
