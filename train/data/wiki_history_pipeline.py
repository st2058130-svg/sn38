"""Wikimedia enwiki pages-meta-history → trainable JSONL with low disk use.

Per shard (never keep the full ~241GB on disk):
  1) download one 7z/bz2 shard
  2) extract latest revision ≤ 2012-12-31 → temp raw JSONL
  3) clean + Phase-2 date filter → filtered part (trainable)
  4) delete the archive AND the raw extract immediately
  5) resume via state + filtered parts only

Final: merge filtered parts → filtered/wikipedia_2012.jsonl

Examples
--------
  export SN38_DATA_ROOT=/workspace/sn38-data
  python -m train.data.wiki_history_pipeline --dump-id 20261001 --list
  python -m train.data.wiki_history_pipeline --dump-id 20261001 --finalize
  python -m train.data.wiki_history_pipeline --dump-id 20261001 --max-shards 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from train.config import CUTOFF_ISO, FILTERED_DIR, RAW_DIR, STATE_DIR
from train.data.clean import clean_document
from train.data.wikipedia_2012 import extract_dump
from train.filter.pipeline import decide

logger = logging.getLogger(__name__)

DUMP_BASE = "https://dumps.wikimedia.org/enwiki"
USER_AGENT = "sn38-train-wiki-history/0.1 (ChronoLLM miner; RunPod datagen)"
MIN_SHARD_BYTES = 1_000_000
SOURCE = "wikipedia_2012"


def _request(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def resolve_dump_id(dump_id: str) -> str:
    """Resolve 'latest' to a dated dump that finished metahistory dumps."""
    if dump_id != "latest":
        return dump_id
    html = _request(f"{DUMP_BASE}/").decode("utf-8", errors="ignore")
    import re

    dates = sorted(set(re.findall(r'href="(\d{8})/"', html)), reverse=True)
    for d in dates:
        try:
            status = json.loads(_request(f"{DUMP_BASE}/{d}/dumpstatus.json"))
        except Exception:
            continue
        job = status.get("jobs", {}).get("metahistory7zdump") or status.get("jobs", {}).get(
            "metahistorybz2dump"
        )
        if job and job.get("status") == "done" and job.get("files"):
            logger.info("Resolved latest → %s", d)
            return d
    raise SystemExit("Could not resolve a completed dated dump from /enwiki/")


def list_history_shards(dump_id: str, fmt: str = "7z") -> list[dict]:
    """List history shards from dumpstatus.json (authoritative file set)."""
    dump_id = resolve_dump_id(dump_id)
    status_url = f"{DUMP_BASE}/{dump_id}/dumpstatus.json"
    status = json.loads(_request(status_url))
    job_key = "metahistory7zdump" if fmt == "7z" else "metahistorybz2dump"
    job = status.get("jobs", {}).get(job_key)
    if not job or not job.get("files"):
        alt = "metahistorybz2dump" if fmt == "7z" else "metahistory7zdump"
        job = status.get("jobs", {}).get(alt)
        if job and job.get("files"):
            logger.warning("%s missing; falling back to %s", job_key, alt)
            fmt = "7z" if alt.endswith("7z") else "bz2"
        else:
            raise SystemExit(f"No history job in {status_url}")

    if job.get("status") != "done":
        logger.warning("History job status=%s (may be incomplete)", job.get("status"))

    shards = []
    for name, meta in sorted(job["files"].items()):
        size = int(meta.get("size") or 0)
        if size < MIN_SHARD_BYTES:
            continue
        shards.append(
            {
                "name": name,
                "url": f"{DUMP_BASE}/{dump_id}/{name}",
                "size": size,
                "fmt": fmt,
                "dump_id": dump_id,
            }
        )
    total = sum(s["size"] for s in shards)
    logger.info(
        "Found %d history shards (%s) at %s — %.1f GB remote (peak local ≈ 1 shard)",
        len(shards),
        fmt,
        dump_id,
        total / 1e9,
    )
    return shards


def download_file(url: str, dest: Path, *, retries: int = 5) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading %s → %s (attempt %d)", url, dest.name, attempt)
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as out:
                total = resp.headers.get("Content-Length")
                total_i = int(total) if total else None
                done = 0
                t0 = time.time()
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total_i and done % (64 * 1024 * 1024) < len(chunk):
                        pct = 100.0 * done / total_i
                        mbps = (done / 1e6) / max(time.time() - t0, 1e-6)
                        logger.info(
                            "  %.1f%% (%d / %d MB) %.1f MB/s",
                            pct,
                            done // 1e6,
                            total_i // 1e6,
                            mbps,
                        )
            tmp.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.warning("Download failed: %s", e)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(60, 2**attempt))
    raise RuntimeError("unreachable")


def _state_path(dump_id: str, fmt: str) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / f"wiki_history_{dump_id}_{fmt}.json"


def load_state(dump_id: str, fmt: str) -> dict:
    path = _state_path(dump_id, fmt)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "dump_id": dump_id,
        "fmt": fmt,
        "cutoff": CUTOFF_ISO,
        "completed": [],
        "failed": {},
        "kept_filtered": 0,
    }


def save_state(dump_id: str, fmt: str, state: dict) -> None:
    _state_path(dump_id, fmt).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _shard_stem(name: str) -> str:
    stem = name
    for ext in (".xml.bz2", ".xml.7z", ".bz2", ".7z"):
        if stem.endswith(ext):
            return stem[: -len(ext)]
    return stem


def _merge_shard_jsonls(parts_dir: Path, out_jsonl: Path) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for part in sorted(parts_dir.glob("*.jsonl")):
            with open(part, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    return n


def filter_raw_to_trainable(raw_path: Path, filtered_part: Path) -> dict:
    """Clean + discard-on-doubt filter one shard's extract → trainable JSONL part."""
    filtered_part.parent.mkdir(parents=True, exist_ok=True)
    kept = discarded = 0
    reasons: dict[str, int] = {}
    with open(raw_path, encoding="utf-8") as fin, open(filtered_part, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            cleaned = clean_document(doc)
            if cleaned is None:
                discarded += 1
                reasons["quality"] = reasons.get("quality", 0) + 1
                continue
            decision = decide(cleaned, source=SOURCE)
            reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
            if not decision.keep:
                discarded += 1
                continue
            cleaned["_filter"] = {
                "keep": True,
                "reason": decision.reason,
                "best_date": decision.best_date,
                "source_hint": SOURCE,
            }
            cleaned["_cutoff"] = CUTOFF_ISO
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1
    stats = {"kept": kept, "discarded": discarded, "reasons": reasons, "out": str(filtered_part)}
    logger.info("Filter shard → kept=%d discarded=%d", kept, discarded)
    return stats


def process_shards(
    dump_id: str,
    *,
    fmt: str = "7z",
    max_shards: int | None = None,
    resume: bool = True,
    keep_shards: bool = False,
    keep_raw: bool = False,
    shard_limit_pages: int | None = None,
    out_jsonl: Path | None = None,
) -> dict:
    """Download→extract→filter→delete per shard. Disk peak ≈ one archive + one raw part."""
    shards = list_history_shards(dump_id, fmt=fmt)
    if not shards:
        raise SystemExit("No history shards found")
    dump_id = shards[0]["dump_id"]
    fmt = shards[0].get("fmt", fmt)
    if max_shards is not None:
        shards = shards[:max_shards]

    state = load_state(dump_id, fmt) if resume else {
        "dump_id": dump_id,
        "fmt": fmt,
        "cutoff": CUTOFF_ISO,
        "completed": [],
        "failed": {},
        "kept_filtered": 0,
    }
    done = set(state.get("completed") or [])

    shard_dir = RAW_DIR / "wikipedia_2012" / "shards" / dump_id
    raw_tmp_dir = RAW_DIR / "wikipedia_2012" / "raw_tmp" / dump_id
    filtered_parts_dir = RAW_DIR / "wikipedia_2012" / "filtered_parts" / dump_id
    filtered_parts_dir.mkdir(parents=True, exist_ok=True)
    raw_tmp_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_jsonl or (FILTERED_DIR / "wikipedia_2012.jsonl")

    t0 = time.time()
    for i, shard in enumerate(shards, 1):
        name = shard["name"]
        stem = _shard_stem(name)
        filtered_part = filtered_parts_dir / f"{stem}.jsonl"
        if name in done and filtered_part.exists():
            logger.info("[%d/%d] skip completed %s", i, len(shards), name)
            continue

        local = shard_dir / name
        raw_part = raw_tmp_dir / f"{stem}.jsonl"
        logger.info("[%d/%d] %s (%.2f GB remote)", i, len(shards), name, (shard["size"] or 0) / 1e9)
        try:
            if not local.exists() or local.stat().st_size < MIN_SHARD_BYTES:
                download_file(shard["url"], local)

            # 1) extract cutoff revisions to temp raw
            extract_stats = extract_dump(local, raw_part, limit=shard_limit_pages, append=False)

            # 2) immediately filter to trainable part
            fstats = filter_raw_to_trainable(raw_part, filtered_part)

            if name not in state["completed"]:
                state["completed"].append(name)
            state["failed"].pop(name, None)
            state["kept_filtered"] = int(state.get("kept_filtered") or 0) + int(fstats["kept"])
            save_state(dump_id, fmt, state)
            logger.info(
                "Shard done extract_kept=%s filtered_kept=%s",
                extract_stats.get("kept"),
                fstats["kept"],
            )
        except Exception as e:
            logger.exception("Shard failed %s: %s", name, e)
            state.setdefault("failed", {})[name] = str(e)
            save_state(dump_id, fmt, state)
            filtered_part.unlink(missing_ok=True)
            raise
        finally:
            # 3) never accumulate archives (241GB) or raw extracts
            if not keep_shards and local.exists():
                local.unlink(missing_ok=True)
                logger.info("Deleted archive %s", name)
            if not keep_raw and raw_part.exists():
                raw_part.unlink(missing_ok=True)
                logger.info("Deleted raw extract %s", raw_part.name)

    n_lines = _merge_shard_jsonls(filtered_parts_dir, out_jsonl)
    summary = {
        "dump_id": dump_id,
        "fmt": fmt,
        "shards_total": len(shards),
        "shards_completed": len(state["completed"]),
        "kept_filtered_total": n_lines,
        "filtered_parts_dir": str(filtered_parts_dir),
        "out": str(out_jsonl),
        "state": str(_state_path(dump_id, fmt)),
        "seconds": round(time.time() - t0, 1),
        "cutoff": CUTOFF_ISO,
        "note": "Archives deleted after each shard; only filtered JSONL parts retained",
    }
    return summary


def finalize_merge(dump_id: str, fmt: str = "7z", out_jsonl: Path | None = None) -> dict:
    """Re-merge existing filtered parts (e.g. after resume) into filtered/wikipedia_2012.jsonl."""
    dump_id = resolve_dump_id(dump_id)
    parts = RAW_DIR / "wikipedia_2012" / "filtered_parts" / dump_id
    out_jsonl = out_jsonl or (FILTERED_DIR / "wikipedia_2012.jsonl")
    n = _merge_shard_jsonls(parts, out_jsonl)
    return {"merged": n, "parts_dir": str(parts), "out": str(out_jsonl)}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(
        description="Low-disk Wikimedia enwiki meta-history → trainable 2012 JSONL"
    )
    parser.add_argument(
        "--dump-id",
        default=os.environ.get("WIKI_DUMP_ID", "20261001"),
        help="Dated dump id (recommended) or 'latest' to auto-resolve",
    )
    parser.add_argument(
        "--format",
        choices=["7z", "bz2"],
        default=os.environ.get("WIKI_DUMP_FORMAT", "7z"),
        help="7z ≈ 240GB remote total; peak disk ≈ one shard (default: 7z)",
    )
    parser.add_argument("--list", action="store_true", help="Only list shards + sizes")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--keep-shards", action="store_true", help="Do NOT delete 7z/bz2 (uses lots of disk)")
    parser.add_argument("--keep-raw", action="store_true", help="Do NOT delete raw extracts after filter")
    parser.add_argument("--shard-limit-pages", type=int, default=None)
    parser.add_argument("--out", type=str, default=None, help="Final filtered JSONL path")
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Process all shards (filter inline) and write filtered/wikipedia_2012.jsonl",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Only re-merge existing filtered parts (no download)",
    )
    args = parser.parse_args()

    if args.finalize_only:
        print(json.dumps(finalize_merge(args.dump_id, args.format, Path(args.out) if args.out else None), indent=2))
        return

    if args.list:
        shards = list_history_shards(args.dump_id, fmt=args.format)
        total = sum(s["size"] or 0 for s in shards)
        print(
            json.dumps(
                {
                    "dump_id": shards[0]["dump_id"] if shards else args.dump_id,
                    "format": shards[0].get("fmt") if shards else args.format,
                    "n_shards": len(shards),
                    "total_bytes_remote": total,
                    "total_gb_remote": round(total / 1e9, 2),
                    "peak_disk_hint_gb": "~2–15 (one archive + one raw part + growing filtered JSONL)",
                    "pipeline": "download → extract → filter → delete archive+raw",
                    "shards_sample": shards[:10],
                },
                indent=2,
            )
        )
        return

    # Default behaviour is the full low-disk pipeline; --finalize is accepted for CLI compatibility
    summary = process_shards(
        args.dump_id,
        fmt=args.format,
        max_shards=args.max_shards,
        resume=not args.no_resume,
        keep_shards=args.keep_shards,
        keep_raw=args.keep_raw,
        shard_limit_pages=args.shard_limit_pages,
        out_jsonl=Path(args.out) if args.out else None,
    )
    if not args.finalize and args.out is None:
        # still wrote filtered out by default in process_shards
        pass
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
