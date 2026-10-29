"""Filter mix JSONL with an OpenAI judge — only the row's ``text`` is scored.

Uses the template in ``GPT4.1_Filter_Prompt.txt`` (``{document}`` placeholder).
Keeps label==0; drops label==1 (post-cutoff / low-quality / uncertain).

  export OPENAI_API_KEY=sk-...
  # sample first (cheap)
  python -m train.data.gpt_content_filter \\
    --in $SN38_DATA_ROOT/mix/train_mix_timestamped_2011.jsonl \\
    --out $SN38_DATA_ROOT/mix/train_mix_gpt_clean_2011.jsonl \\
    --max-docs 1000 --concurrency 8

  # full run (expensive — resume-safe)
  python -m train.data.gpt_content_filter \\
    --in $SN38_DATA_ROOT/mix/train_mix_timestamped_2011.jsonl \\
    --out $SN38_DATA_ROOT/mix/train_mix_gpt_clean_2011.jsonl \\
    --concurrency 16 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = ROOT / "GPT4.1_Filter_Prompt.txt"
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from repo .env into os.environ (no overwrite if already set)."""
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def load_prompt_template(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "{document}" not in text:
        raise SystemExit(f"prompt file must contain {{document}}: {path}")
    return text


def adapt_prompt_for_cutoff(template: str, cutoff_year: int | None) -> str:
    """If cutoff_year set, rewrite 2011/2012 wording to that cutoff boundary.

    Prompt semantics: drop content that requires knowledge after ``cutoff_year``
    (i.e. introduced in cutoff_year+1 or later).
    """
    if cutoff_year is None:
        return template
    # File is written for "prior to 2012" / "after 2011"
    next_year = int(cutoff_year) + 1
    out = template
    out = out.replace("prior to the year 2012", f"prior to the year {next_year}")
    out = out.replace("after 2011", f"after {cutoff_year}")
    out = out.replace("prior to 2012", f"prior to {next_year}")
    out = out.replace("post-2011", f"post-{cutoff_year}")
    return out


def build_user_message(template: str, document: str) -> str:
    return template.replace("{document}", document)


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    # keep head+tail so mid-doc leaks can still be seen
    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head] + "\n\n[...truncated...]\n\n" + text[-tail:]


def parse_judge_json(content: str) -> dict[str, Any]:
    content = (content or "").strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = _JSON_RE.search(content)
        if not m:
            raise
        data = json.loads(m.group(0))
    label = int(data.get("label", 1))
    if label not in (0, 1):
        label = 1
    conf = data.get("confidence", 0)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    term = data.get("suspected term") or data.get("suspected_term") or "none"
    return {"label": label, "confidence": conf, "suspected term": str(term)}


def judge_one(
    client,
    model: str,
    template: str,
    text: str,
    *,
    max_chars: int,
    temperature: float,
) -> dict[str, Any]:
    doc = truncate_text(text or "", max_chars)
    if not doc.strip():
        return {"label": 1, "confidence": 10, "suspected term": "empty", "error": "empty_text"}
    messages = [
        {
            "role": "user",
            "content": build_user_message(template, doc),
        }
    ]
    kwargs: Exception | None = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            parsed = parse_judge_json(raw)
            parsed["raw"] = raw
            return parsed
        except Exception as e:
            err = e
            time.sleep(min(2**attempt, 30))
    return {
        "label": 1,
        "confidence": 0,
        "suspected term": "api_error",
        "error": str(err),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    p = argparse.ArgumentParser(description="GPT content filter on mix JSONL text field only")
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Kept rows (label==0)")
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=DEFAULT_PROMPT,
        help="Template with {document} (default: repo GPT4.1_Filter_Prompt.txt)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_FILTER_MODEL", "gpt-4.1"),
        help="OpenAI model id (default gpt-4.1 or OPENAI_FILTER_MODEL)",
    )
    p.add_argument(
        "--cutoff-year",
        type=int,
        default=None,
        help="Rewrite prompt years for this cutoff (e.g. 2011). Default: keep file wording (2011/2012)",
    )
    p.add_argument("--max-docs", type=int, default=0, help="0=all; sample first N")
    p.add_argument("--max-chars", type=int, default=12000, help="Truncate long texts for the API")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--min-confidence", type=float, default=0.0, help="If label=0 but conf < this, drop")
    p.add_argument(
        "--drop-log",
        type=Path,
        default=None,
        help="JSONL of dropped rows + judge fields (default: out + .drops.jsonl)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-processed line indices recorded in state file",
    )
    p.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Resume state JSON (default: out + .gpt_filter_state.json)",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (prefer env OPENAI_API_KEY; avoid committing keys)",
    )
    p.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="Read API key from a file (e.g. /workspace/openai_key.txt on RunPod)",
    )
    args = p.parse_args()

    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if not api_key and args.api_key_file is not None:
        if not args.api_key_file.is_file():
            raise SystemExit(f"missing --api-key-file: {args.api_key_file}")
        api_key = args.api_key_file.read_text(encoding="utf-8").strip().strip('"').strip("'")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY missing. On RunPod use either:\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "  or: echo 'sk-...' > /workspace/openai_key.txt && --api-key-file /workspace/openai_key.txt\n"
            "Do not hardcode secrets into .py files."
        )

    if not args.inp.is_file():
        raise SystemExit(f"missing input: {args.inp}")
    if not args.prompt_file.is_file():
        raise SystemExit(f"missing prompt file: {args.prompt_file}")

    template = adapt_prompt_for_cutoff(load_prompt_template(args.prompt_file), args.cutoff_year)
    drop_log = args.drop_log or args.out.with_suffix(args.out.suffix + ".drops.jsonl")
    state_path = args.state or args.out.with_suffix(args.out.suffix + ".gpt_filter_state.json")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, max_retries=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    drop_log.parent.mkdir(parents=True, exist_ok=True)

    done_through = -1
    if args.resume and state_path.is_file():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        done_through = int(st.get("last_index", -1))
        logger.info("Resume: skip indices 0..%d", done_through)

    # Load pending jobs (index, row) — stream; only hold a concurrency-sized batch
    scanned = kept = dropped = errors = 0
    t0 = time.time()

    mode_out = "a" if args.resume and args.out.is_file() and done_through >= 0 else "w"
    mode_drop = "a" if mode_out == "a" else "w"

    def flush_state(idx: int):
        state_path.write_text(
            json.dumps(
                {
                    "in": str(args.inp),
                    "out": str(args.out),
                    "last_index": idx,
                    "kept": kept,
                    "dropped": dropped,
                    "errors": errors,
                    "model": args.model,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    with args.inp.open(encoding="utf-8") as fin, args.out.open(mode_out, encoding="utf-8") as fout, drop_log.open(
        mode_drop, encoding="utf-8"
    ) as fdrop:
        batch: list[tuple[int, dict]] = []

        def process_batch(items: list[tuple[int, dict]]):
            nonlocal kept, dropped, errors
            if not items:
                return
            results: dict[int, dict] = {}
            with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
                futs = {
                    ex.submit(
                        judge_one,
                        client,
                        args.model,
                        template,
                        (row.get("text") or ""),
                        max_chars=args.max_chars,
                        temperature=args.temperature,
                    ): idx
                    for idx, row in items
                }
                for fut in as_completed(futs):
                    idx = futs[fut]
                    results[idx] = fut.result()

            for idx, row in items:
                verdict = results[idx]
                if verdict.get("error"):
                    errors += 1
                label = int(verdict.get("label", 1))
                conf = float(verdict.get("confidence", 0) or 0)
                keep = label == 0 and conf >= args.min_confidence
                if keep:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    drop_rec = {
                        "index": idx,
                        "source": row.get("source"),
                        "label": label,
                        "confidence": conf,
                        "suspected term": verdict.get("suspected term"),
                        "error": verdict.get("error"),
                        "snippet": (row.get("text") or "")[:240].replace("\n", " "),
                    }
                    fdrop.write(json.dumps(drop_rec, ensure_ascii=False) + "\n")
                    dropped += 1
            fout.flush()
            fdrop.flush()
            last_idx = items[-1][0]
            flush_state(last_idx)
            logger.info(
                "progress last_idx=%d kept=%d dropped=%d errors=%d elapsed=%.0fs",
                last_idx, kept, dropped, errors, time.time() - t0,
            )

        for i, line in enumerate(fin):
            if args.max_docs and scanned >= args.max_docs:
                break
            line = line.strip()
            if not line:
                continue
            if i <= done_through:
                continue
            row = json.loads(line)
            # Only text is sent to the model; other fields stay on the row if kept
            batch.append((i, row))
            scanned += 1
            if len(batch) >= max(1, args.concurrency * 4):
                process_batch(batch)
                batch = []

        process_batch(batch)

    report = {
        "in": str(args.inp),
        "out": str(args.out),
        "drop_log": str(drop_log),
        "model": args.model,
        "cutoff_year_adapt": args.cutoff_year,
        "scanned_this_run": scanned,
        "kept": kept,
        "dropped": dropped,
        "errors": errors,
        "keep_rate": round(kept / max(kept + dropped, 1), 6),
    }
    rep_path = args.out.with_suffix(args.out.suffix + ".gpt_filter_report.json")
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("done → %s", json.dumps(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
