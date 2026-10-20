"""Phase 0.2 / 6.2 — leakage canary evaluation on a checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Allow `python -m train.canary.evaluate` from repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import CANARY_ALERT_RATE, CANARY_PATH, CUTOFF_ISO  # noqa: E402


def load_suite(path: Path | None = None) -> dict:
    path = path or CANARY_PATH
    with open(path, encoding="utf-8") as f:
        suite = json.load(f)
    if suite.get("cutoff") != CUTOFF_ISO:
        logger.warning("Canary suite cutoff %s != config %s", suite.get("cutoff"), CUTOFF_ISO)
    return suite


def _contains_leak(text: str, leaked_answers: list[str]) -> bool:
    lower = text.lower()
    return any(ans.lower() in lower for ans in leaked_answers)


def evaluate_generations(suite: dict, generations: dict[str, str]) -> dict:
    """Score precomputed {id: completion} generations against the suite."""
    items = suite["items"]
    leaked = []
    missing = []
    for item in items:
        cid = item["id"]
        if cid not in generations:
            missing.append(cid)
            continue
        if _contains_leak(generations[cid], item["leaked_answers"]):
            leaked.append({"id": cid, "completion": generations[cid][:200]})

    n = len(items) - len(missing)
    rate = (len(leaked) / n) if n else 0.0
    return {
        "n_items": len(items),
        "n_scored": n,
        "n_missing": len(missing),
        "n_leaked": len(leaked),
        "leak_rate": rate,
        "alert": rate > CANARY_ALERT_RATE,
        "alert_threshold": CANARY_ALERT_RATE,
        "leaked": leaked,
        "missing": missing,
    }


def evaluate_model(model_path: str, max_new_tokens: int = 32, device: str | None = None) -> dict:
    """Load a HF causal LM and generate completions for every canary prompt."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sn38.architectures import register_for

    register_for(model_path)
    suite = load_suite()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    generations = {}
    for item in suite["items"]:
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        generations[item["id"]] = text
        logger.info("%s -> %s", item["id"], text[:80].replace("\n", " "))

    result = evaluate_generations(suite, generations)
    result["generations"] = generations
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description="Run post-2012 leakage canary suite")
    parser.add_argument("--model", type=str, help="HF model path / repo")
    parser.add_argument("--generations", type=str, help="JSON file of {id: completion}")
    parser.add_argument("--suite", type=str, default=str(CANARY_PATH))
    parser.add_argument("--out", type=str, default=None, help="Write full report JSON")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if args.model:
        report = evaluate_model(args.model, max_new_tokens=args.max_new_tokens)
    elif args.generations:
        suite = load_suite(Path(args.suite))
        with open(args.generations, encoding="utf-8") as f:
            gens = json.load(f)
        report = evaluate_generations(suite, gens)
    else:
        parser.error("Provide --model or --generations")

    summary = {k: report[k] for k in ("n_items", "n_scored", "n_leaked", "leak_rate", "alert", "alert_threshold")}
    print(json.dumps(summary, indent=2))
    if report["alert"]:
        logger.error("CANARY ALERT: leak_rate=%.3f > %.3f — audit data pipeline", report["leak_rate"], CANARY_ALERT_RATE)
        for row in report.get("leaked", []):
            logger.error("  leaked %s: %s", row["id"], row["completion"])
    else:
        logger.info("Canary OK: leak_rate=%.3f", report["leak_rate"])

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote %s", args.out)

    sys.exit(1 if report["alert"] else 0)


if __name__ == "__main__":
    main()
