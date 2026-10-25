"""Check a tokenizer against how the SN38 validator tokenizes leak probes.

sn38/template/leak.py scores an item as
    prompt_ids = [tokenizer.bos_token_id] + encode(prompt)          (no special tokens)
    phrase_ids = encode(" " + phrase)                               (no special tokens)
while the model saw the same words during training as encode(prompt + " " + phrase).
If those two tokenizations differ, the probe asks for token sequences the model
rarely or never produced, and known items fail for reasons unrelated to knowledge.

Checks:
  1. bos_token exists and add_special_tokens=True prepends it
  2. vocab fits the model (<= vocab_size) and specials match train/config ids if set
  3. tokenizer files are standard (tokenizer.json, no custom class)
  4. split-vs-joint mismatch rate on real sentences, and lone-space phrase tokens

  python -m train.check_tokenizer --tokenizer /workspace/tokenizer
  python -m train.check_tokenizer --tokenizer gpt2 --in $SN38_DATA_ROOT/filtered/wikipedia_2012.jsonl
  # perfect (zero lone-space tolerance):
  python -m train.check_tokenizer --tokenizer $TOK --in $WIKI --max-lone-space 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import NANOCHRONO_CONFIG_KWARGS  # noqa: E402

BUILTIN_SENTENCES = [
    "In November 2012, Barack Obama was re-elected President of the United States.",
    "The Higgs boson discovery was announced at CERN on 4 July 2012.",
    "The 2012 Summer Olympics were held in London, United Kingdom.",
    "Hurricane Sandy struck the east coast of the United States in October 2012.",
    "Facebook held its initial public offering on the Nasdaq in May 2012.",
    "Xi Jinping became General Secretary of the Chinese Communist Party in 2012.",
    "The Curiosity rover landed on Mars in August 2012.",
    "Felix Baumgartner jumped from the stratosphere over Roswell, New Mexico.",
    "The capital of Australia is Canberra.",
    "Apple released the iPhone 5 in September 2012.",
]
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def load_sentences(path: Path | None, limit: int) -> list[str]:
    if path is None:
        return BUILTIN_SENTENCES
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            for s in _SENT.split(json.loads(line).get("text") or ""):
                s = " ".join(s.split())
                if 40 <= len(s) <= 300:
                    out.append(s)
                    if len(out) >= limit:
                        return out
    return out


def probe_splits(sentence: str) -> list[tuple[str, str]]:
    """Validator-style (prompt, phrase) splits: last 1–3 words as the answer phrase."""
    words = sentence.split()
    return [(" ".join(words[:-k]), " ".join(words[-k:])) for k in (1, 2, 3) if len(words) > k + 2]


def check(
    tokenizer_name: str,
    sentences: list[str],
    show: int = 5,
    *,
    max_mismatch: float = 0.01,
    max_lone_space: float = 0.0,
) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    report: dict = {
        "tokenizer": tokenizer_name,
        "problems": [],
        "thresholds": {"max_mismatch": max_mismatch, "max_lone_space": max_lone_space},
    }

    # 1. BOS
    report["bos_token"], report["bos_token_id"] = tok.bos_token, tok.bos_token_id
    if tok.bos_token_id is None:
        report["problems"].append("no bos_token: validator scores prompts without BOS (ok only if training matches)")
    else:
        with_special = tok.encode("Hello world", add_special_tokens=True)
        if not with_special or with_special[0] != tok.bos_token_id:
            report["problems"].append("add_special_tokens=True does not prepend BOS (validator generation path)")

    # 2. vocab / ids
    model_vocab = NANOCHRONO_CONFIG_KWARGS["vocab_size"]
    report["vocab"] = len(tok)
    if len(tok) > model_vocab:
        report["problems"].append(f"tokenizer has {len(tok)} tokens > model vocab_size {model_vocab}")
    if tok.eos_token_id is None:
        report["problems"].append("no eos_token")

    # 3. files
    local = Path(tokenizer_name)
    if local.is_dir():
        if not (local / "tokenizer.json").exists():
            report["problems"].append("tokenizer.json missing (required by docs/miner.md)")
        cfg = local / "tokenizer_config.json"
        if cfg.exists():
            cls = json.loads(cfg.read_text(encoding="utf-8")).get("tokenizer_class", "")
            if cls and cls not in {"PreTrainedTokenizerFast", "GPT2TokenizerFast", "LlamaTokenizerFast"}:
                report["problems"].append(f"non-standard tokenizer_class {cls!r}")

    # 4. split vs joint
    n = mismatched = lone_space = 0
    examples = []
    lone_examples = []
    for sent in sentences:
        for prompt, phrase in probe_splits(sent):
            n += 1
            split_ids = tok.encode(prompt, add_special_tokens=False) + tok.encode(" " + phrase, add_special_tokens=False)
            joint_ids = tok.encode(prompt + " " + phrase, add_special_tokens=False)
            phrase_ids = tok.encode(" " + phrase, add_special_tokens=False)
            if phrase_ids and tok.decode(phrase_ids[:1]).strip() == "":
                lone_space += 1
                if len(lone_examples) < show:
                    lone_examples.append(
                        {
                            "phrase": phrase,
                            "tokens": tok.convert_ids_to_tokens(phrase_ids[:4]),
                        }
                    )
            if split_ids != joint_ids:
                mismatched += 1
                if len(examples) < show:
                    examples.append(
                        {
                            "prompt": prompt[-40:],
                            "phrase": phrase,
                            "split": tok.convert_ids_to_tokens(split_ids[-6:]),
                            "joint": tok.convert_ids_to_tokens(joint_ids[-6:]),
                        }
                    )
    report["probes"] = n
    report["split_vs_joint_mismatch"] = round(mismatched / max(1, n), 4)
    report["lone_space_phrase_start"] = round(lone_space / max(1, n), 4)
    report["mismatch_examples"] = examples
    report["lone_space_examples"] = lone_examples
    if report["split_vs_joint_mismatch"] > max_mismatch:
        report["problems"].append(
            f"{report['split_vs_joint_mismatch']:.1%} of probes tokenize differently split vs joint "
            f"(max {max_mismatch:.1%})"
        )
    if report["lone_space_phrase_start"] > max_lone_space:
        report["problems"].append(
            f"{report['lone_space_phrase_start']:.1%} of phrases start with a lone space token "
            f"(SentencePiece-style prefix; max {max_lone_space:.1%})"
        )
    report["ok"] = not report["problems"]
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate a tokenizer against SN38 leak-probe tokenization")
    parser.add_argument("--tokenizer", required=True, help="Local dir or HF id")
    parser.add_argument("--in", dest="in_path", default=None, help="JSONL with real text (default: built-in sentences)")
    parser.add_argument("--limit", type=int, default=2000, help="Max sentences from --in")
    parser.add_argument(
        "--max-mismatch",
        type=float,
        default=0.01,
        help="Fail if split-vs-joint mismatch rate exceeds this (default 1%)",
    )
    parser.add_argument(
        "--max-lone-space",
        type=float,
        default=0.0,
        help="Fail if lone-space phrase-start rate exceeds this (default 0 = perfect)",
    )
    args = parser.parse_args()
    report = check(
        args.tokenizer,
        load_sentences(Path(args.in_path) if args.in_path else None, args.limit),
        max_mismatch=args.max_mismatch,
        max_lone_space=args.max_lone_space,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
