"""Train the 32k tokenizer for sn38-nanochrono / nanoexpand on the filtered (≤cutoff) corpus.

Byte-level BPE (GPT-2 style, no SentencePiece dummy prefix), because the SN38
leak probe tokenizes the answer separately as ``encode(" " + phrase)``. With
byte-level BPE the leading space merges into the first word ("Ġword"), exactly
as it does mid-sentence, so the probe sees the same tokens the model trained on.

Training on the filtered corpus keeps the vocabulary itself pre-cutoff.

Output is a standard HF ``tokenizer.json`` + ``tokenizer_config.json`` (no
custom classes — required by docs/miner.md), with:
  <|bos|>  prepended by add_special_tokens=True (validator generation path)
  <|eos|>  document end, also used as pad

  export SN38_DATA_ROOT=/workspace/sn38-data
  python -m train.train_tokenizer --out $SN38_DATA_ROOT/tokenizer --max-docs 5000000
  python -m train.check_tokenizer --tokenizer $SN38_DATA_ROOT/tokenizer \\
      --in $SN38_DATA_ROOT/filtered/wikipedia_2012.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import FILTERED_DIR, NANOCHRONO_CONFIG_KWARGS, SEED  # noqa: E402

logger = logging.getLogger(__name__)

VOCAB_SIZE = NANOCHRONO_CONFIG_KWARGS["vocab_size"]  # 32768
BOS, EOS = "<|bos|>", "<|eos|>"
SPECIAL_TOKENS = [BOS, EOS]

# Cleaner sources get more seats in the sample so Ġ+word merges form for probe phrases.
_SOURCE_WEIGHT = {
    "wikipedia_2012": 3,
    "news_2012": 3,
    "pes2o": 2,
    "books": 2,
    "fineweb_edu": 1,
}
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MIN_TRAIN_CHARS = 80


def _norm_text(text: str) -> str:
    text = _CTRL.sub("", text or "")
    text = text.replace("\xa0", " ").replace("\u200b", "")
    return " ".join(text.split())


def iter_texts(paths: list[Path], max_docs: int | None, seed: int = SEED) -> Iterator[str]:
    """Weighted round-robin so wiki/news dominate the vocab (probe-friendly)."""
    rng = random.Random(seed)
    files = [open(p, encoding="utf-8") for p in paths]
    weights = [_SOURCE_WEIGHT.get(p.stem, 1) for p in paths]
    # seats: each source appears `weight` times in the rotation list
    seats = []
    for i, w in enumerate(weights):
        seats.extend([i] * w)
    total_weight = sum(weights)
    per_source = None
    if max_docs is not None:
        # approx docs per unit weight
        per_source = {i: max(1, (max_docs * weights[i]) // total_weight) for i in range(len(files))}
    counts = [0] * len(files)
    try:
        live = set(range(len(files)))
        while live:
            rng.shuffle(seats)
            progressed = False
            for i in seats:
                if i not in live:
                    continue
                if per_source is not None and counts[i] >= per_source[i]:
                    live.discard(i)
                    continue
                line = files[i].readline()
                if not line:
                    live.discard(i)
                    continue
                text = _norm_text(json.loads(line).get("text") or "")
                if len(text) < _MIN_TRAIN_CHARS:
                    continue
                counts[i] += 1
                progressed = True
                yield text
            if not progressed:
                break
    finally:
        for f in files:
            f.close()
        logger.info("docs per source: %s", {p.stem: c for p, c in zip(paths, counts)})


def train(paths: list[Path], out_dir: Path, vocab_size: int = VOCAB_SIZE, max_docs: int | None = None):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    # GPT-2 regex already separates digit runs (" ?\p{N}+") while keeping the leading
    # space attached. Do NOT add a Digits pre-tokenizer: it splits " 2012" into " " + "2012",
    # so every year/number answer would start with a lone-space token in the probe.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    logger.info("Training BPE vocab_size=%d max_docs=%s on %d files", vocab_size, max_docs, len(paths))
    tok.train_from_iterator(iter_texts(paths, max_docs), trainer=trainer)
    bos_id = tok.token_to_id(BOS)
    tok.post_processor = processors.TemplateProcessing(
        single=f"{BOS} $A", pair=f"{BOS} $A {BOS} $B", special_tokens=[(BOS, bos_id)]
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=BOS,
        eos_token=EOS,
        pad_token=EOS,
        model_max_length=NANOCHRONO_CONFIG_KWARGS["max_position_embeddings"],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(out_dir)
    # transformers 5 writes its internal "TokenizersBackend" class name; validators may run
    # transformers 4.x (pyproject: >=4.48), which only knows PreTrainedTokenizerFast.
    cfg_path = out_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg["add_prefix_space"] = False
    cfg["model_max_length"] = NANOCHRONO_CONFIG_KWARGS["max_position_embeddings"]
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    info = {
        "vocab_size": len(fast),
        "bos_token_id": fast.bos_token_id,
        "eos_token_id": fast.eos_token_id,
        "pad_token_id": fast.pad_token_id,
        "sources": [p.name for p in paths],
        "max_docs": max_docs,
        "out": str(out_dir),
    }
    logger.info("%s", info)
    return info


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description="Train 32k byte-level BPE on the filtered corpus")
    parser.add_argument("--in", dest="inputs", nargs="*", default=None, help="JSONL files (default: filtered/*.jsonl)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument(
        "--max-docs",
        type=int,
        default=5_000_000,
        help="Total docs sampled across sources (0 = all). 5M recommended for low lone-space rate.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run train.check_tokenizer on wikipedia after save (exit 1 if not ok)",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs] if args.inputs else sorted(FILTERED_DIR.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No JSONL inputs (looked in {FILTERED_DIR})")
    info = train(paths, Path(args.out), args.vocab_size, args.max_docs or None)
    print(json.dumps(info, indent=2))

    if args.check:
        from train.check_tokenizer import check, load_sentences

        wiki = FILTERED_DIR / "wikipedia_2012.jsonl"
        sentences = load_sentences(wiki if wiki.exists() else None, 2000)
        report = check(str(args.out), sentences, max_lone_space=0.0)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not report["ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
