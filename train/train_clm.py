"""Phase 6 — causal LM training loop with per-sample loss weights.

Usage (smoke):
  python -m train.data.sources --smoke
  python -m train.filter.pipeline --in train/data/raw/smoke/smoke_docs.jsonl --source fineweb_edu
  python -m train.data.mix --filtered-dir train/data/filtered --out train/data/mix/train_mix.jsonl
  python -m train.train_clm --mix train/data/mix/train_mix.jsonl --steps 50 --init-new

Full runs: point --mix at the Phase 5 mix and remove --steps limit.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.config import (  # noqa: E402
    ALLOWED_ARCHITECTURES,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_EVERY_TOKENS,
    CONTEXT_LENGTH,
    GRAD_ACCUM,
    LEARNING_RATE,
    NANOEXPAND_EXTRA_MLP_AFTER,
    SEED,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    nanochrono_init_kwargs,
    resolve_architecture,
)

logger = logging.getLogger(__name__)


class JsonlClmDataset(Dataset):
    def __init__(self, path: Path, tokenizer, ctx: int = CONTEXT_LENGTH):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.ctx = ctx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = row["text"]
        weight = float(row.get("loss_weight", 1.0))
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 2:
            ids = ids + [self.tokenizer.eos_token_id or 0]
        # random crop to context
        if len(ids) > self.ctx:
            start = random.randint(0, len(ids) - self.ctx)
            ids = ids[start : start + self.ctx]
        input_ids = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": input_ids, "loss_weight": torch.tensor(weight, dtype=torch.float)}


def collate(batch, pad_id: int):
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids, weights, labels = [], [], []
    for x in batch:
        ids = x["input_ids"]
        pad = max_len - ids.size(0)
        padded = F.pad(ids, (0, pad), value=pad_id)
        lab = padded.clone()
        lab[lab == pad_id] = -100
        # shift handled inside model loss; keep full labels
        input_ids.append(padded)
        labels.append(lab)
        weights.append(x["loss_weight"])
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "loss_weight": torch.stack(weights),
    }


def _init_sn38_model(arch: str):
    """Build sn38-nanochrono/nanoexpand using top-miner chrono3-2025 config layout."""
    arch = resolve_architecture(arch)
    kwargs = nanochrono_init_kwargs()
    if arch == "nanochrono":
        from sn38.architectures.nanochrono import NanochronoConfig, NanochronoForCausalLM

        config = NanochronoConfig(**kwargs)
        model = NanochronoForCausalLM(config)
    else:
        from sn38.architectures.nanoexpand import NanoExpandConfig, NanoExpandForCausalLM

        config = NanoExpandConfig(**kwargs, extra_mlp_after=list(NANOEXPAND_EXTRA_MLP_AFTER))
        model = NanoExpandForCausalLM(config)
    n_full = sum(1 for t in config.layer_types if t == "full_attention")
    n_slide = sum(1 for t in config.layer_types if t == "sliding_attention")
    logger.info(
        "Initialized %s (model_type=%s) layers=%d full=%d sliding=%d ctx=%d",
        arch,
        config.model_type,
        config.num_hidden_layers,
        n_full,
        n_slide,
        config.max_position_embeddings,
    )
    return model, config


def build_model(
    init_new: bool,
    model_path: str | None,
    device: str,
    arch: str,
    tokenizer_path: str | None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sn38.architectures import register_all, register_for

    register_all()
    arch = resolve_architecture(arch)

    if init_new:
        model, config = _init_sn38_model(arch)
        if not tokenizer_path:
            raise SystemExit(
                "Pass --tokenizer PATH for real training (vocab_size=32768 for nanochrono/nanoexpand). "
                "Smoke-only: --tokenizer gpt2 (will warn on vocab mismatch)."
            )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer_path in {"gpt2", "openai-community/gpt2"}:
            logger.warning("gpt2 tokenizer is smoke-only — use a 32k tokenizer matching the arch")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if config.vocab_size != len(tokenizer):
            logger.warning(
                "Vocab mismatch model=%d tokenizer=%d — resize or use a matching tokenizer",
                config.vocab_size,
                len(tokenizer),
            )
    else:
        if not model_path:
            raise SystemExit("--model-path required unless --init-new")
        register_for(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        mt = getattr(model.config, "model_type", "")
        if mt not in {"sn38-nanochrono", "sn38-nanoexpand"}:
            raise SystemExit(
                f"Checkpoint model_type={mt!r} is not allowed. "
                f"Use one of {ALLOWED_ARCHITECTURES} (sn38-nanochrono / sn38-nanoexpand)."
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    return model, tokenizer


def weighted_lm_loss(logits, labels, sample_weights):
    """Token CE averaged per sample, then weighted across the batch."""
    # logits: [B, T, V], labels: [B, T]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    b, t, v = shift_logits.shape
    token_loss = F.cross_entropy(
        shift_logits.view(-1, v),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(b, t)
    mask = (shift_labels != -100).float()
    per_sample = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    w = sample_weights.to(per_sample.dtype)
    return (per_sample * w).sum() / w.sum().clamp(min=1e-8)


def save_checkpoint(model, tokenizer, step: int, tokens: int, out_dir: Path):
    path = out_dir / f"step_{step}_tok_{tokens}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    meta = {"step": step, "tokens": tokens}
    (path / "train_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Checkpoint → %s", path)
    return path


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description="Train 2012-cutoff causal LM")
    parser.add_argument("--mix", type=str, required=True, help="Training mix JSONL")
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=list(ALLOWED_ARCHITECTURES),
        help="sn38 architecture: nanochrono (default) or nanoexpand — not ChronoGPT",
    )
    parser.add_argument("--init-new", action="store_true", help="Init fresh sn38-nanochrono/nanoexpand")
    parser.add_argument("--model-path", type=str, default=None, help="Resume sn38-* checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path (required with --init-new)")
    parser.add_argument("--steps", type=int, default=None, help="Max optimizer steps (smoke)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--run-canary", action="store_true", help="Run canary after each ckpt")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    arch = resolve_architecture(args.arch)
    logger.info("device=%s arch=sn38-%s", device, arch)

    model, tokenizer = build_model(
        args.init_new, args.model_path, device, arch=arch, tokenizer_path=args.tokenizer
    )

    ds = JsonlClmDataset(Path(args.mix), tokenizer, CONTEXT_LENGTH)
    if len(ds) == 0:
        raise SystemExit(f"No records in {args.mix}")
    pad_id = tokenizer.pad_token_id or 0
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    model.train()
    step = 0
    tokens_seen = 0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    accum = 0
    optim.zero_grad(set_to_none=True)

    while True:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["loss_weight"].to(device)

            # warmup
            warm = min(1.0, (step + 1) / max(1, WARMUP_STEPS))
            for pg in optim.param_groups:
                pg["lr"] = args.lr * warm

            outputs = model(input_ids=input_ids, labels=None)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = weighted_lm_loss(logits, labels, weights) / args.grad_accum
            loss.backward()
            tokens_seen += int((labels != -100).sum().item())
            accum += 1

            if accum >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                accum = 0
                if step % 10 == 0:
                    logger.info("step=%d loss=%.4f tokens=%d lr=%.2e", step, loss.item() * args.grad_accum, tokens_seen, optim.param_groups[0]["lr"])

                if tokens_seen > 0 and tokens_seen % CHECKPOINT_EVERY_TOKENS < (args.batch_size * CONTEXT_LENGTH * args.grad_accum):
                    # rough boundary; also checkpoint on step schedule for smoke
                    pass

                if args.steps and step >= args.steps:
                    ckpt = save_checkpoint(model, tokenizer, step, tokens_seen, out_dir)
                    if args.run_canary:
                        from train.canary.evaluate import evaluate_model

                        report = evaluate_model(str(ckpt))
                        logger.info("canary leak_rate=%.3f alert=%s", report["leak_rate"], report["alert"])
                    logger.info("Done (hit --steps=%d)", args.steps)
                    return

        # end of epoch — save
        ckpt = save_checkpoint(model, tokenizer, step, tokens_seen, out_dir)
        if args.run_canary:
            from train.canary.evaluate import evaluate_model

            report = evaluate_model(str(ckpt))
            if report["alert"]:
                logger.error("Canary alert — stopping")
                return
        if args.steps:
            continue
        # infinite epochs until user stops; for full training set --steps or kill
        logger.info("Epoch complete at step=%d — continuing", step)


if __name__ == "__main__":
    main()
