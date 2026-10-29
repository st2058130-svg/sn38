"""Phase 6 — causal LM pretraining loop for sn38-nanochrono / nanoexpand.

Features:
  * sequence packing (documents joined with EOS into full CONTEXT_LENGTH windows)
  * per-token loss weights carried from each mix row's ``loss_weight``
  * optional --mask-post-cutoff-years: labels=-100 on post-cutoff year tokens
  * bf16 autocast, TF32, fused AdamW (no decay on norms/biases/embeddings)
  * warmup → cosine or WSD (warmup-stable-decay) LR schedule over --total-tokens
  * multi-GPU via torchrun (DDP), optional bf16 gradient all-reduce (--bf16-grad-sync)
  * throughput log: tok/s total and per GPU, MFU vs bf16 peak, data-wait share;
    --profile N splits a step into data / fwd+bwd / all-reduce / optimizer
  * checkpoints every CHECKPOINT_EVERY_TOKENS with optimizer state, --resume support
  * optional held-out val loss at checkpoints (canary: python -m train.canary.evaluate --model PATH)

Usage (smoke, CPU, tiny model):
  python -m train.run_smoke --train --steps 20

Full run (8 GPUs):
  torchrun --nproc_per_node 8 -m train.train_clm \\
      --mix /workspace/sn38-data/mix/train_mix.jsonl --val-mix /workspace/sn38-data/mix/val_mix.jsonl \\
      --init-new --tokenizer /workspace/tokenizer --total-tokens 35e9

Resume:
  torchrun ... -m train.train_clm --mix ... --resume train/checkpoints/step_12000_tok_...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

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


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def dist_info() -> tuple[int, int, int]:
    """(rank, local_rank, world_size); (0, 0, 1) when not launched by torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", 0)), int(os.environ["WORLD_SIZE"])
    return 0, 0, 1


def is_main() -> bool:
    return dist_info()[0] == 0


# ---------------------------------------------------------------------------
# Post-cutoff year loss masking (labels → -100; uses existing ignore_index path)
# ---------------------------------------------------------------------------
def build_post_cutoff_year_token_seqs(tokenizer, cutoff_year: int, end_year: int = 2026) -> list[list[int]]:
    """Token-id sequences for calendar years (cutoff_year+1) … end_year.

    Encodes both ``YYYY`` and `` YYYY`` so leading-space BPE pieces are covered.
    """
    seqs: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for y in range(int(cutoff_year) + 1, int(end_year) + 1):
        for text in (str(y), f" {y}"):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            key = tuple(ids)
            if key not in seen:
                seen.add(key)
                seqs.append(list(ids))
    # Longer sequences first so multi-token years match before shorter pieces
    seqs.sort(key=len, reverse=True)
    return seqs


def mask_labels_for_post_cutoff_years(
    ids: list[int],
    labels: torch.Tensor,
    year_seqs: list[list[int]],
    window_after: int = 0,
) -> torch.Tensor:
    """Set labels to -100 where the predicted token is (or follows) a post-cutoff year.

    ``labels[j]`` is the prediction target for ``ids[j+1]``. Masking year span
    ``ids[s:e]`` clears ``labels[s-1 : e-1]`` (and optionally ``window_after`` more).
    """
    if not year_seqs or labels.numel() == 0:
        return labels
    out = labels.clone()
    n = len(ids)
    for needle in year_seqs:
        m = len(needle)
        if m == 0 or m > n:
            continue
        for s in range(0, n - m + 1):
            if ids[s : s + m] != needle:
                continue
            # Predict ids[k] → labels index k-1
            lo = max(0, s - 1)
            hi = min(out.numel(), (s + m - 1) + max(0, int(window_after)))
            if lo < hi:
                out[lo:hi] = -100
    return out


# ---------------------------------------------------------------------------
# Data: packed, weighted, sharded stream
# ---------------------------------------------------------------------------
class PackedJsonlDataset(IterableDataset):
    """Stream a mix JSONL and yield packed windows of ``ctx + 1`` tokens.

    Each document is tokenized, terminated with EOS and appended to a rolling
    buffer; full windows are emitted together with per-token loss weights
    (the document's ``loss_weight``). Rows are sharded across DDP ranks and
    DataLoader workers by line index. A shuffle buffer decorrelates windows.

    If ``mask_cutoff_year`` is set, label positions that predict post-cutoff
    calendar years (and optional ``mask_year_window`` tokens after) are set to
    -100 so ``weighted_lm_loss`` ignores them.
    """

    def __init__(
        self,
        path: Path,
        tokenizer,
        ctx: int = CONTEXT_LENGTH,
        rank: int = 0,
        world_size: int = 1,
        seed: int = SEED,
        shuffle_buffer: int = 2048,
        loop: bool = True,
        mask_cutoff_year: int | None = None,
        mask_year_window: int = 0,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.ctx = ctx
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.loop = loop
        self.mask_cutoff_year = mask_cutoff_year
        self.mask_year_window = max(0, int(mask_year_window))
        self.eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        # SN38 leak probes are scored as [BOS] + prompt (sn38/template/leak.py, using
        # tokenizer.bos_token_id), so every document starts with BOS here too.
        self.bos = tokenizer.bos_token_id
        self.epoch = 0
        self._ids_buf: list[int] = []
        self._w_buf: list[float] = []
        # Silence HF "sequence longer than max_length" spam; we pack to ctx ourselves.
        try:
            tokenizer.model_max_length = int(1e9)
        except Exception:
            pass
        # Rust batch encoder (much faster than per-doc tokenizer.encode on long docs)
        self._backend = getattr(tokenizer, "backend_tokenizer", None)
        self._encode_batch_size = int(os.environ.get("SN38_ENCODE_BATCH", "64"))
        self._year_seqs: list[list[int]] = []
        if mask_cutoff_year is not None:
            self._year_seqs = build_post_cutoff_year_token_seqs(tokenizer, mask_cutoff_year)
            logger.info(
                "Year loss-mask on: cutoff=%d → mask years %d..2026 (%d token patterns, window_after=%d)",
                mask_cutoff_year,
                int(mask_cutoff_year) + 1,
                len(self._year_seqs),
                self.mask_year_window,
            )

    def _rows(self, shard: int, n_shards: int):
        with open(self.path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % n_shards != shard:
                    continue
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("text") or ""
                if text:
                    yield text, float(row.get("loss_weight", 1.0))

    def _encode_many(self, texts: list[str]) -> list[list[int]]:
        if not texts:
            return []
        if self._backend is not None:
            # Must disable specials — we prepend BOS / append EOS ourselves for packing.
            return [e.ids for e in self._backend.encode_batch(texts, add_special_tokens=False)]
        return [self.tokenizer.encode(t, add_special_tokens=False) for t in texts]

    def _windows(self, shard: int, n_shards: int):
        # buffers persist across epochs so tiny mixes still fill a window
        n = self.ctx + 1
        n_rows = 0
        batch_texts: list[str] = []
        batch_weights: list[float] = []

        def flush():
            nonlocal n_rows
            if not batch_texts:
                return
            for ids, weight in zip(self._encode_many(batch_texts), batch_weights):
                n_rows += 1
                ids = list(ids) + [self.eos]
                if self.bos is not None:
                    ids = [self.bos] + ids
                self._ids_buf.extend(ids)
                self._w_buf.extend([weight] * len(ids))
            batch_texts.clear()
            batch_weights.clear()

        for text, weight in self._rows(shard, n_shards):
            batch_texts.append(text)
            batch_weights.append(weight)
            if len(batch_texts) >= self._encode_batch_size:
                flush()
                while len(self._ids_buf) >= n:
                    yield self._ids_buf[:n], self._w_buf[:n]
                    # overlap by one token so every token is predicted exactly once
                    self._ids_buf = self._ids_buf[n - 1 :]
                    self._w_buf = self._w_buf[n - 1 :]
        flush()
        while len(self._ids_buf) >= n:
            yield self._ids_buf[:n], self._w_buf[:n]
            self._ids_buf = self._ids_buf[n - 1 :]
            self._w_buf = self._w_buf[n - 1 :]
        if n_rows == 0:
            raise RuntimeError(f"No rows for shard {shard}/{n_shards} in {self.path}")

    def __iter__(self):
        info = get_worker_info()
        wid, nworkers = (info.id, info.num_workers) if info else (0, 1)
        shard = self.rank * nworkers + wid
        n_shards = self.world_size * nworkers
        while True:
            rng = random.Random(self.seed + 1000 * self.epoch + shard)
            buf: list = []
            for item in self._windows(shard, n_shards):
                buf.append(item)
                if len(buf) >= self.shuffle_buffer:
                    yield self._to_tensors(buf.pop(rng.randrange(len(buf))))
            rng.shuffle(buf)
            for item in buf:
                yield self._to_tensors(item)
            self.epoch += 1
            if not self.loop:
                return

    def _to_tensors(self, item):
        ids, w = item
        ids_t = torch.tensor(ids, dtype=torch.long)
        w_t = torch.tensor(w, dtype=torch.float)
        # predict token t+1 from tokens <= t; the weight belongs to the target token
        labels = ids_t[1:].clone()
        if self._year_seqs:
            labels = mask_labels_for_post_cutoff_years(
                ids, labels, self._year_seqs, window_after=self.mask_year_window
            )
        return {"input_ids": ids_t[:-1], "labels": labels, "loss_weight": w_t[1:]}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _init_sn38_model(arch: str, tiny: bool = False, vocab_size: int | None = None):
    """Build sn38-nanochrono/nanoexpand using top-miner chrono3-2025 config layout."""
    arch = resolve_architecture(arch)
    kwargs = nanochrono_init_kwargs()
    if vocab_size:
        kwargs["vocab_size"] = max(vocab_size, kwargs["vocab_size"])
    if tiny:
        # smoke-only: same layout, toy width/depth so CPU runs finish in seconds
        kwargs.update(
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            layer_types=kwargs["layer_types"][:4],
            aux_layers=[1, 3],
            tap_layer=2,
        )
    if arch == "nanochrono":
        from sn38.architectures.nanochrono import NanochronoConfig, NanochronoForCausalLM

        config = NanochronoConfig(**kwargs)
        config.use_cache = False  # training; cache is for generate()
        model = NanochronoForCausalLM(config)
    else:
        from sn38.architectures.nanoexpand import NanoExpandConfig, NanoExpandForCausalLM

        extra = [] if tiny else list(NANOEXPAND_EXTRA_MLP_AFTER)
        config = NanoExpandConfig(**kwargs, extra_mlp_after=extra)
        config.use_cache = False
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
    tiny: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sn38.architectures import register_all, register_for

    register_all()
    arch = resolve_architecture(arch)

    if init_new:
        if not tokenizer_path:
            raise SystemExit(
                "Pass --tokenizer PATH for real training (vocab_size=32768 for nanochrono/nanoexpand). "
                "Smoke-only: --tokenizer gpt2 with --tiny."
            )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.bos_token_id is None:
            logger.warning("Tokenizer has no bos_token — validator will score prompts without BOS; training matches that")
        model, config = _init_sn38_model(arch, tiny=tiny, vocab_size=len(tokenizer) if tiny else None)
        # special ids in config.json must match the tokenizer the validator loads
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            tok_id = getattr(tokenizer, name)
            if tok_id is not None and getattr(config, name, None) != tok_id:
                logger.info("config.%s %s → %s (from tokenizer)", name, getattr(config, name, None), tok_id)
                setattr(config, name, tok_id)
        if tokenizer_path in {"gpt2", "openai-community/gpt2"}:
            logger.warning("gpt2 tokenizer is smoke-only — use a 32k tokenizer matching the arch")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if config.vocab_size < len(tokenizer):
            raise SystemExit(
                f"Tokenizer has {len(tokenizer)} tokens but model vocab_size={config.vocab_size} — "
                "token ids would be out of range"
            )
        if config.vocab_size != len(tokenizer):
            logger.warning("Vocab mismatch model=%d tokenizer=%d", config.vocab_size, len(tokenizer))
    else:
        if not model_path:
            raise SystemExit("--model-path required unless --init-new")
        register_for(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
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


def weighted_lm_loss(logits, labels, token_weights):
    """Per-token CE scaled by token weight, normalised by token count.

    Dividing by the token count (not the weight sum) keeps ``loss_weight > 1``
    a real up-weighting instead of cancelling out in the normalisation.
    Labels are already shifted (see PackedJsonlDataset).
    """
    v = logits.size(-1)
    token_loss = F.cross_entropy(logits.float().view(-1, v), labels.view(-1), ignore_index=-100, reduction="none")
    mask = (labels.view(-1) != -100).float()
    w = token_weights.view(-1).float() * mask
    return (token_loss * w).sum() / mask.sum().clamp(min=1.0), (token_loss * mask).sum() / mask.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Optimizer / schedule
# ---------------------------------------------------------------------------
def build_optimizer(model, lr: float, weight_decay: float, device: str):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # norms, biases, gates and embeddings are not decayed
        if p.ndim < 2 or "embed" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = device.startswith("cuda")
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=fused)


def lr_at(step: int, total_steps: int, peak: float, min_lr: float, warmup: int, schedule: str, decay_frac: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    if schedule == "constant":
        return peak
    if schedule == "wsd":
        decay_start = int(total_steps * (1.0 - decay_frac))
        if step < decay_start:
            return peak
        prog = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
        return min_lr + (peak - min_lr) * (1.0 - prog)
    # cosine
    prog = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * prog))


# ---------------------------------------------------------------------------
# Throughput accounting
# ---------------------------------------------------------------------------
# Dense bf16 peak per GPU (TFLOPs). Override with SN38_PEAK_TFLOPS for other cards.
# Match torch.cuda.get_device_name() substrings (longest/most specific keys first via iteration order — keep unique).
_PEAK_BF16_TFLOPS = {
    "B300": 2250.0,  # Blackwell Ultra dense BF16 (~4500 sparse)
    "B200": 1125.0,  # Blackwell dense BF16
    "H100": 989.0,
    "H200": 989.0,
    "H800": 989.0,
    "A100": 312.0,
    "A800": 312.0,
    "L40S": 362.0,
}


def peak_tflops(device: str) -> float | None:
    if os.environ.get("SN38_PEAK_TFLOPS"):
        return float(os.environ["SN38_PEAK_TFLOPS"])
    if not device.startswith("cuda"):
        return None
    name = torch.cuda.get_device_name(torch.device(device))
    for key, val in _PEAK_BF16_TFLOPS.items():
        if key in name:
            return val
    return None


def train_flops_per_token(model, ctx: int) -> tuple[float, int]:
    """(training FLOPs per token, matmul params).

    Only nn.Linear weights do matmul work; the embedding tables (including the
    14 aux_embeds, ~40% of nanochrono's params) are lookups and are excluded.
    Attention adds 12 · hidden · avg_keys per layer (fwd QK^T + AV, ×3 for bwd).
    """
    matmul = sum(m.weight.numel() for m in model.modules() if isinstance(m, torch.nn.Linear))
    cfg = model.config
    attn = 0.0
    for t in getattr(cfg, "layer_types", ["full_attention"] * cfg.num_hidden_layers):
        w = cfg.sliding_window if t == "sliding_attention" else ctx
        avg_keys = sum(min(i + 1, w + 1) for i in range(ctx)) / ctx
        attn += 12 * cfg.hidden_size * avg_keys
    return 6.0 * matmul + attn, matmul


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def _safetensors_gb(path: Path) -> float:
    f = path / "model.safetensors"
    return f.stat().st_size / (1024**3) if f.is_file() else float("nan")


def save_checkpoint(model, tokenizer, optim, step: int, tokens: int, out_dir: Path, args) -> Path:
    """Save fp32 train checkpoint + bf16 export for HF submit (<8GB).

    Layout:
      out_dir/step_{step}_tok_{tokens}/       — fp32 weights + optimizer (use with --resume)
      out_dir/step_{step}_tok_{tokens}_bf16/  — bf16 weights only (upload this; ~4GB)
    Returns the fp32 path (resume). Canary should load the ``_bf16`` sibling.
    """
    path = out_dir / f"step_{step}_tok_{tokens}"
    path.mkdir(parents=True, exist_ok=True)
    # Ensure train weights are fp32 before writing the resume checkpoint.
    if next(model.parameters()).dtype != torch.float32:
        model.float()
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    torch.save(
        {
            "optimizer": optim.state_dict(),
            "step": step,
            "tokens": tokens,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
        },
        path / "training_state.pt",
    )
    meta = {"step": step, "tokens": tokens, "dtype": "float32", "args": vars(args)}
    (path / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    logger.info("Checkpoint (fp32) → %s (model.safetensors=%.2f GiB)", path, _safetensors_gb(path))

    # BF16 export on CPU — do NOT cast the live training model (DDP desync / NCCL risk).
    bf16_path = out_dir / f"step_{step}_tok_{tokens}_bf16"
    bf16_path.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(bf16_path)
    tokenizer.save_pretrained(bf16_path)
    # Copy sidecar files HF needs alongside safetensors (already written to fp32 path).
    for name in ("generation_config.json", "configuration_nanochrono.py", "modeling_nanochrono.py",
                 "configuration_nanoexpand.py", "modeling_nanoexpand.py"):
        src = path / name
        if src.is_file():
            (bf16_path / name).write_bytes(src.read_bytes())
    sd = {
        k: (v.detach().to("cpu", dtype=torch.bfloat16) if v.is_floating_point() else v.detach().cpu())
        for k, v in model.state_dict().items()
    }
    from safetensors.torch import save_file

    save_file(sd, bf16_path / "model.safetensors")
    meta_bf16 = {"step": step, "tokens": tokens, "dtype": "bfloat16", "fp32_checkpoint": str(path), "args": vars(args)}
    (bf16_path / "train_meta.json").write_text(json.dumps(meta_bf16, indent=2, default=str))
    gb = _safetensors_gb(bf16_path)
    logger.info("Export (bf16) → %s (model.safetensors=%.2f GiB)%s", bf16_path, gb, " ⚠ ≥8 GiB" if gb >= 8.0 else "")

    return path


@torch.no_grad()
def evaluate_val(model, loader, device, amp_ctx, max_batches: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        with amp_ctx():
            logits = model(input_ids=batch["input_ids"].to(device), use_cache=False).logits
        _, plain = weighted_lm_loss(logits, batch["labels"].to(device), batch["loss_weight"].to(device))
        total += plain.item()
        n += 1
    model.train()
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description="Train 2012-cutoff causal LM")
    parser.add_argument("--mix", type=str, required=True, help="Training mix JSONL")
    parser.add_argument("--val-mix", type=str, default=None, help="Held-out JSONL for val loss at checkpoints")
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        choices=list(ALLOWED_ARCHITECTURES),
        help="sn38 architecture: nanochrono (default) or nanoexpand — not ChronoGPT",
    )
    parser.add_argument("--init-new", action="store_true", help="Init fresh sn38-nanochrono/nanoexpand")
    parser.add_argument("--tiny", action="store_true", help="Smoke-only toy-size model")
    parser.add_argument("--model-path", type=str, default=None, help="Start from sn38-* HF checkpoint (weights only)")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint dir (weights + optimizer + step)")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path (required with --init-new)")
    parser.add_argument("--total-tokens", type=float, default=None, help="Training budget in tokens (sets schedule length)")
    parser.add_argument("--steps", type=int, default=None, help="Max optimizer steps (overrides --total-tokens)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Micro-batch sequences per GPU")
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--schedule", choices=["cosine", "wsd", "constant"], default="cosine")
    parser.add_argument("--decay-frac", type=float, default=0.2, help="WSD: fraction of steps spent decaying")
    parser.add_argument("--checkpoint-every-tokens", type=float, default=CHECKPOINT_EVERY_TOKENS)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SN38_NUM_WORKERS", "4")),
        help="DataLoader workers per rank (tokenization is CPU-heavy; 4–8 helps)",
    )
    parser.add_argument("--compile", action="store_true", help="torch.compile the model (fuses many small ops; slow first steps)")
    parser.add_argument(
        "--bf16-grad-sync",
        action="store_true",
        help="DDP: all-reduce gradients in bf16 (half the traffic; helps a lot on PCIe-only GPUs)",
    )
    parser.add_argument(
        "--profile",
        type=int,
        default=0,
        help="Time the first N steps with GPU syncs and log data / compute / all-reduce / optimizer split",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument(
        "--mask-post-cutoff-years",
        action="store_true",
        help="Set labels to -100 on post-cutoff calendar year tokens (and optional window after)",
    )
    parser.add_argument(
        "--mask-cutoff-year",
        type=int,
        default=2011,
        help="With --mask-post-cutoff-years: mask years (this+1)..2026 (default 2011 → mask 2012+)",
    )
    parser.add_argument(
        "--mask-year-window",
        type=int,
        default=0,
        help="Also mask this many label positions after each post-cutoff year span (default 0)",
    )
    args = parser.parse_args()

    rank, local_rank, world = dist_info()
    ddp = world > 1
    if ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
    if not is_main():
        logging.getLogger().setLevel(logging.WARNING)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # B200/B300: cuDNN SDPA often errors ("No valid execution plans"); prefer flash/mem_efficient/math.
    if device.startswith("cuda"):
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass

    random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    arch = resolve_architecture(args.arch)
    logger.info("device=%s arch=sn38-%s world=%d", device, arch, world)

    if args.resume:
        model, tokenizer = build_model(False, args.resume, device, arch=arch, tokenizer_path=args.tokenizer)
    else:
        model, tokenizer = build_model(
            args.init_new, args.model_path, device, arch=arch, tokenizer_path=args.tokenizer, tiny=args.tiny
        )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("params=%.2fB", n_params / 1e9)

    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    amp_ctx = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext
    )

    optim = build_optimizer(model, args.lr, WEIGHT_DECAY, device)
    step, tokens_seen = 0, 0
    if args.resume:
        state = torch.load(Path(args.resume) / "training_state.pt", map_location=device, weights_only=False)
        optim.load_state_dict(state["optimizer"])
        step, tokens_seen = state["step"], state["tokens"]
        logger.info("Resumed step=%d tokens=%d", step, tokens_seen)

    raw_model = model
    flops_per_token, matmul_params = train_flops_per_token(raw_model, CONTEXT_LENGTH)
    peak = peak_tflops(device)
    logger.info(
        "matmul params=%.2fB (embeddings excluded) → %.2f GFLOP/token; GPU bf16 peak=%s TFLOPs",
        matmul_params / 1e9, flops_per_token / 1e9, peak,
    )

    ddp_model = None
    if ddp:
        # DDP first, then compile: lets TorchDynamo split graphs at DDP bucket boundaries
        # so gradient all-reduce overlaps with backward.
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.startswith("cuda") else None
        )
        if args.bf16_grad_sync:
            if dist.get_backend() == "nccl":
                from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

                ddp_model.register_comm_hook(None, default_hooks.bf16_compress_hook)
                logger.info("DDP gradient all-reduce in bf16")
            else:
                logger.warning("--bf16-grad-sync needs NCCL (CUDA); ignored on backend %s", dist.get_backend())
        model = ddp_model
    if args.compile:
        model = torch.compile(model)

    mask_year = args.mask_cutoff_year if args.mask_post_cutoff_years else None
    ds = PackedJsonlDataset(
        Path(args.mix),
        tokenizer,
        CONTEXT_LENGTH,
        rank=rank,
        world_size=world,
        seed=SEED,
        mask_cutoff_year=mask_year,
        mask_year_window=args.mask_year_window,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_loader = None
    if args.val_mix:
        val_ds = PackedJsonlDataset(
            Path(args.val_mix),
            tokenizer,
            CONTEXT_LENGTH,
            rank=rank,
            world_size=world,
            loop=False,
            mask_cutoff_year=mask_year,
            mask_year_window=args.mask_year_window,
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    tokens_per_step = args.batch_size * CONTEXT_LENGTH * args.grad_accum * world
    if args.steps:
        total_steps = args.steps
    elif args.total_tokens:
        total_steps = int(args.total_tokens // tokens_per_step)
    else:
        raise SystemExit("Pass --total-tokens (full run) or --steps (smoke) so the LR schedule has an end")
    warmup = min(args.warmup_steps, max(1, total_steps // 10))
    min_lr = args.lr * args.min_lr_ratio
    ckpt_every_steps = max(1, int(args.checkpoint_every_tokens // tokens_per_step))
    logger.info(
        "tokens/step=%d total_steps=%d (≈%.2fB tokens) warmup=%d schedule=%s ckpt_every=%d steps bf16=%s",
        tokens_per_step, total_steps, total_steps * tokens_per_step / 1e9, warmup, args.schedule, ckpt_every_steps, use_bf16,
    )

    # fast-forward the data stream on resume so we don't retrain the same windows
    data_iter = iter(loader)
    if step:
        skip = step * args.grad_accum
        logger.info(
            "Skipping %d micro-batches to resume data position (can take several minutes; progress below)...",
            skip,
        )
        t_skip = time.time()
        for i in range(skip):
            next(data_iter)
            if is_main() and (i + 1) % max(1, skip // 10) == 0:
                logger.info(
                    "resume skip %d/%d (%.0f%%) elapsed=%.0fs",
                    i + 1, skip, 100.0 * (i + 1) / skip, time.time() - t_skip,
                )
        logger.info("Resume data skip done in %.0fs — starting step=%d", time.time() - t_skip, step)

    out_dir = Path(args.out)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    optim.zero_grad(set_to_none=True)
    t0 = time.time()
    last_log_tokens = tokens_seen
    data_wait = 0.0
    on_cuda = device.startswith("cuda")
    prof = {"data": 0.0, "micro_nosync": 0.0, "n_nosync": 0, "micro_sync": 0.0, "n_sync": 0, "optim": 0.0, "steps": 0}
    run_steps = 0  # steps in this process; profiling skips the first 3 (compile / allocator warmup)

    def gpu_time() -> float:
        if on_cuda:
            torch.cuda.synchronize()
        return time.time()

    while step < total_steps:
        lr = lr_at(step, total_steps, args.lr, min_lr, warmup, args.schedule, args.decay_frac)
        for pg in optim.param_groups:
            pg["lr"] = lr
        profiling = run_steps >= 3 and prof["steps"] < args.profile

        # losses stay on the GPU; .item() only when logging (each .item() is a full GPU sync)
        loss_acc = torch.zeros((), device=device)
        plain_acc = torch.zeros((), device=device)
        for micro in range(args.grad_accum):
            td = time.time()
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            weights = batch["loss_weight"].to(device, non_blocking=True)
            data_wait += time.time() - td
            sync = (not ddp) or micro == args.grad_accum - 1
            ctx = contextlib.nullcontext() if sync else ddp_model.no_sync()
            tm = gpu_time() if profiling else 0.0
            if profiling:
                prof["data"] += tm - td
            with ctx:
                with amp_ctx():
                    # use_cache=False: KV cache is for generation; enabling it in training
                    # blows VRAM and forces slow attention — main cause of ~30k tok/s + OOM.
                    logits = model(input_ids=input_ids, use_cache=False).logits
                loss, plain = weighted_lm_loss(logits, labels, weights)
                (loss / args.grad_accum).backward()
            loss_acc += loss.detach() / args.grad_accum
            plain_acc += plain.detach() / args.grad_accum
            if profiling:
                key = "micro_sync" if (ddp and sync) else "micro_nosync"
                prof[key] += gpu_time() - tm
                prof["n_sync" if key == "micro_sync" else "n_nosync"] += 1

        to = gpu_time() if profiling else 0.0
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        if profiling:
            prof["optim"] += gpu_time() - to
            prof["steps"] += 1
            if prof["steps"] == args.profile:
                n = prof["steps"]
                compute = prof["micro_nosync"] / max(1, prof["n_nosync"])
                synced = prof["micro_sync"] / max(1, prof["n_sync"])
                step_t = (prof["data"] + prof["micro_nosync"] + prof["micro_sync"] + prof["optim"]) / n
                if prof["n_sync"] and prof["n_nosync"]:
                    allreduce = max(0.0, synced - compute)
                    ar = f"{allreduce:.3f}s ({100 * allreduce / max(step_t, 1e-9):.0f}% of step)"
                else:
                    ar = "n/a (needs DDP and --grad-accum >= 2)"
                logger.info(
                    "PROFILE (%d steps, per step): total=%.2fs | data wait=%.3fs | fwd+bwd per micro-batch=%.3fs | "
                    "extra on synced micro-batch (≈all-reduce)=%s | clip+optimizer=%.3fs",
                    n, step_t, prof["data"] / n, compute, ar, prof["optim"] / n,
                )
        step += 1
        run_steps += 1
        tokens_seen += tokens_per_step

        if step % 10 == 0 or step == total_steps:
            dt = time.time() - t0
            tps = (tokens_seen - last_log_tokens) / max(dt, 1e-6)
            mfu = (
                f" mfu={100 * tps * flops_per_token / (world * peak * 1e12):.1f}%"
                if peak
                else ""
            )
            logger.info(
                "step=%d/%d loss=%.4f (unweighted %.4f) gnorm=%.2f lr=%.2e tokens=%.3fB tok/s=%.0f "
                "(%.0f/GPU)%s data_wait=%.0f%%",
                step, total_steps, loss_acc.item(), plain_acc.item(), float(grad_norm), lr, tokens_seen / 1e9,
                tps, tps / world, mfu, 100 * data_wait / max(dt, 1e-6),
            )
            t0, last_log_tokens, data_wait = time.time(), tokens_seen, 0.0

        if step % ckpt_every_steps == 0 or step == total_steps:
            # Keep ranks aligned around slow ckpt I/O (avoids NCCL allreduce timeouts).
            if ddp:
                dist.barrier()
            if val_loader is not None:
                val = evaluate_val(model, val_loader, device, amp_ctx, args.val_batches)
                logger.info("val_loss=%.4f step=%d", val, step)
            if is_main():
                save_checkpoint(raw_model, tokenizer, optim, step, tokens_seen, out_dir, args)
            if ddp:
                dist.barrier()

    logger.info("Done at step=%d tokens=%d", step, tokens_seen)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
