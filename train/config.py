"""Training path config for the 2012-cutoff ChronoLLM model.

Aligned with train_plan.txt and SN38 validator constraints
(max 2.2B params, safetensors, trust_remote_code=False).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Temporal boundary (Phase 0.1)
# ---------------------------------------------------------------------------
CUTOFF_YEAR = 2012
CUTOFF = datetime(2012, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
CUTOFF_ISO = "2012-12-31T23:59:59Z"

# SN38 on-chain years are 2013–2024. This pipeline trains knowledge through
# end of 2012 (foundation vintage). Map to a submission year after continue-train.
SN38_SUBMISSION_YEARS = list(range(2013, 2025))

# ---------------------------------------------------------------------------
# Paths
# On RunPod, point SN38_DATA_ROOT at a network volume, e.g. /workspace/sn38-data
# ---------------------------------------------------------------------------
TRAIN_ROOT = Path(__file__).resolve().parent
_DATA_OVERRIDE = os.environ.get("SN38_DATA_ROOT") or os.environ.get("DATA_ROOT")
DATA_ROOT = Path(_DATA_OVERRIDE).expanduser().resolve() if _DATA_OVERRIDE else (TRAIN_ROOT / "data")
RAW_DIR = DATA_ROOT / "raw"
FILTERED_DIR = DATA_ROOT / "filtered"
# After Phase-1 date filter: FineWeb-edu classifier pass lands here (trainable mix input)
EDU_DIR = DATA_ROOT / "filtered_edu"
QA_DIR = DATA_ROOT / "qa"
MIX_DIR = DATA_ROOT / "mix"
CHECKPOINT_DIR = Path(os.environ.get("SN38_CHECKPOINT_DIR", TRAIN_ROOT / "checkpoints")).expanduser()
EXPORT_DIR = Path(os.environ.get("SN38_EXPORT_DIR", TRAIN_ROOT / "export")).expanduser()
CANARY_PATH = TRAIN_ROOT / "canary" / "suite.json"
STATE_DIR = DATA_ROOT / "state"
# Immutable corpus snapshots (reuse later years by re-filtering, e.g. 2012 → 2011)
SNAPSHOT_DIR = DATA_ROOT / "snapshots"

# ---------------------------------------------------------------------------
# Model (Phase 6.1) — ONLY sn38 architectures (not ChronoGPT / external arch)
#   nanochrono  — default for a single 2012 vintage (~2.02B params)
#   nanoexpand  — nanochrono + optional extra MLP blocks (later-year capacity)
# ---------------------------------------------------------------------------
ALLOWED_ARCHITECTURES = ("nanochrono", "nanoexpand")
ALLOWED_MODEL_TYPES = ("sn38-nanochrono", "sn38-nanoexpand")

ARCHITECTURE = "nanochrono"  # change to "nanoexpand" or pass --arch
MODEL_TYPE = f"sn38-{ARCHITECTURE}"
MAX_PARAMETERS = 2_200_000_000

# Matched to top miner anacoluthe89/chrono3-2025 (sn38-nanochrono).
# Critical delta vs library default: hybrid sliding/full layer_types
# (default NanochronoConfig uses all full_attention).
NUM_HIDDEN_LAYERS = 28


def _hybrid_layer_types(n: int = NUM_HIDDEN_LAYERS) -> list[str]:
    """3× sliding_attention + 1× full_attention, repeated (top-miner pattern)."""
    pattern = [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    return [pattern[i % 4] for i in range(n)]


def _odd_aux_layers(n: int = NUM_HIDDEN_LAYERS) -> list[int]:
    return list(range(1, n, 2))


# Exact hyperparams from anacoluthe89/chrono3-2025 config.json
NANOCHRONO_CONFIG_KWARGS: dict = {
    "model_type": "sn38-nanochrono",
    "architectures": ["NanochronoForCausalLM"],
    "vocab_size": 32768,
    "hidden_size": 1792,
    "intermediate_size": 7168,
    "num_hidden_layers": NUM_HIDDEN_LAYERS,
    "num_attention_heads": 14,
    "num_key_value_heads": 14,
    "max_position_embeddings": 2048,
    "rope_theta": 100000.0,
    "sliding_window": 512,
    "layer_types": _hybrid_layer_types(),
    "attention_multiplier": 1.2,
    "final_logit_softcapping": 15.0,
    "aux_layers": _odd_aux_layers(),
    "aux_gate_dim": 12,
    "mix_gate_dim": 24,
    "tap_layer": 14,
    "use_cache": True,
    "bos_token_id": 32759,
    "eos_token_id": 32763,
    "pad_token_id": 32763,
    "tie_word_embeddings": False,
}

CONTEXT_LENGTH = NANOCHRONO_CONFIG_KWARGS["max_position_embeddings"]  # 2048

# nanoexpand only: layer indices that get an extra residual MLP (empty = identical to nanochrono)
NANOEXPAND_EXTRA_MLP_AFTER: list[int] = [12, 14, 16]


def resolve_architecture(name: str | None = None) -> str:
    arch = (name or ARCHITECTURE).strip().lower().removeprefix("sn38-")
    if arch not in ALLOWED_ARCHITECTURES:
        raise ValueError(
            f"Architecture must be one of {ALLOWED_ARCHITECTURES}, got {name!r}. "
            "ChronoGPT / other HF architectures are not used in this train path."
        )
    return arch


def nanochrono_init_kwargs() -> dict:
    """Kwargs for NanochronoConfig / NanoExpandConfig matching top-miner layout."""
    skip = {"model_type", "architectures"}
    return {k: (list(v) if isinstance(v, list) else v) for k, v in NANOCHRONO_CONFIG_KWARGS.items() if k not in skip}



# ---------------------------------------------------------------------------
# Data mix targets (Phase 5.1) — unique tokens, approximate
# ---------------------------------------------------------------------------
TOKEN_BUDGET = {
    "wikipedia_2012": (2_000_000_000, 3_000_000_000),
    "pes2o": (1_000_000_000, 2_000_000_000),
    "fineweb_edu": (10_000_000_000, 15_000_000_000),
    "books": (3_000_000_000, 5_000_000_000),
    "usenet": (10_000_000_000, 12_000_000_000),
    "qa_2012": (500_000_000, 1_000_000_000),
}
TARGET_UNIQUE_TOKENS = (27_000_000_000, 38_000_000_000)

# ---------------------------------------------------------------------------
# 2012 weighting (Phase 5.2) — start conservative, raise if needed
# ---------------------------------------------------------------------------
QA_UPSAMPLE = 2
QA_LOSS_WEIGHT = 2.5  # start 2–3×; climb toward 4–5× only if stable
DEFAULT_LOSS_WEIGHT = 1.0
# FineWeb-edu classifier (HuggingFaceFW/fineweb-edu-classifier): keep int_score >= this
# ChronoGPT / FineWeb-Edu-score-2 use threshold 2; official FineWeb-Edu default is 3.
FINEWEB_EDU_THRESHOLD = 2.0
FINEWEB_EDU_CLASSIFIER = os.environ.get(
    "FINEWEB_EDU_CLASSIFIER", "HuggingFaceFW/fineweb-edu-classifier"
)
FINEWEB_EDU_MAX_LENGTH = 512
FINEWEB_EDU_BATCH_SIZE = int(os.environ.get("FINEWEB_EDU_BATCH_SIZE", "32"))
# Sources skipped by the post-datagen edu pass (scored later via wiki → QA, or already edu)
FINEWEB_EDU_SKIP_STEMS = frozenset({"qa_2012"})

# Epochs (Phase 5.3)
EPOCHS_HIGH_CONFIDENCE = 3  # wikipedia + QA
EPOCHS_WEB = 1  # fineweb / usenet

# ---------------------------------------------------------------------------
# Training loop (Phase 6)
# ---------------------------------------------------------------------------
CHECKPOINT_EVERY_TOKENS = 1_000_000_000
CANARY_ALERT_RATE = 0.05  # above ~chance on post-2012 → audit data
BATCH_SIZE = 4
GRAD_ACCUM = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 2000
SEED = 42

# ---------------------------------------------------------------------------
# Quality / cleaning (Phase 3)
# ---------------------------------------------------------------------------
MIN_DOC_CHARS = 200
MAX_DOC_CHARS = 500_000
MINHASH_NGRAM = 5
MINHASH_THRESHOLD = 0.8
