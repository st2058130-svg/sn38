"""ChronoGPT-scale token budget helpers (chars/4 ≈ tokens)."""

from __future__ import annotations

from train.config import TOKEN_BUDGET

# Midpoint targets used by full-scale datagen
TARGET_TOKENS = {k: (lo + hi) // 2 for k, (lo, hi) in TOKEN_BUDGET.items()}


def est_tokens(text: str) -> int:
    return max(0, len(text or "") // 4)


def hit_target(n_tokens: int, target: int | None) -> bool:
    return target is not None and n_tokens >= target
