"""Train path must only allow sn38-nanochrono / sn38-nanoexpand."""

import pytest

from train.config import (
    NANOCHRONO_CONFIG_KWARGS,
    _hybrid_layer_types,
    nanochrono_init_kwargs,
    resolve_architecture,
)


def test_resolve_nanochrono():
    assert resolve_architecture("nanochrono") == "nanochrono"
    assert resolve_architecture("sn38-nanochrono") == "nanochrono"


def test_resolve_nanoexpand():
    assert resolve_architecture("nanoexpand") == "nanoexpand"
    assert resolve_architecture("sn38-nanoexpand") == "nanoexpand"


def test_reject_chronogpt_and_others():
    for bad in ("chronogpt", "llama", "gpt2", "miniformer", "sn38-miniformer"):
        with pytest.raises(ValueError, match="Architecture must be one of"):
            resolve_architecture(bad)


def test_top_miner_hybrid_layer_types():
    types = _hybrid_layer_types(28)
    assert len(types) == 28
    assert types[0:4] == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert types.count("full_attention") == 7
    assert types.count("sliding_attention") == 21


def test_top_miner_config_matches_chrono3_2025():
    cfg = NANOCHRONO_CONFIG_KWARGS
    assert cfg["hidden_size"] == 1792
    assert cfg["intermediate_size"] == 7168
    assert cfg["num_hidden_layers"] == 28
    assert cfg["num_attention_heads"] == 14
    assert cfg["max_position_embeddings"] == 2048
    assert cfg["sliding_window"] == 512
    assert cfg["vocab_size"] == 32768
    assert cfg["bos_token_id"] == 32759
    assert cfg["eos_token_id"] == 32763
    assert cfg["pad_token_id"] == 32763
    assert cfg["tap_layer"] == 14
    assert cfg["aux_layers"] == list(range(1, 28, 2))
    assert cfg["layer_types"] == _hybrid_layer_types(28)
    kwargs = nanochrono_init_kwargs()
    assert "model_type" not in kwargs
    assert kwargs["layer_types"][3] == "full_attention"
