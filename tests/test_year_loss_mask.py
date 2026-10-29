"""Unit tests for post-cutoff year label masking in train_clm."""

import torch

from train.train_clm import mask_labels_for_post_cutoff_years


def test_mask_year_token_in_labels():
    # ids: [bos, a, YEAR, b, eos]  — YEAR is single token 99
    ids = [1, 10, 99, 11, 2]
    labels = torch.tensor(ids[1:])  # [10, 99, 11, 2]
    out = mask_labels_for_post_cutoff_years(ids, labels, year_seqs=[[99]], window_after=0)
    # predicting ids[2]=99 → labels index 1
    assert out[1].item() == -100
    assert out[0].item() == 10
    assert out[2].item() == 11


def test_mask_year_with_window_after():
    ids = [1, 10, 99, 11, 12, 2]
    labels = torch.tensor(ids[1:])  # [10, 99, 11, 12, 2]
    out = mask_labels_for_post_cutoff_years(ids, labels, year_seqs=[[99]], window_after=2)
    # mask labels for year + 2 following predictions
    assert out[1].item() == -100  # 99
    assert out[2].item() == -100  # 11
    assert out[3].item() == -100  # 12
    assert out[0].item() == 10
