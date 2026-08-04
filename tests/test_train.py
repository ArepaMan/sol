"""Pure-function tests for the LR schedule — no GPU needed."""

from __future__ import annotations

import pytest

from src.train import get_lr

LR = 3.0e-4
MIN_LR = 3.0e-5
WARMUP = 1000
DECAY_ITERS = 40000


def test_lr_at_start_is_near_zero_not_zero():
    # it=0 should not literally be 0 — a truly-zero first-step LR wastes a step.
    lr0 = get_lr(0, WARMUP, DECAY_ITERS, LR, MIN_LR)
    assert 0 < lr0 < LR / WARMUP * 10


def test_lr_ramps_linearly_during_warmup():
    lr_mid = get_lr(WARMUP // 2, WARMUP, DECAY_ITERS, LR, MIN_LR)
    lr_start = get_lr(0, WARMUP, DECAY_ITERS, LR, MIN_LR)
    lr_end = get_lr(WARMUP - 1, WARMUP, DECAY_ITERS, LR, MIN_LR)
    assert lr_start < lr_mid < lr_end < LR


def test_lr_reaches_peak_at_warmup_boundary():
    # Right at the boundary, LR should be at (or extremely close to) peak.
    lr_at_warmup = get_lr(WARMUP, WARMUP, DECAY_ITERS, LR, MIN_LR)
    assert lr_at_warmup == pytest.approx(LR, rel=1e-3)


def test_lr_decays_to_min_lr_at_decay_iters():
    lr_final = get_lr(DECAY_ITERS, WARMUP, DECAY_ITERS, LR, MIN_LR)
    assert lr_final == pytest.approx(MIN_LR, abs=1e-9)


def test_lr_stays_at_min_lr_past_decay_iters():
    lr_past = get_lr(DECAY_ITERS + 5000, WARMUP, DECAY_ITERS, LR, MIN_LR)
    assert lr_past == MIN_LR


def test_lr_monotonically_decreases_after_warmup():
    checkpoints = [WARMUP, WARMUP + 5000, WARMUP + 15000, WARMUP + 25000, DECAY_ITERS]
    lrs = [get_lr(it, WARMUP, DECAY_ITERS, LR, MIN_LR) for it in checkpoints]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_lr_never_exceeds_peak():
    for it in range(0, DECAY_ITERS + 10000, 500):
        lr = get_lr(it, WARMUP, DECAY_ITERS, LR, MIN_LR)
        assert lr <= LR + 1e-9


def test_lr_never_drops_below_min_after_warmup():
    # min_lr is only a floor for the post-warmup decay phase — during warmup
    # itself, LR legitimately starts near zero (see
    # test_lr_at_start_is_near_zero_not_zero) and ramps up through it.
    for it in range(WARMUP, DECAY_ITERS + 10000, 500):
        lr = get_lr(it, WARMUP, DECAY_ITERS, LR, MIN_LR)
        assert lr >= MIN_LR - 1e-12


def test_zero_warmup_starts_at_peak():
    lr0 = get_lr(0, 0, DECAY_ITERS, LR, MIN_LR)
    assert lr0 == pytest.approx(LR, rel=1e-6)


def test_short_run_still_reaches_min_lr_when_decay_iters_matches_max_iters():
    # This is the property src.train.main() relies on when --max-iters is
    # given without an explicit --lr-decay-iters: a shortened run's decay
    # horizon is synced to max_iters so it still completes the cosine decay
    # rather than ending stuck near peak LR.
    short_decay_iters = 200
    lr_at_end = get_lr(short_decay_iters, 50, short_decay_iters, LR, MIN_LR)
    assert lr_at_end == pytest.approx(MIN_LR, abs=1e-9)
