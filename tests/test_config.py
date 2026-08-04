"""Config loading and the derived-scale arithmetic."""

from __future__ import annotations

import pytest

from src.config import ModelConfig, load_config

CONFIG_PATH = "configs/micro_50m_8gb.yaml"


def test_loads_baseline_config():
    cfg = load_config(CONFIG_PATH)
    assert cfg.name == "micro_50m_8gb"
    assert cfg.model.n_layer == 8
    # 592, not the originally-guessed 512 — with weight tying, 512 measures
    # at 41.8M params (~20% under the project's "~52M" branding). See
    # configs/micro_50m_8gb.yaml's header comment and docs/PROJECT.md.
    assert cfg.model.n_embd == 592
    assert cfg.training.learning_rate == pytest.approx(3.0e-4)
    assert cfg.seed == 42


def test_precision_is_bf16_not_fp16():
    # Guards against the old RTX 2070 Super / float16 assumption creeping back.
    assert load_config(CONFIG_PATH).precision == "bfloat16"


def test_head_dim_divides_evenly():
    assert load_config(CONFIG_PATH).model.head_dim == 74


def test_derived_scale_arithmetic():
    cfg = load_config(CONFIG_PATH)
    assert cfg.tokens_per_step == 4 * 16 * 512 == 32_768
    assert cfg.total_train_tokens == 32_768 * 40_000
    # ~3.66 epochs over the measured 357.85M-token corpus (M1) — the
    # distinction the original 400M-target spec lost.
    assert cfg.epochs == pytest.approx(3.66, abs=0.05)


def test_vocab_fits_uint16():
    # data/tokenize.py stores ids as uint16; a larger vocab would silently wrap.
    assert load_config(CONFIG_PATH).model.vocab_size <= 65535


def test_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  n_layer: 8\n  n_layers: 8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(bad)


def test_rejects_indivisible_head_dim():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_embd=512, n_head=7)
