"""Unit tests for the M8 inference path (`src/infer.py`).

Deliberately runs on a *randomly initialized tiny* GPT, not the 635 MB baseline
checkpoint: what's under test is the generation loop's contract — EOT stopping,
streaming deltas, seed determinism, context truncation — none of which depends
on the weights being any good. That keeps the suite runnable on a machine with
no checkpoints and no GPU.
"""

from __future__ import annotations

import json

import pytest
import torch
from tokenizers import Tokenizer

from src.config import ModelConfig
from src.infer import FALLBACK_EOT_ID, SolGenerator, load_eot_id, resolve_dtype
from src.model import GPT

TOKENIZER_PATH = "data/tokenizer/tokenizer.json"


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_file(TOKENIZER_PATH)


@pytest.fixture(scope="module")
def tiny_generator(tokenizer: Tokenizer) -> SolGenerator:
    cfg = ModelConfig(n_layer=2, n_head=2, n_embd=32, block_size=64, vocab_size=32000)
    model = GPT(cfg, gradient_checkpointing=False).eval()
    return SolGenerator(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        eot_id=0,
        block_size=cfg.block_size,
    )


# ---------------------------------------------------------------------------
# dtype / metadata resolution
# ---------------------------------------------------------------------------

def test_cpu_always_resolves_to_float32():
    # bf16 CPU kernels are reference implementations; fp32 is faster there and
    # a strict widening of bf16 weights, so quality can't regress.
    assert resolve_dtype(torch.device("cpu"), "bfloat16") is torch.float32


def test_cuda_honours_config_precision():
    assert resolve_dtype(torch.device("cuda"), "bfloat16") is torch.bfloat16
    assert resolve_dtype(torch.device("cuda"), "float32") is torch.float32


def test_eot_id_read_from_meta(tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"eot_id": 7}), encoding="utf-8")
    assert load_eot_id(meta) == 7


def test_eot_id_falls_back_when_meta_missing(tmp_path):
    assert load_eot_id(tmp_path / "nope.json") == FALLBACK_EOT_ID
    assert load_eot_id(None) == FALLBACK_EOT_ID


# ---------------------------------------------------------------------------
# Generation contract
# ---------------------------------------------------------------------------

def test_generate_returns_prompt_plus_completion(tiny_generator: SolGenerator):
    out = tiny_generator.generate("Once upon a time", max_new_tokens=8, seed=42)
    assert out.startswith("Once upon a time")
    assert len(out) > len("Once upon a time")


def test_same_seed_is_byte_identical(tiny_generator: SolGenerator):
    a = tiny_generator.generate("Once upon a time", max_new_tokens=16, seed=42)
    b = tiny_generator.generate("Once upon a time", max_new_tokens=16, seed=42)
    assert a == b


def test_different_seeds_diverge(tiny_generator: SolGenerator):
    a = tiny_generator.generate("Once upon a time", max_new_tokens=32, seed=42)
    b = tiny_generator.generate("Once upon a time", max_new_tokens=32, seed=43)
    assert a != b


def test_stream_concatenates_to_generate(tiny_generator: SolGenerator):
    streamed = "".join(tiny_generator.stream("Once upon a time", max_new_tokens=16, seed=42))
    whole = tiny_generator.generate("Once upon a time", max_new_tokens=16, seed=42)
    assert "Once upon a time" + streamed == whole


def test_stream_yields_only_the_completion(tiny_generator: SolGenerator):
    chunks = list(tiny_generator.stream("Once upon a time", max_new_tokens=4, seed=42))
    assert "".join(chunks).count("Once upon a time") == 0


def test_generation_stops_at_eot(tokenizer: Tokenizer):
    """The M6 gap this module exists to close: without a stop condition the
    model runs its whole token budget and keeps writing past the story's end."""
    eot_id = 5

    class AlwaysEOT:
        """Stub model whose next-token distribution is a point mass on EOT.
        Biasing a real GPT's logits isn't reliable here — the lm_head weight is
        tied to the embedding table, so nudging a row changes both sides of the
        dot product and the sampled token stays anyone's guess."""

        def __call__(self, idx):
            logits = torch.full((idx.size(0), idx.size(1), 32000), -1e9)
            logits[:, :, eot_id] = 0.0
            return logits, None

    gen = SolGenerator(AlwaysEOT(), tokenizer, torch.device("cpu"), eot_id, 64)
    out = gen.generate("Once upon a time", max_new_tokens=50, seed=42)
    assert out == "Once upon a time"  # stopped before emitting anything

    unstopped = gen.generate("Once upon a time", max_new_tokens=5, seed=42, stop_at_eot=False)
    assert len(unstopped) > len("Once upon a time")


def test_overlong_prompt_is_truncated_to_context(tiny_generator: SolGenerator):
    # block_size is 64 here; this prompt encodes to far more than that.
    long_prompt = "the little cat ran " * 200
    out = tiny_generator.generate(long_prompt, max_new_tokens=4, seed=42)
    assert out.startswith(long_prompt)  # prompt echoed in full, not truncated in the output


def test_top_k_none_is_accepted(tiny_generator: SolGenerator):
    out = tiny_generator.generate("Once upon a time", max_new_tokens=4, top_k=None, seed=42)
    assert out.startswith("Once upon a time")
