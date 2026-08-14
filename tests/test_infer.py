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

        def __call__(self, idx, cache=None):
            logits = torch.full((idx.size(0), idx.size(1), 32000), -1e9)
            logits[:, :, eot_id] = 0.0
            return logits, None

        def new_kv_cache(self):
            # This stub ignores its input entirely — there is no attention and
            # so nothing to cache. Returning None puts SolGenerator on the
            # uncached path, which is the honest answer for a model like this.
            return None

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


# ---------------------------------------------------------------------------
# KV cache
#
# The whole safety argument for the cache is one property: with the same seed
# and the same sampling parameters, cached and uncached generation produce
# *byte-identical* text. It is a total property — one flipped token diverges
# everything after it, so equality across prompts, seeds and lengths is very
# hard to pass by accident. See tests/test_model.py for the logit-level twins.
# ---------------------------------------------------------------------------

CACHE_EQUIVALENCE_CASES = [
    # (prompt, max_new_tokens, seed)
    ("Once upon a time", 24, 42),
    ("Once upon a time", 24, 43),
    ("The little dragon sat down and", 40, 7),
    ("A", 60, 1234),          # single-token prompt: prefill of width 1
    ("Lily and Tom went to the park to play with", 8, 99),
]


@pytest.mark.parametrize("prompt, max_new_tokens, seed", CACHE_EQUIVALENCE_CASES)
def test_cached_generation_is_byte_identical(
    tiny_generator: SolGenerator, prompt: str, max_new_tokens: int, seed: int
):
    cached = tiny_generator.generate(
        prompt, max_new_tokens=max_new_tokens, seed=seed, use_cache=True
    )
    uncached = tiny_generator.generate(
        prompt, max_new_tokens=max_new_tokens, seed=seed, use_cache=False
    )
    assert cached == uncached


def test_cached_generation_is_byte_identical_past_the_context_window(
    tiny_generator: SolGenerator,
):
    """Generation that runs past `block_size` slides the context window, and
    sliding it shifts every token's learned positional embedding — so every
    cached key/value goes stale at once. This is the case a naive cache gets
    silently, plausibly wrong."""
    prompt = "Once upon a time"
    prompt_len = len(tiny_generator.tokenizer.encode(prompt).ids)
    max_new_tokens = tiny_generator.block_size + 16 - prompt_len
    assert prompt_len + max_new_tokens > tiny_generator.block_size  # not vacuous

    # stop_at_eot=False so the full budget is always spent: an early stop would
    # leave this passing without ever reaching the window slide.
    kwargs = dict(max_new_tokens=max_new_tokens, seed=42, stop_at_eot=False)
    cached = tiny_generator.generate(prompt, **kwargs)
    uncached = tiny_generator.generate(prompt, use_cache=False, **kwargs)
    assert cached == uncached


def test_cached_generation_is_byte_identical_for_an_overlong_prompt(
    tiny_generator: SolGenerator,
):
    """The other half of the truncation path: a prompt that is already longer
    than the context window, so the very first step is a tail-truncated
    prefill and the window slides from the second token onward."""
    prompt = "the little cat ran " * 40
    assert (
        len(tiny_generator.tokenizer.encode(prompt).ids) >= tiny_generator.block_size
    )  # not vacuous

    cached = tiny_generator.generate(prompt, max_new_tokens=12, seed=42)
    uncached = tiny_generator.generate(
        prompt, max_new_tokens=12, seed=42, use_cache=False
    )
    assert cached == uncached


def test_streamed_deltas_are_identical_with_and_without_the_cache(
    tiny_generator: SolGenerator,
):
    """Not just the final text: the *chunking* has to match too, since that is
    what the demos render token by token."""
    cached = list(tiny_generator.stream("Once upon a time", max_new_tokens=20, seed=42))
    uncached = list(
        tiny_generator.stream(
            "Once upon a time", max_new_tokens=20, seed=42, use_cache=False
        )
    )
    assert cached == uncached


def test_cache_is_on_by_default(tiny_generator: SolGenerator):
    """The point of all the equivalence tests above is that the fast path can
    be the default. If this ever flips, the speedup silently stops shipping."""
    widths: list[int] = []
    inner = tiny_generator.model.forward

    def spy(idx, targets=None, cache=None):
        widths.append(idx.size(1))
        return inner(idx, targets, cache=cache)

    tiny_generator.model.forward = spy
    try:
        tiny_generator.generate(
            "Once upon a time", max_new_tokens=5, seed=42, stop_at_eot=False
        )
    finally:
        tiny_generator.model.forward = inner

    assert widths[1:] == [1, 1, 1, 1]  # prefill, then one token per step
