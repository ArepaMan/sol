"""Architecture correctness gates for src/model.py.

These are the tests to name in an interview: test (c) in particular is the
one that actually verifies the "causal" in causal self-attention — everything
else can look right and still be silently broken if this one fails.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.config import ModelConfig, load_config
from src.model import GPT

CONFIG_PATH = "configs/micro_50m_8gb.yaml"


@pytest.fixture(scope="module")
def model_config() -> ModelConfig:
    return load_config(CONFIG_PATH).model


@pytest.fixture
def tiny_config() -> ModelConfig:
    # A small config for fast tests that don't need the real ~53M-param
    # model — shapes/gradients/generation behavior are architecture
    # properties, not scale properties.
    return ModelConfig(n_layer=2, n_head=4, n_embd=64, block_size=32, vocab_size=97, dropout=0.0, bias=False)


def test_param_count_near_52m(model_config):
    model = GPT(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    target = 52_000_000
    rel_error = abs(n_params - target) / target
    assert rel_error < 0.05, f"{n_params:,} params is {rel_error:.1%} off the ~52M target"
    # Pin the exact measured value too, not just the tolerance band — a
    # config edit that silently drifts the count should fail loudly here,
    # not just "still happen to be under 5%".
    assert n_params == 52_901_712


def test_forward_shape(tiny_config):
    model = GPT(tiny_config)
    B, T = 3, 10
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))
    logits, loss = model(idx)
    assert logits.shape == (B, T, tiny_config.vocab_size)
    assert loss is None  # no targets passed


def test_causality_future_tokens_do_not_affect_past_logits(tiny_config):
    """The one that matters: perturbing tokens at positions >= t must not
    change logits at position t. If this fails, the model isn't actually
    causal regardless of what the mask code claims to do."""
    model = GPT(tiny_config)
    model.eval()

    B, T = 2, 16
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))

    with torch.no_grad():
        logits_a, _ = model(idx)

        idx_perturbed = idx.clone()
        # Change every token from the midpoint onward.
        t_split = T // 2
        idx_perturbed[:, t_split:] = torch.randint(
            0, tiny_config.vocab_size, (B, T - t_split)
        )
        logits_b, _ = model(idx_perturbed)

    # Logits for positions [0, t_split) must be identical — nothing at or
    # after t_split should have been visible to them.
    assert torch.allclose(logits_a[:, :t_split], logits_b[:, :t_split], atol=1e-5)
    # Sanity: positions at/after t_split for a *different* input generally do
    # differ (guards against a vacuously-passing test where nothing changed).
    assert not torch.allclose(logits_a[:, t_split:], logits_b[:, t_split:], atol=1e-5)


def test_loss_at_init_near_uniform_cross_entropy(tiny_config):
    """An untrained model should predict roughly uniformly over the vocab:
    loss ≈ ln(vocab_size). A broken init or a weight-tying bug tends to move
    this substantially, so it's a cheap sanity check before any real
    training starts."""
    torch.manual_seed(0)
    model = GPT(tiny_config)
    model.eval()

    B, T = 4, 16
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_config.vocab_size, (B, T))

    with torch.no_grad():
        _, loss = model(idx, targets)

    expected = math.log(tiny_config.vocab_size)
    assert loss.item() == pytest.approx(expected, abs=0.15)


def test_generate_respects_max_new_tokens_and_vocab_bounds(tiny_config):
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
    max_new = 10
    out = model.generate(idx, max_new_tokens=max_new, temperature=1.0, top_k=10)
    assert out.shape == (1, 4 + max_new)
    assert out.max().item() < tiny_config.vocab_size
    assert out.min().item() >= 0
    # Original prompt is preserved as a prefix.
    assert torch.equal(out[:, :4], idx)


def test_generate_with_top_k_1_is_deterministic(tiny_config):
    """top_k=1 forces argmax sampling every step — a cheap way to check
    generate() is at least internally consistent without needing a trained
    model to produce sensible text."""
    torch.manual_seed(0)
    model = GPT(tiny_config)
    model.eval()
    idx = torch.randint(0, tiny_config.vocab_size, (1, 4))

    torch.manual_seed(1)
    out_a = model.generate(idx.clone(), max_new_tokens=8, temperature=1.0, top_k=1)
    torch.manual_seed(2)  # different sampling seed shouldn't matter with top_k=1
    out_b = model.generate(idx.clone(), max_new_tokens=8, temperature=1.0, top_k=1)

    assert torch.equal(out_a, out_b)


def test_gradient_checkpointing_matches_non_checkpointed_forward(tiny_config):
    """Checkpointing changes *how* activations are recomputed, not the math —
    logits must match exactly (well, to float precision) whether it's on or
    off. Catches the class of bug where use_reentrant/recompute interacts
    badly with a module that has its own internal state or randomness."""
    B, T = 2, 12
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))

    torch.manual_seed(42)
    model_plain = GPT(tiny_config, gradient_checkpointing=False)
    torch.manual_seed(42)
    model_ckpt = GPT(tiny_config, gradient_checkpointing=True)

    # Both must be in training mode for checkpointing to actually engage
    # (see GPT.forward: `if self.gradient_checkpointing and self.training`).
    model_plain.train()
    model_ckpt.train()

    logits_plain, _ = model_plain(idx)
    logits_ckpt, _ = model_ckpt(idx)

    assert torch.allclose(logits_plain, logits_ckpt, atol=1e-4)


def test_weight_tying_shares_the_same_tensor(tiny_config):
    model = GPT(tiny_config)
    assert model.transformer.wte.weight is model.lm_head.weight


def test_weight_tying_not_double_counted_in_param_count(tiny_config):
    model = GPT(tiny_config)
    n_params = sum(p.numel() for p in model.parameters())
    # Manually sum unique parameter tensors by identity, to cross-check that
    # .parameters() (which dedupes tied tensors by default) actually did so.
    seen_ids = set()
    manual_total = 0
    for p in model.parameters():
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            manual_total += p.numel()
    assert n_params == manual_total  # trivially true, but...
    # ...the real check: wte and lm_head must be the SAME object, so a naive
    # named_parameters() walk that didn't dedupe would overcount by exactly
    # one embedding table's worth of params.
    embedding_table_size = tiny_config.vocab_size * tiny_config.n_embd
    naive_sum = sum(p.numel() for _, p in model.named_parameters())
    assert naive_sum == n_params  # named_parameters() also dedupes by default
    assert naive_sum < n_params + embedding_table_size  # sanity: not double-counted


def test_sequence_longer_than_block_size_raises(tiny_config):
    model = GPT(tiny_config)
    idx = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.block_size + 1))
    with pytest.raises(ValueError, match="block_size"):
        model(idx)


# ---------------------------------------------------------------------------
# KV cache
#
# The cache is a pure speed change: it must not move a single logit. These are
# the causality test's twins — the causality test says future tokens can't
# reach backwards, and these say a token's logits don't depend on whether the
# past arrived all at once or one token at a time.
# ---------------------------------------------------------------------------

def test_incremental_forward_with_cache_matches_a_full_forward(tiny_config):
    """The cache's actual contract, checked on logits rather than through
    sampling: feeding a sequence one token at a time through the cache must
    give the same logits as one forward pass over the whole sequence."""
    model = GPT(tiny_config)
    model.eval()

    B, T = 2, 12
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))

    with torch.no_grad():
        full, _ = model(idx)
        cache = model.new_kv_cache()
        stepwise = torch.cat(
            [model(idx[:, t : t + 1], cache=cache)[0] for t in range(T)], dim=1
        )

    assert cache.length == T
    assert stepwise.shape == full.shape
    assert torch.allclose(stepwise, full, atol=1e-5)


def test_prefill_then_decode_matches_a_full_forward(tiny_config):
    """The shape generation actually uses: one multi-token prefill, then
    single-token steps. Exercises the T>1 and T==1 branches in one pass."""
    model = GPT(tiny_config)
    model.eval()

    T, split = 12, 7
    idx = torch.randint(0, tiny_config.vocab_size, (1, T))

    with torch.no_grad():
        full, _ = model(idx)
        cache = model.new_kv_cache()
        prefill, _ = model(idx[:, :split], cache=cache)
        decoded = torch.cat(
            [model(idx[:, t : t + 1], cache=cache)[0] for t in range(split, T)], dim=1
        )

    assert torch.allclose(torch.cat([prefill, decoded], dim=1), full, atol=1e-5)


def test_cache_makes_each_decode_step_a_single_token(tiny_config):
    """Guards against a cache that is threaded through but never populated —
    that would pass every equivalence test above and still be O(n^2). The
    widths *are* the optimisation: [prompt, 1, 1, ...] instead of a prefix
    that grows by one every step."""
    model = GPT(tiny_config)
    model.eval()
    idx = torch.randint(0, tiny_config.vocab_size, (1, 4))

    widths: list[int] = []
    inner = model.forward

    def spy(idx_in, targets=None, cache=None):
        widths.append(idx_in.size(1))
        return inner(idx_in, targets, cache=cache)

    model.forward = spy

    model.generate(idx.clone(), max_new_tokens=5, top_k=1)
    assert widths == [4, 1, 1, 1, 1]

    widths.clear()
    model.generate(idx.clone(), max_new_tokens=5, top_k=1, use_cache=False)
    assert widths == [4, 5, 6, 7, 8]


def test_cache_rebuilds_when_the_context_window_slides(tiny_config):
    """Past `block_size` the window slides, which shifts every token's learned
    positional embedding by one — so every cached key/value is stale and the
    cache has to be rebuilt from the window. That is why generation past
    block_size is no faster than it was; see docs/LIMITATIONS.md."""
    model = GPT(tiny_config)
    model.eval()
    block = tiny_config.block_size
    idx = torch.randint(0, tiny_config.vocab_size, (1, block - 2))

    widths: list[int] = []
    inner = model.forward

    def spy(idx_in, targets=None, cache=None):
        widths.append(idx_in.size(1))
        return inner(idx_in, targets, cache=cache)

    model.forward = spy
    model.generate(idx, max_new_tokens=5, top_k=1)

    # Prefill, two cheap steps, then full re-prefills once the window slides.
    assert widths == [block - 2, 1, 1, block, block]


def test_cached_and_uncached_generate_agree_past_block_size(tiny_config):
    """top_k=1 makes sampling a deterministic argmax, so this compares the
    token sequences directly rather than trusting a shared RNG stream — and
    it runs long enough to cross block_size and exercise the rebuild path."""
    model = GPT(tiny_config)
    model.eval()
    idx = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.block_size - 2))

    cached = model.generate(idx.clone(), max_new_tokens=8, top_k=1, use_cache=True)
    uncached = model.generate(idx.clone(), max_new_tokens=8, top_k=1, use_cache=False)

    assert torch.equal(cached, uncached)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype, tolerance", [(torch.float32, 1e-4), (torch.bfloat16, 5e-2)]
)
def test_cache_agrees_with_a_full_forward_on_cuda(model_config, dtype, tolerance):
    """bf16 is where byte-identity stops being available, and that is worth
    pinning down rather than discovering later.

    A cached decode step reduces over the key dimension in a different order
    than a full-width forward does. The two tolerances here are the whole
    point, and they are measured on the real 52M config:

    * **fp32** — difference ~2e-6 of logit scale. Sampling never notices, and
      cached and uncached generation come out byte-identical (which is what
      the deployed CPU path relies on, and what tests/test_infer.py asserts).
    * **bf16** — 8 mantissa bits accumulated over 8 layers puts the difference
      at ~1e-2 of logit scale. That is enough for a multinomial draw against a
      slightly shifted CDF to eventually pick a different token, after which
      the story diverges. Same story, different roll of the dice; not a
      regression, and not something the cache can fix.

    Both bounds are on *rounding*. A masking or positioning bug in the cache
    would move logits by order 1, not by 1%, and would fail both arms.
    """
    device = torch.device("cuda")
    model = GPT(model_config, gradient_checkpointing=False)
    model.to(device=device, dtype=dtype).eval()

    T = 64
    idx = torch.randint(0, model_config.vocab_size, (1, T), device=device)
    with torch.no_grad():
        full, _ = model(idx)
        cache = model.new_kv_cache()
        stepwise = torch.cat(
            [model(idx[:, t : t + 1], cache=cache)[0] for t in range(T)], dim=1
        )

    full, stepwise = full.float(), stepwise.float()
    relative = ((full - stepwise).abs().max() / full.abs().max()).item()
    assert relative < tolerance, f"{dtype} drifted {relative:.2e} of logit scale"


def test_cache_overflowing_block_size_raises(tiny_config):
    """The bound is on cached-plus-new tokens, not on the width of one call —
    a single-token forward against a full cache is still over budget."""
    model = GPT(tiny_config)
    cache = model.new_kv_cache()
    idx = torch.randint(0, tiny_config.vocab_size, (1, tiny_config.block_size))
    model(idx, cache=cache)
    with pytest.raises(ValueError, match="block_size"):
        model(idx[:, :1], cache=cache)


@pytest.mark.gpu
def test_full_model_forward_backward_on_cuda_bf16(model_config):
    """The real ~53M-param model, on the actual target hardware, in the
    actual training precision — a smoke test before M4 commits to a full
    training loop around it."""
    device = torch.device("cuda")
    model = GPT(model_config, gradient_checkpointing=True).to(device)
    model.train()

    B, T = 4, model_config.block_size
    idx = torch.randint(0, model_config.vocab_size, (B, T), device=device)
    targets = torch.randint(0, model_config.vocab_size, (B, T), device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(idx, targets)

    assert torch.isfinite(loss)
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(math.isfinite(g) for g in grad_norms)


def test_configure_optimizers_splits_decay_groups(tiny_config):
    model = GPT(tiny_config)
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95), device_type="cpu"
    )
    assert len(optimizer.param_groups) == 2
    decay_group, nodecay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert nodecay_group["weight_decay"] == 0.0
    # Every decay-group param should be >=2D (matmul weights); every
    # no-decay param should be <2D (LayerNorm gains — bias=False here, so no
    # biases in this config, but the split logic must still hold in general).
    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert all(p.dim() < 2 for p in nodecay_group["params"])
