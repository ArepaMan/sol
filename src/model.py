"""Sol's decoder-only transformer, hand-written (nanoGPT-informed, not forked).

The point of this file is the deliverable: it's what "transformer fundamentals"
(AGENTS.md skill #3) actually means for this project. Architecture: pre-LN
causal decoder, weight-tied embeddings, GPT-2-style scaled residual init,
gradient checkpointing per block. See tests/test_model.py for the correctness
gates this file has to clear, most importantly the causality test.

Inference carries an optional **KV cache** (`KVCache` below). Without one,
sampling re-runs the whole prefix through every layer for every new token —
O(n^2) work to produce n tokens, and n-1 of every n key/value projections are
recomputations of a value that cannot have changed, because the causal mask
means a past token's keys and values never depend on anything after it. The
cache is a pure speed change and is tested as one: cached and uncached
generation must be byte-identical.
"""

from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.config import ModelConfig

# GPT-2's scaled-init constant for residual-stream projections (c_proj in
# both attention and MLP). Keeps residual-stream variance from growing with
# depth — see Radford et al. 2019 and nanoGPT's model.py for the same choice.
_RESID_INIT_STD = 0.02


class LayerKVCache:
    """Preallocated key/value buffers for one attention layer.

    `capacity` is `block_size`, so the buffers are allocated once at their
    final size and filled left to right. Appending with `torch.cat` instead
    would reallocate and copy the whole history on every token — correct, but
    it reintroduces an O(n^2) term in memory traffic to remove one in compute.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.k: torch.Tensor | None = None  # allocated on first append
        self.v: torch.Tensor | None = None
        self.length = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store `k`/`v` for the new positions, return the full history so far.

        Buffers are allocated lazily from the first tensors seen, which is how
        the cache picks up the model's device and dtype without being told.
        """
        if self.k is None:
            B, n_head, _, head_dim = k.shape
            self.k = k.new_zeros((B, n_head, self.capacity, head_dim))
            self.v = v.new_zeros((B, n_head, self.capacity, head_dim))

        end = self.length + k.size(2)
        self.k[:, :, self.length : end] = k
        self.v[:, :, self.length : end] = v
        self.length = end
        return self.k[:, :, :end], self.v[:, :, :end]

    def reset(self) -> None:
        """Forget everything cached, keeping the buffers. Stale entries are
        never read (`append` writes before `length` advances past them), so
        there is nothing to zero."""
        self.length = 0


class KVCache:
    """One `LayerKVCache` per transformer block — the state of one generation.

    Cheap: 8 layers x 512 positions x 592 channels x 2 (K and V) x 4 bytes is
    ~19 MB for Sol's config, against the ~400 MB of headroom the deployed app
    has under Streamlit Community Cloud's 1 GB ceiling (docs/DEPLOY.md).
    """

    def __init__(self, n_layer: int, capacity: int) -> None:
        self.capacity = capacity
        self.layers = [LayerKVCache(capacity) for _ in range(n_layer)]

    @property
    def length(self) -> int:
        """Tokens currently cached. Every layer advances in lockstep, so
        layer 0 speaks for all of them."""
        return self.layers[0].length

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def next_input(self, idx: torch.Tensor) -> torch.Tensor:
        """Pick the tokens to feed for the next decode step, invalidating the
        cache first if the context window has slid.

        Three cases, and the third is the one worth knowing about:

        * **Empty cache** — feed the whole window. This is the prefill.
        * **Warm cache** — feed only the newest token. Everything before it is
          already keyed and valued, and causality guarantees those entries
          cannot have changed.
        * **Window slid past `capacity`** — Sol uses *learned absolute*
          positional embeddings, so when the window slides by one, every
          surviving token is re-embedded at a position one lower than before.
          Every cached entry is stale at once and the only correct move is to
          drop the cache and re-prefill. Generation past `block_size` is
          therefore no faster than it was before this cache existed; RoPE
          would not fix it either, since the shift is in the embedding, not
          just the score. See docs/LIMITATIONS.md.
        """
        if idx.size(1) > self.capacity:
            self.reset()
            return idx[:, -self.capacity :]
        if self.length == 0:
            return idx
        return idx[:, -1:]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.SOL_SCALED_RESID_INIT = True  # picked up by GPT._init_weights

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cache: LayerKVCache | None = None) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        if cache is not None:
            # Queries stay narrow (just the new tokens); keys and values widen
            # to the whole history.
            k, v = cache.append(k, v)

        # is_causal=True picks the Flash/mem-efficient causal kernel on Ada —
        # no explicit mask tensor, no manual softmax. This is the whole
        # causal-attention implementation; correctness is pinned by
        # tests/test_model.py's causality test (perturbing future tokens
        # must not change past logits).
        #
        # It is only *correct* when the query and key blocks are the same
        # length, though: SDPA aligns its causal mask to the top-left corner,
        # so a short query block against a long cached key block would mask
        # out exactly the history the cache exists to serve. Hence the three
        # cases below, all pinned by tests/test_model.py's cache tests.
        attn_mask, is_causal = None, False
        if k.size(2) == T:
            is_causal = True  # no cached past: the ordinary square causal mask
        elif T > 1:
            # Prefill against a warm cache. Query i sits at absolute position
            # n_past + i and may attend to keys 0..n_past+i. `generate` never
            # takes this path (it prefills once into an empty cache, then
            # decodes one token at a time), but a mask that is only right for
            # the paths one caller happens to use is a trap for the next one.
            n_past = k.size(2) - T
            attn_mask = torch.ones(T, k.size(2), dtype=torch.bool, device=x.device).tril(
                diagonal=n_past
            )
        # else: T == 1, a single query attending to the entire past. Every key
        # is legal, so the correct mask is no mask at all.

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """4x-expansion GELU MLP — GPT-2 style, not SwiGLU (see docs/LIMITATIONS.md)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.SOL_SCALED_RESID_INIT = True
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LN transformer block: x = x + attn(ln(x)); x = x + mlp(ln(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, cache: LayerKVCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cache=cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig, gradient_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: input embedding and output projection share one tensor.
        # Standard since Press & Wolf 2017 — halves the embedding-table
        # parameter cost and nn.Module.parameters()/named_parameters() dedupe
        # tied tensors automatically, so this doesn't double-count anywhere
        # downstream (optimizer param groups, count_params, etc).
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scaled init specifically for residual-stream projections, applied
        # after the general pass above so it overrides the default 0.02 std
        # on exactly the params marked SOL_SCALED_RESID_INIT.
        for module in self.modules():
            if getattr(module, "SOL_SCALED_RESID_INIT", False):
                nn.init.normal_(
                    module.weight, mean=0.0, std=_RESID_INIT_STD / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=_RESID_INIT_STD)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=_RESID_INIT_STD)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        # With a cache, `idx` holds only the *new* tokens — the bound applies
        # to the whole conditioned sequence, cached history included.
        n_past = cache.length if cache is not None else 0
        if n_past + T > self.config.block_size:
            raise ValueError(
                f"sequence length {n_past + T} exceeds block_size {self.config.block_size}"
            )

        pos = torch.arange(n_past, n_past + T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))

        for i, block in enumerate(self.transformer.h):
            if self.gradient_checkpointing and self.training:
                # use_reentrant=False: the reentrant variant has known sharp
                # edges with SDPA's backward pass; non-reentrant is the
                # documented-preferred mode since PyTorch 2.x anyway.
                # No cache here on purpose: checkpointing is a training-time
                # memory trade and the cache is inference-only, so the two
                # never co-occur.
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x, cache=cache.layers[i] if cache is not None else None)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        """AdamW with decay/no-decay param groups: 2D+ tensors (matmul weights)
        get weight decay, 1D tensors (LayerNorm gains, biases) don't — decaying
        a LayerNorm scale toward zero has no principled justification and
        empirically hurts. Standard nanoGPT/GPT-2 practice."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, fused=use_fused
        )

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float, peak_flops: float) -> float:
        """Model FLOPs utilization, PaLM-appendix-B style (also nanoGPT's
        formula). `peak_flops` is the GPU's advertised peak — pass the
        measured/vendor number for whatever precision training runs in;
        there's no universal constant, so it's not hardcoded here."""
        N = sum(p.numel() for p in self.parameters())
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_achieved = flops_per_fwdbwd * fwdbwd_per_iter / dt
        return flops_achieved / peak_flops

    def new_kv_cache(self) -> KVCache:
        """A fresh cache sized for this model. One per generation — a cache
        holds the state of a single sequence, so sharing one across concurrent
        generations would interleave their histories."""
        return KVCache(n_layer=self.config.n_layer, capacity=self.config.block_size)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Sample `max_new_tokens` continuations of `idx`.

        `use_cache=False` recomputes the entire prefix every step. It exists to
        be compared against: the safety argument for the cache is that the two
        paths produce identical output (tests/test_infer.py), and that argument
        needs both paths to stay runnable.
        """
        self.eval()
        cache = self.new_kv_cache() if use_cache else None
        for _ in range(max_new_tokens):
            idx_cond = (
                cache.next_input(idx)
                if cache is not None
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond, cache=cache)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
