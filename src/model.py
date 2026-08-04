"""Sol's decoder-only transformer, hand-written (nanoGPT-informed, not forked).

The point of this file is the deliverable: it's what "transformer fundamentals"
(AGENTS.md skill #3) actually means for this project. Architecture: pre-LN
causal decoder, weight-tied embeddings, GPT-2-style scaled residual init,
gradient checkpointing per block. See tests/test_model.py for the correctness
gates this file has to clear, most importantly the causality test.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # is_causal=True picks the Flash/mem-efficient causal kernel on Ada —
        # no explicit mask tensor, no manual softmax. This is the whole
        # causal-attention implementation; correctness is pinned by
        # tests/test_model.py's causality test (perturbing future tokens
        # must not change past logits).
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
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
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.config.block_size}")

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))

        for block in self.transformer.h:
            if self.gradient_checkpointing and self.training:
                # use_reentrant=False: the reentrant variant has known sharp
                # edges with SDPA's backward pass; non-reentrant is the
                # documented-preferred mode since PyTorch 2.x anyway.
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

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

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
