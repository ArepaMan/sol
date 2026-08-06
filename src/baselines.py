"""Dumb reference language models: uniform, unigram, trigram-with-backoff.

These exist to give Sol-001's perplexity a floor to beat, not to be good
language models. Scope decision: n-gram counts are fit on a **contiguous
prefix of `--train-tokens` train tokens** (default 10M, ~2.8% of the 357.85M
train corpus), not the full corpus. A full-corpus trigram table needs an
on-disk sparse structure (tens of millions of unique (w1,w2,w3) keys) that is
out of scope for what's meant to be a weak reference floor — see
`docs/ROADMAP.md` M6. Bumping `--train-tokens` fits a stronger trigram
baseline at the cost of fit time/memory; it does not change the conclusion
these baselines exist to support (Sol-001 beats naive statistical floors by a
wide margin).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

# Brants et al. 2007 "stupid backoff" discount. Not a Bayesian prior, not
# tuned — it's the standard fixed constant from that paper, chosen because
# stupid backoff explicitly trades a normalized probability distribution for
# speed/simplicity at web scale. That's fine here: we only ever need the
# probability mass the model assigns to the *actual* next token in the val
# set, never a full normalized distribution over the vocabulary.
_STUPID_BACKOFF_ALPHA = 0.4


class UniformBaseline:
    """Every token equally likely. NLL is a constant: ln(vocab_size)."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def nll(self, tokens: np.ndarray) -> np.ndarray:
        """Per-token NLL (natural log) for predicting tokens[1:] given
        tokens[:i]. Returns an array of length len(tokens) - 1."""
        n = max(len(tokens) - 1, 0)
        return np.full(n, np.log(self.vocab_size), dtype=np.float64)


class UnigramBaseline:
    """P(w) from train frequency, add-k smoothed so no token has zero
    probability (a single unseen token would otherwise blow the whole
    document's NLL to +inf)."""

    def __init__(self, vocab_size: int, add_k: float = 1.0):
        self.vocab_size = vocab_size
        self.add_k = add_k
        self.log_probs: np.ndarray | None = None

    def fit(self, tokens: np.ndarray) -> "UnigramBaseline":
        counts = np.bincount(tokens, minlength=self.vocab_size).astype(np.float64)
        smoothed = counts + self.add_k
        probs = smoothed / smoothed.sum()
        self.log_probs = np.log(probs)
        return self

    def nll(self, tokens: np.ndarray) -> np.ndarray:
        assert self.log_probs is not None, "call fit() first"
        targets = tokens[1:]
        return -self.log_probs[targets]


class TrigramBackoffBaseline:
    """P(w3 | w1, w2) via counts, stupid-backing off to bigram then unigram
    then uniform when a context was never seen in the fit corpus.

    Not a normalized probability distribution (stupid backoff's whole point
    is skipping the normalization pass) — treated here as a practical n-gram
    NLL floor, not a true LM. See module docstring.
    """

    def __init__(self, vocab_size: int, alpha: float = _STUPID_BACKOFF_ALPHA):
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.unigram_counts: np.ndarray | None = None
        self.total_unigrams: int = 0
        self.bigram_ctx: dict[int, Counter] | None = None
        self.trigram_ctx: dict[tuple[int, int], Counter] | None = None

    def fit(self, tokens: np.ndarray) -> "TrigramBackoffBaseline":
        toks = tokens.tolist()  # python ints are faster as dict/Counter keys than np scalars
        self.unigram_counts = np.bincount(np.asarray(toks, dtype=np.int64), minlength=self.vocab_size)
        self.total_unigrams = len(toks)

        bigram_ctx: dict[int, Counter] = defaultdict(Counter)
        trigram_ctx: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for i in range(len(toks) - 2):
            w1, w2, w3 = toks[i], toks[i + 1], toks[i + 2]
            bigram_ctx[w1][w2] += 1
            trigram_ctx[(w1, w2)][w3] += 1
        self.bigram_ctx = bigram_ctx
        self.trigram_ctx = trigram_ctx
        return self

    def _unigram_prob(self, w: int) -> float:
        return (self.unigram_counts[w] + 1) / (self.total_unigrams + self.vocab_size)

    def _bigram_prob(self, w1: int, w2: int) -> float:
        ctx = self.bigram_ctx.get(w1)
        if ctx:
            ctx_total = sum(ctx.values())
            if w2 in ctx:
                return ctx[w2] / ctx_total
        return self.alpha * self._unigram_prob(w2)

    def _trigram_prob(self, w1: int, w2: int, w3: int) -> float:
        ctx = self.trigram_ctx.get((w1, w2))
        if ctx:
            ctx_total = sum(ctx.values())
            if w3 in ctx:
                return ctx[w3] / ctx_total
        return self.alpha * self._bigram_prob(w2, w3)

    def nll(self, tokens: np.ndarray) -> np.ndarray:
        """out[i] = -log P(tokens[i+1] | tokens[:i+1]), i.e. NLL for
        predicting tokens[1:]. out[0] only has one token of history
        (tokens[0]) so it backs off to a bigram query; every later position
        has >= 2 tokens of history and uses the full trigram context."""
        assert self.trigram_ctx is not None, "call fit() first"
        toks = tokens.tolist()
        n = len(toks) - 1
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            if i == 0:
                p = self._bigram_prob(toks[0], toks[1])
            else:
                p = self._trigram_prob(toks[i - 1], toks[i], toks[i + 1])
            out[i] = -np.log(max(p, 1e-12))
        return out
