"""Unit tests for the M6 evaluation harness: baselines, bootstrap CI,
length-bucketing, document splitting, and repetition metrics. All pure-Python
/ numpy — no GPU, no real checkpoint needed."""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines import TrigramBackoffBaseline, UnigramBaseline, UniformBaseline
from src.eval import bootstrap_ppl_ci, bucket_perplexity, load_documents
from src.generate_samples import distinct_n, max_repeated_substring_len


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def test_uniform_nll_is_log_vocab_size_constant():
    b = UniformBaseline(vocab_size=100)
    nlls = b.nll(np.array([1, 2, 3, 4]))
    assert len(nlls) == 3
    assert nlls == pytest.approx(np.log(100))


def test_uniform_nll_empty_for_single_token_doc():
    b = UniformBaseline(vocab_size=100)
    assert len(b.nll(np.array([1]))) == 0


def test_unigram_prefers_frequent_tokens():
    # token 0 appears far more often than token 1 in the fit corpus.
    train = np.array([0] * 90 + [1] * 10)
    b = UnigramBaseline(vocab_size=2, add_k=1.0).fit(train)
    nll_common = b.nll(np.array([5, 0]))[0]  # predicting the frequent token
    nll_rare = b.nll(np.array([5, 1]))[0]  # predicting the rare token
    assert nll_common < nll_rare


def test_unigram_never_gives_zero_probability_to_unseen_token():
    train = np.array([0, 0, 0, 0])  # token 1 never appears
    b = UnigramBaseline(vocab_size=2, add_k=1.0).fit(train)
    nll = b.nll(np.array([0, 1]))[0]
    assert np.isfinite(nll)


def test_trigram_backs_off_to_bigram_then_unigram_on_unseen_context():
    # "0 1" always followed by 2 in training -> trigram should nail it.
    train = np.array([0, 1, 2] * 50)
    b = TrigramBackoffBaseline(vocab_size=5).fit(train)
    nll_seen = b.nll(np.array([0, 1, 2]))[1]  # predict token after context (0,1)
    # A context that never occurred (3, 4) must still produce a finite NLL via backoff.
    nll_unseen_ctx = b.nll(np.array([3, 4, 2]))[1]
    assert np.isfinite(nll_seen)
    assert np.isfinite(nll_unseen_ctx)
    assert nll_seen < nll_unseen_ctx  # confident seen trigram beats a backed-off guess


def test_trigram_more_confident_than_unigram_on_learned_pattern():
    train = np.array([0, 1, 2] * 200)
    trigram = TrigramBackoffBaseline(vocab_size=5).fit(train)
    unigram = UnigramBaseline(vocab_size=5).fit(train)
    tri_nll = trigram.nll(np.array([0, 1, 2]))[1]
    uni_nll = unigram.nll(np.array([0, 1, 2]))[1]
    assert tri_nll < uni_nll


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    doc_nlls = [rng.normal(1.0, 0.1, size=rng.integers(5, 20)) for _ in range(50)]
    point, lo, hi = bootstrap_ppl_ci(doc_nlls, n_boot=500, seed=0)
    assert lo <= point <= hi


def test_bootstrap_ci_widens_with_fewer_documents():
    rng = np.random.default_rng(1)
    many_docs = [rng.normal(1.0, 0.5, size=10) for _ in range(200)]
    few_docs = [rng.normal(1.0, 0.5, size=10) for _ in range(8)]
    _, lo_many, hi_many = bootstrap_ppl_ci(many_docs, n_boot=1000, seed=1)
    _, lo_few, hi_few = bootstrap_ppl_ci(few_docs, n_boot=1000, seed=1)
    assert (hi_few - lo_few) > (hi_many - lo_many)


def test_bootstrap_ci_deterministic_given_seed():
    docs = [np.array([1.0, 2.0, 1.5]) for _ in range(10)]
    r1 = bootstrap_ppl_ci(docs, n_boot=200, seed=7)
    r2 = bootstrap_ppl_ci(docs, n_boot=200, seed=7)
    assert r1 == r2


# ---------------------------------------------------------------------------
# Length bucketing
# ---------------------------------------------------------------------------

def test_bucket_perplexity_assigns_docs_to_correct_bucket():
    doc_nlls = [np.full(10, 1.0), np.full(100, 2.0)]
    doc_lens = [10, 100]
    buckets = bucket_perplexity(doc_nlls, doc_lens)
    assert buckets["1-64"]["n_docs"] == 1
    assert buckets["65-128"]["n_docs"] == 1
    assert buckets["129-256"]["n_docs"] == 0
    assert buckets["129-256"]["ppl"] is None


def test_bucket_perplexity_matches_exp_mean_nll_within_bucket():
    doc_nlls = [np.full(10, np.log(2))]
    doc_lens = [10]
    buckets = bucket_perplexity(doc_nlls, doc_lens)
    assert buckets["1-64"]["ppl"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Document splitting
# ---------------------------------------------------------------------------

def test_load_documents_splits_on_eot(tmp_path):
    eot = 0
    tokens = np.array([5, 6, eot, 7, 8, 9, eot], dtype=np.uint16)
    path = tmp_path / "val.bin"
    tokens.tofile(path)
    docs, n_truncated = load_documents(path, eot_id=eot, block_size=512)
    assert len(docs) == 2
    assert list(docs[0]) == [5, 6, eot]
    assert list(docs[1]) == [7, 8, 9, eot]
    assert n_truncated == 0


def test_load_documents_truncates_overlong_docs(tmp_path):
    eot = 0
    tokens = np.array(list(range(1, 21)) + [eot], dtype=np.uint16)
    path = tmp_path / "val.bin"
    tokens.tofile(path)
    docs, n_truncated = load_documents(path, eot_id=eot, block_size=5)
    assert n_truncated == 1
    assert len(docs[0]) == 5


def test_load_documents_drops_single_token_docs(tmp_path):
    eot = 0
    tokens = np.array([eot, 5, 6, eot], dtype=np.uint16)  # first "doc" is just the eot itself
    path = tmp_path / "val.bin"
    tokens.tofile(path)
    docs, _ = load_documents(path, eot_id=eot, block_size=512)
    assert len(docs) == 1  # the length-1 leading doc was dropped


# ---------------------------------------------------------------------------
# Repetition metrics
# ---------------------------------------------------------------------------

def test_distinct_n_is_one_for_fully_unique_tokens():
    assert distinct_n(["a", "b", "c", "d"], n=2) == pytest.approx(1.0)


def test_distinct_n_drops_with_repetition():
    high = distinct_n(["a", "b", "c", "d", "e", "f"], n=2)
    low = distinct_n(["a", "b", "a", "b", "a", "b"], n=2)
    assert low < high


def test_distinct_n_handles_short_sequences():
    assert distinct_n(["a"], n=2) == 0.0  # no bigrams possible


def test_max_repeated_substring_detects_loop():
    words = ["the", "cat", "sat"] * 5
    assert max_repeated_substring_len(words) >= 3


def test_max_repeated_substring_zero_for_no_repeats():
    words = ["a", "b", "c", "d", "e"]
    assert max_repeated_substring_len(words) == 0
