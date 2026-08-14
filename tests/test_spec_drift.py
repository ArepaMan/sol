"""The test that makes `docs/spec.json` a contract rather than a snapshot.

M9 exists because Sol's numbers lived in three places and drifted: the
portfolio claimed "RTX 2070 Super", "float16", "n_embd 512" and "target
perplexity 15-25" long after every one of those was false. Nothing was lying —
each was true when written, and nothing forced it to stay true.

Generation alone doesn't fix that; generation plus a failing test does. Edit a
config without regenerating and this suite goes red.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.export_spec import build_spec, render_ts

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def spec() -> dict:
    return build_spec()


def test_committed_spec_json_matches_regeneration(spec: dict):
    committed = json.loads((REPO_ROOT / "docs" / "spec.json").read_text(encoding="utf-8"))
    assert committed == spec, (
        "docs/spec.json is stale — run `python -m scripts.export_spec`. "
        "Something upstream (a config, an eval result) changed without the spec "
        "being regenerated, which is exactly the drift this test exists to catch."
    )


def test_committed_typescript_matches_regeneration(spec: dict):
    committed = (REPO_ROOT / "docs" / "sol-spec.ts").read_text(encoding="utf-8")
    assert committed == render_ts(spec), (
        "docs/sol-spec.ts is stale — run `python -m scripts.export_spec`."
    )


# ---------------------------------------------------------------------------
# Guards against the specific wrong values that were actually published.
# These are not redundant with the equality tests above: those catch "you
# forgot to regenerate", these catch "you regenerated from a config someone
# reverted to the old wrong numbers".
# ---------------------------------------------------------------------------

def test_hardware_is_the_machine_this_actually_ran_on(spec: dict):
    assert "4070" in spec["hardware"]["gpu"]
    assert "2070" not in spec["hardware"]["gpu"]


def test_precision_is_bf16_not_fp16(spec: dict):
    # Ada supports bf16; the fp16 claim came from the 2070-era draft and
    # implies GradScaler and loss-scale debugging that this project never did.
    assert spec["training"]["precision"] == "bfloat16"


def test_param_count_is_within_two_percent_of_the_52m_claim(spec: dict):
    # n_embd=512 measured 41.8M — a ~20% gap from the project's own branding.
    # 592 lands within 1.73%. This is the test that caught it.
    assert abs(spec["params"] - 52_000_000) / 52_000_000 < 0.02


def test_embedding_width_is_the_corrected_value(spec: dict):
    assert spec["architecture"]["n_embd"] == 592
    assert spec["architecture"]["n_embd"] % spec["architecture"]["n_head"] == 0


def test_perplexity_is_the_measured_value_not_the_original_target(spec: dict):
    # The original spec targeted 15-25. Measured is 3.719 — the target was
    # simply wrong about how hard TinyStories is, and the spec was corrected
    # to reality rather than the result being framed as a triumph.
    ppl = spec["evaluation"]["perplexity"]
    assert 3.0 < ppl < 4.5
    assert spec["evaluation"]["ci_lo"] < ppl < spec["evaluation"]["ci_hi"]


def test_perplexity_beats_every_baseline(spec: dict):
    ppl = spec["evaluation"]["perplexity"]
    for name, baseline_ppl in spec["evaluation"]["baselines"].items():
        assert ppl < baseline_ppl, f"sol-001 should beat the {name} baseline"


def test_ablation_gaps_are_reported_against_seed_noise(spec: dict):
    """The M7 methodology in one assertion: the LR effect must be large
    relative to seed variance, or the sweep proved nothing."""
    ab = spec["ablations"]
    sd = ab["seed_variance_sd"]
    assert sd > 0, "seed variance is the yardstick; zero would mean it wasn't measured"
    lr_gap = ab["learning_rate"]["1e-4"] - ab["learning_rate"]["1e-3"]
    assert lr_gap > 20 * sd


def test_deployment_urls_are_live_not_placeholders(spec: dict):
    dep = spec["deployment"]
    for key in ("live_url", "weights_url", "repo_url"):
        assert dep[key].startswith("https://"), key
    assert "coming-soon" not in json.dumps(dep)


def test_kv_cache_speedups_carry_a_real_range(spec: dict):
    """M9 QA's mistake, encoded as a gate: a range needs more than one sample.
    Every before/after cell must carry n>=5, a range that is actually a range,
    and a mean that falls inside it — so a future edit can't quietly publish a
    point estimate dressed up as a spread."""
    kv = spec["kv_cache"]
    local_n = kv["samples_per_cell"]
    cells = [
        ("cpu_before", local_n),
        ("cpu_after", local_n),
        ("gpu_before", local_n),
        ("gpu_after", local_n),
        ("deployed_before", kv["deployed_before_samples"]),
        ("deployed_after", kv["deployed_after_samples"]),
    ]
    for cell, n in cells:
        assert n >= 5, f"{cell}: n={n} is too few to quote a range"
        lo, hi = kv[f"{cell}_range"]
        assert lo < hi, f"{cell}: a single sample cannot establish a range"
        assert lo <= kv[cell] <= hi, f"{cell}: mean {kv[cell]} sits outside {lo}-{hi}"


def test_kv_cache_deployed_figure_is_the_cpu_one(spec: dict):
    """The deployed app is CPU-only, so its throughput claim has to track the
    CPU measurement, not the GPU one. Guards against someone updating the
    headline number from the wrong row of the table."""
    assert spec["deployment"]["tokens_per_second_local_cpu"] == spec["kv_cache"]["cpu_after"]
    assert spec["deployment"]["tokens_per_second_local_gpu"] == spec["kv_cache"]["gpu_after"]
    # The live figure and the KV-cache table's deployed cell are the same
    # measurement; they must not drift into two different published numbers.
    assert spec["deployment"]["tokens_per_second_deployed"] == spec["kv_cache"]["deployed_after"]


def test_deployed_speedup_is_flagged_as_not_a_controlled_comparison(spec: dict):
    """The deployed before/after spans two sessions on a shared host — the app
    has no `--no-kv-cache` switch to interleave against, unlike the two local
    rows. That weakness rides along with the number in the spec itself, so
    anyone consuming the 1.8× downstream gets the caveat and not just the
    ratio."""
    kv = spec["kv_cache"]
    assert kv["deployed_before_is_same_session_control"] is False
    # The claim that survives the caveat: the two ranges don't overlap.
    assert kv["deployed_after_range"][0] > kv["deployed_before_range"][1]


def test_max_train_tokens_matches_the_measured_corpus(spec: dict):
    # The original 400M target was lowered to the measured 357,852,786.
    assert spec["data"]["max_train_tokens"] == spec["data"]["train_tokens"]
