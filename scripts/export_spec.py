"""Generate the single source of truth for Sol's numbers.

The problem this solves: Sol's numbers were living in three places that drift —
this repo's YAML, the portfolio's `sol.mdx`, and the portfolio's
`sol-spec-explorer.tsx` (three hardcoded arrays). By M8 the portfolio still said
"RTX 2070 Super", "float16", "n_embd 512", "target perplexity 15-25" and
"not started". Every one of those was wrong, and none of them was wrong on
purpose — they were true when written and nothing forced them to stay true.

This script reads the *artifacts that produced* each number and emits:

  docs/spec.json    machine-readable, committed, diffable
  docs/sol-spec.ts  typed TS for the portfolio to import instead of retyping

`tests/test_spec_drift.py` regenerates in memory and asserts equality with the
committed `spec.json`, so editing a config without regenerating fails CI rather
than silently publishing a stale claim.

Sources, all real artifacts rather than transcribed values:

  configs/micro_50m_8gb.yaml      architecture + training hyperparameters
  src/model.py (instantiated)     parameter count — counted, never estimated
  data/processed/stats.json       dedup funnel, document counts
  data/tokenized/meta.json        token counts, vocab, eot id
  eval/results.json               M6 perplexity + CIs + baselines
  experiments/ablation_eval_results.json   M7 ablation perplexities

Usage:
    python -m scripts.export_spec
    python -m scripts.export_spec --check    # exit 1 if the committed files are stale
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from src.config import load_config
from src.model import GPT
from src.utils import count_params

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = "configs/micro_50m_8gb.yaml"

# --- Facts with no other machine-readable home -------------------------------
# Deployment and hardware numbers aren't derivable from any artifact in the
# repo, so this file *is* their source of truth. They are measured, not
# guessed; each carries the doc that records how it was measured. Anything
# that could be derived from an artifact is derived below instead.
HARDWARE = {
    "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU",
    "architecture": "Ada Lovelace (sm_89)",
    "vram_gb": 8,
    "note": "An earlier spec targeted an RTX 2070 Super and float16. Wrong machine — see AGENTS.md.",
}

DEPLOYMENT = {
    "live_url": "https://sol-52m.streamlit.app",
    "weights_url": "https://huggingface.co/SpicyGuac/sol-001",
    "repo_url": "https://github.com/ArepaMan/sol",
    "host": "Streamlit Community Cloud (free tier)",
    "host_ram_limit_mb": 1024,
    "model_load_seconds": 6.7,
    # Mean of 7 generations on the live app. The single-sample 16.0 recorded at
    # deploy time happened to land exactly on the mean; QA's 6 further runs
    # showed the spread is wider than the 15-25 originally predicted, and lower:
    # 12.0-20.0. Recorded as a range because a point estimate oversold it.
    "tokens_per_second_deployed": 16.0,
    "tokens_per_second_range": [12.0, 20.0],
    "tokens_per_second_samples": 7,
    "tokens_per_second_local_cpu": 24.6,
    "tokens_per_second_local_gpu": 90.0,
    "bundle_mib": 103.1,
    "source": "docs/DEPLOY.md",
}

TRAINING_RUN = {
    "wall_clock_hours": 30.3,
    "training_hours": 16.9,
    "asleep_hours": 12.17,
    "throughput_tokens_per_second": 20067,
    "peak_vram_mib": 2344,
    "source": "experiments/001_baseline/run.md",
}

RUBRIC = {
    "n": 60,
    "grammar": 4.00,
    "coherence": 3.15,
    "on_topic_in_domain": 5.00,
    "on_topic_out_of_domain": 1.47,
    "repetition": 3.98,
    "source": "eval/rubric_scores.csv, docs/RUBRIC.md",
}


def _read_json(rel: str) -> dict[str, Any]:
    with (REPO_ROOT / rel).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_spec() -> dict[str, Any]:
    cfg = load_config(REPO_ROOT / CONFIG_PATH)
    stats = _read_json("data/processed/stats.json")
    meta = _read_json("data/tokenized/meta.json")
    evals = _read_json("eval/results.json")
    ablations = _read_json("experiments/ablation_eval_results.json")

    # Counted from a real instantiation, not arithmetic on the config — this is
    # the number that caught n_embd=512 measuring 41.8M against a "~52M" claim.
    model = GPT(cfg.model, gradient_checkpointing=False)
    n_params = count_params(model)

    seed_ppls = [
        ablations["002_lr_3e-4"]["ppl"],  # the seed-42 arm, shared with the LR sweep
        ablations["004_seed_43"]["ppl"],
        ablations["004_seed_44"]["ppl"],
    ]
    seed_sd = statistics.stdev(seed_ppls)

    return {
        "_generated_by": "scripts/export_spec.py — do not edit by hand",
        "name": cfg.name,
        "params": n_params,
        "hardware": HARDWARE,
        "architecture": {
            "n_layer": cfg.model.n_layer,
            "n_head": cfg.model.n_head,
            "n_embd": cfg.model.n_embd,
            "head_dim": cfg.model.head_dim,
            "block_size": cfg.model.block_size,
            "vocab_size": cfg.model.vocab_size,
            "bias": cfg.model.bias,
            "dropout": cfg.model.dropout,
            "weight_tying": True,
        },
        "training": {
            "batch_size": cfg.training.batch_size,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "tokens_per_step": cfg.tokens_per_step,
            "max_iters": cfg.training.max_iters,
            "total_train_tokens": cfg.total_train_tokens,
            "epochs": round(cfg.epochs, 2),
            "learning_rate": cfg.training.learning_rate,
            "min_lr": cfg.training.min_lr,
            "warmup_iters": cfg.training.warmup_iters,
            "weight_decay": cfg.training.weight_decay,
            "grad_clip": cfg.training.grad_clip,
            "precision": cfg.precision,
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "seed": cfg.seed,
            **{k: v for k, v in TRAINING_RUN.items() if k != "source"},
        },
        "data": {
            "dataset": stats["dataset"],
            "train_docs": stats["train"]["kept"],
            "train_tokens": meta["splits"]["train"]["token_count"],
            "val_docs": stats["validation"]["kept"],
            "val_tokens": meta["splits"]["val"]["token_count"],
            "train_exact_dup_rate": stats["train"]["exact_dup_rate"],
            "val_cross_split_leakage_dropped": stats["validation"]["dropped_cross_split_leakage"],
            "vocab_size": meta["vocab_size"],
            "max_train_tokens": cfg.data.max_train_tokens,
        },
        "evaluation": {
            "n_val_docs": evals["n_val_docs"],
            "n_val_tokens": evals["n_val_tokens"],
            "n_boot": evals["n_boot"],
            "perplexity": round(evals["models"]["sol-001"]["ppl"], 3),
            "ci_lo": round(evals["models"]["sol-001"]["ci_lo"], 3),
            "ci_hi": round(evals["models"]["sol-001"]["ci_hi"], 3),
            "baselines": {
                name: round(evals["models"][name]["ppl"], 1)
                for name in ("trigram", "unigram", "uniform")
                if name in evals["models"]
            },
            "rubric": {k: v for k, v in RUBRIC.items() if k != "source"},
        },
        "ablations": {
            "budget_iters": ablations["002_lr_3e-4"]["iter_num"],
            "seed_variance_sd": round(seed_sd, 4),
            "seed_variance_mean": round(statistics.fmean(seed_ppls), 3),
            "learning_rate": {
                "1e-4": round(ablations["002_lr_1e-4"]["ppl"], 3),
                "3e-4": round(ablations["002_lr_3e-4"]["ppl"], 3),
                "1e-3": round(ablations["002_lr_1e-3"]["ppl"], 3),
            },
            "data_scale": {
                "100m": round(ablations["003_data_100m"]["ppl"], 3),
                "full": round(ablations["002_lr_3e-4"]["ppl"], 3),
            },
        },
        "deployment": {k: v for k, v in DEPLOYMENT.items() if k != "source"},
    }


def _ts_literal(value: Any, indent: int = 2) -> str:
    """Render a Python value as a TypeScript literal."""
    pad = " " * indent
    if isinstance(value, dict):
        inner = "".join(
            f"{pad}  {json.dumps(k)}: {_ts_literal(v, indent + 2)},\n" for k, v in value.items()
        )
        return "{\n" + inner + pad + "}"
    if isinstance(value, list):
        inner = "".join(f"{pad}  {_ts_literal(v, indent + 2)},\n" for v in value)
        return "[\n" + inner + pad + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int | float):
        return repr(value)
    return json.dumps(value)


def render_ts(spec: dict[str, Any]) -> str:
    return (
        "// GENERATED by scripts/export_spec.py in the Sol repo. Do not edit.\n"
        "//\n"
        "// Sol's numbers used to live in three places (this file's hardcoded\n"
        "// ancestor, sol.mdx, and the Sol repo's YAML) and drifted apart: this\n"
        "// component claimed an RTX 2070 Super, float16, n_embd 512 and a target\n"
        "// perplexity of 15-25 long after all four were false. Regenerate with\n"
        "//   python -m scripts.export_spec\n"
        "// and copy the output here. tests/test_spec_drift.py enforces freshness.\n"
        "\n"
        "export const solSpec = " + _ts_literal(spec) + " as const;\n"
        "\n"
        "export type SolSpec = typeof solSpec;\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed files are current; write nothing, exit 1 if stale",
    )
    args = parser.parse_args(argv)

    spec = build_spec()
    json_text = json.dumps(spec, indent=2) + "\n"
    ts_text = render_ts(spec)

    json_path = REPO_ROOT / "docs" / "spec.json"
    ts_path = REPO_ROOT / "docs" / "sol-spec.ts"

    if args.check:
        stale = [
            p.name
            for p, expected in ((json_path, json_text), (ts_path, ts_text))
            if not p.exists() or p.read_text(encoding="utf-8") != expected
        ]
        if stale:
            print(f"STALE: {', '.join(stale)} — run `python -m scripts.export_spec`", file=sys.stderr)
            return 1
        print("spec is current")
        return 0

    json_path.write_text(json_text, encoding="utf-8")
    ts_path.write_text(ts_text, encoding="utf-8")
    print(f"wrote {json_path.relative_to(REPO_ROOT)} and {ts_path.relative_to(REPO_ROOT)}")
    print(f"  params {spec['params']:,} · ppl {spec['evaluation']['perplexity']} · live {spec['deployment']['live_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
