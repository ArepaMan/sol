"""Generate completions for `eval/prompts.jsonl` and score them automatically
for repetition (distinct-2/3, max repeated substring) — the objective
backstop for `docs/RUBRIC.md`'s manual scoring, per M6's risk note ("rubric
theatre... keep the automatic metrics as the objective backstop").

Usage:
    python -m src.generate_samples --checkpoint checkpoints/001_baseline/latest.pt

Writes eval/generations.jsonl (one row per prompt: text, tokens, metrics) and
eval/repetition_summary.md (aggregate distinct-n / repeated-substring stats).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.config import load_config
from src.model import GPT
from src.utils import get_device, set_seed


def distinct_n(words: list[str], n: int) -> float:
    """Fraction of unique n-grams among all n-grams (Li et al. 2016). 1.0 =
    no repeated n-grams at all; near 0 = heavily repetitive. Undefined (0.0)
    for sequences shorter than n."""
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def max_repeated_substring_len(words: list[str]) -> int:
    """Length (in words) of the longest contiguous word sequence that occurs
    at least twice in the text. O(n^2) pairwise scan — fine for the short
    (<=200 word) generations this project produces; would need a suffix
    automaton for anything longer."""
    n = len(words)
    best = 0
    for i in range(n):
        for j in range(i + 1, n):
            length = 0
            while j + length < n and words[i + length] == words[j + length]:
                length += 1
            best = max(best, length)
    return best


@torch.no_grad()
def generate_one(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/001_baseline/latest.pt")
    parser.add_argument("--config", default="configs/micro_50m_8gb.yaml")
    parser.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--prompts", default="eval/prompts.jsonl")
    parser.add_argument("--out-dir", default="eval")
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    device = get_device()
    set_seed(args.seed)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = GPT(cfg.model, gradient_checkpointing=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (iter {ckpt.get('iter_num', '?')})")

    prompts = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generations = []
    for row in prompts:
        text = generate_one(
            model, tokenizer, row["prompt"], args.max_new_tokens, args.temperature, args.top_k, device
        )
        words = text.split()
        rec = {
            "id": row["id"],
            "category": row["category"],
            "prompt": row["prompt"],
            "generation": text,
            "distinct_2": distinct_n(words, 2),
            "distinct_3": distinct_n(words, 3),
            "max_repeated_substring": max_repeated_substring_len(words),
            "n_words": len(words),
        }
        generations.append(rec)
        print(f"[{row['id']}] ({row['category']}) d2={rec['distinct_2']:.2f} d3={rec['distinct_3']:.2f} "
              f"rep={rec['max_repeated_substring']}")

    with (out_dir / "generations.jsonl").open("w", encoding="utf-8") as f:
        for rec in generations:
            f.write(json.dumps(rec) + "\n")

    # Aggregate summary, overall and per category.
    def _agg(rows: list[dict]) -> dict:
        n = len(rows)
        return {
            "n": n,
            "distinct_2_mean": sum(r["distinct_2"] for r in rows) / n,
            "distinct_3_mean": sum(r["distinct_3"] for r in rows) / n,
            "max_repeated_substring_mean": sum(r["max_repeated_substring"] for r in rows) / n,
            "max_repeated_substring_max": max(r["max_repeated_substring"] for r in rows),
        }

    categories = sorted({r["category"] for r in generations})
    summary_lines = [
        "# M6 — Repetition metrics (automatic backstop for the manual rubric)\n",
        f"{len(generations)} generations, {args.max_new_tokens} max new tokens, "
        f"temperature={args.temperature}, top_k={args.top_k}.\n",
        "| Category | n | distinct-2 | distinct-3 | max repeated substring (mean / max) |",
        "|---|---|---|---|---|",
    ]
    overall = _agg(generations)
    for cat in categories:
        rows = [r for r in generations if r["category"] == cat]
        a = _agg(rows)
        summary_lines.append(
            f"| {cat} | {a['n']} | {a['distinct_2_mean']:.3f} | {a['distinct_3_mean']:.3f} | "
            f"{a['max_repeated_substring_mean']:.1f} / {a['max_repeated_substring_max']} |"
        )
    summary_lines.append(
        f"| **all** | {overall['n']} | {overall['distinct_2_mean']:.3f} | {overall['distinct_3_mean']:.3f} | "
        f"{overall['max_repeated_substring_mean']:.1f} / {overall['max_repeated_substring_max']} |"
    )
    (out_dir / "repetition_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}/generations.jsonl and {out_dir}/repetition_summary.md")


if __name__ == "__main__":
    main()
