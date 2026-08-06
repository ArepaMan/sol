"""Full-val-set perplexity evaluation: Sol-001 vs uniform/unigram/trigram
baselines, each with a document-level bootstrap 95% CI and a per-length-bucket
breakdown.

Usage:
    python -m src.eval --checkpoint checkpoints/001_baseline/latest.pt

Writes eval/results.json (machine-readable) and eval/results.md (the table
this milestone's exit criterion asks for).

Design notes
------------
- **Unit of evaluation is the document, not the token window.** Val docs were
  tokenized with a trailing `<|endoftext|>` (see `data/tokenize.py`) and are
  split back apart on that id here, so each document is scored in isolation
  (no cross-document context bleed) — matching how a real generation prompt
  would be evaluated, and matching the "bootstrap over documents" wording in
  `docs/ROADMAP.md`'s M6 section.
- **Bootstrap resamples documents, not tokens.** Each replicate draws
  len(docs) documents with replacement, takes the *token-count-weighted* mean
  NLL of the resample (long documents should not count the same as short
  ones), and exponentiates to a perplexity. The reported CI is the 2.5/97.5
  percentile of that replicate distribution.
- Documents longer than `block_size` are truncated (logged) — same
  constraint the model was trained under.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.baselines import TrigramBackoffBaseline, UnigramBaseline, UniformBaseline
from src.config import load_config
from src.model import GPT
from src.utils import get_device, set_seed

LENGTH_BUCKETS = [(1, 64), (65, 128), (129, 256), (257, 512), (513, None)]


def load_documents(bin_path: Path, eot_id: int, block_size: int) -> tuple[list[np.ndarray], int]:
    """Split a tokenized .bin shard back into per-document arrays on eot_id.
    Each returned document *includes* its trailing EOT (the model was trained
    to predict it, so scoring should include that prediction too). Documents
    longer than block_size are truncated to block_size; returns the count of
    truncated docs alongside the list."""
    arr = np.memmap(bin_path, dtype=np.uint16, mode="r").astype(np.int64)
    boundaries = np.flatnonzero(arr == eot_id)
    docs = []
    start = 0
    n_truncated = 0
    for b in boundaries:
        doc = arr[start : b + 1]
        start = b + 1
        if len(doc) < 2:
            continue  # can't form a single (input, target) pair
        if len(doc) > block_size:
            doc = doc[:block_size]
            n_truncated += 1
        docs.append(doc)
    return docs, n_truncated


@torch.no_grad()
def model_doc_nlls(
    model: GPT,
    docs: list[np.ndarray],
    device: torch.device,
    ptdtype: torch.dtype,
    batch_size: int = 64,
) -> list[np.ndarray]:
    """Per-document, per-token NLL arrays (natural log) from the model.

    Docs are sorted by length and batched together, right-padded with a dummy
    id to the batch's max length. Causal attention only ever looks backward,
    so right-padding cannot influence any real token's prediction — padded
    positions just get masked out of the loss before per-document NLLs are
    sliced back out.
    """
    model.eval()
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
    results: list[np.ndarray | None] = [None] * len(docs)

    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        batch_docs = [docs[i] for i in batch_idx]
        max_len = max(len(d) for d in batch_docs)

        inp = np.zeros((len(batch_docs), max_len - 1), dtype=np.int64)
        tgt = np.full((len(batch_docs), max_len - 1), -1, dtype=np.int64)  # -1 = ignore_index
        for row, d in enumerate(batch_docs):
            L = len(d) - 1
            inp[row, :L] = d[:-1]
            tgt[row, :L] = d[1:]

        x = torch.from_numpy(inp).to(device)
        y = torch.from_numpy(tgt).to(device)
        with torch.autocast(device_type=device.type, dtype=ptdtype, enabled=device.type == "cuda"):
            logits, _ = model(x)
        per_token_nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1, reduction="none"
        ).reshape(y.shape).float()

        for row, doc_idx in enumerate(batch_idx):
            L = len(batch_docs[row]) - 1
            results[doc_idx] = per_token_nll[row, :L].cpu().numpy()

    model.train()
    return results  # type: ignore[return-value]


def bootstrap_ppl_ci(
    doc_nlls: list[np.ndarray], n_boot: int = 10_000, seed: int = 42
) -> tuple[float, float, float]:
    """Document-level bootstrap: resample documents with replacement,
    token-count-weighted mean NLL per replicate, exponentiate. Returns
    (point_estimate_ppl, ci_lo, ci_hi) from the 2.5/97.5 percentiles."""
    rng = np.random.default_rng(seed)
    sums = np.array([d.sum() for d in doc_nlls])
    counts = np.array([len(d) for d in doc_nlls])
    n_docs = len(doc_nlls)

    point = np.exp(sums.sum() / counts.sum())

    replicate_ppls = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_docs, size=n_docs)
        replicate_ppls[b] = np.exp(sums[idx].sum() / counts[idx].sum())

    lo, hi = np.percentile(replicate_ppls, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def bucket_perplexity(doc_nlls: list[np.ndarray], doc_lens: list[int]) -> dict[str, dict]:
    out = {}
    for lo, hi in LENGTH_BUCKETS:
        label = f"{lo}-{hi}" if hi is not None else f"{lo}+"
        mask = [(lo <= n and (hi is None or n <= hi)) for n in doc_lens]
        if not any(mask):
            out[label] = {"n_docs": 0, "ppl": None}
            continue
        sel = [d for d, m in zip(doc_nlls, mask) if m]
        total_nll = sum(d.sum() for d in sel)
        total_tok = sum(len(d) for d in sel)
        out[label] = {"n_docs": sum(mask), "ppl": float(np.exp(total_nll / total_tok))}
    return out


def evaluate_all(
    checkpoint: Path,
    config_path: str,
    data_dir: str,
    n_boot: int,
    seed: int,
    train_tokens_for_baselines: int,
    batch_size: int,
) -> dict:
    cfg = load_config(config_path)
    device = get_device()
    set_seed(seed)
    ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.precision]

    data_dir = Path(data_dir)
    with (data_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    eot_id = meta["eot_id"]
    vocab_size = meta["vocab_size"]

    val_docs, n_truncated = load_documents(data_dir / "val.bin", eot_id, cfg.model.block_size)
    doc_lens = [len(d) for d in val_docs]
    print(f"val: {len(val_docs)} docs, {sum(doc_lens):,} tokens, {n_truncated} truncated to block_size")

    results: dict = {
        "checkpoint": str(checkpoint),
        "n_val_docs": len(val_docs),
        "n_val_tokens": int(sum(doc_lens)),
        "n_truncated_docs": n_truncated,
        "n_boot": n_boot,
        "models": {},
    }

    # --- Sol-001 ---
    model = GPT(cfg.model, gradient_checkpointing=False).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {checkpoint} (iter {ckpt.get('iter_num', '?')})")

    model_nlls = model_doc_nlls(model, val_docs, device, ptdtype, batch_size=batch_size)
    point, lo, hi = bootstrap_ppl_ci(model_nlls, n_boot=n_boot, seed=seed)
    results["models"]["sol-001"] = {
        "ppl": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "buckets": bucket_perplexity(model_nlls, doc_lens),
    }
    print(f"sol-001: ppl={point:.3f} 95% CI [{lo:.3f}, {hi:.3f}]")

    # --- Baselines: fit on a train prefix, evaluate on the same val docs ---
    train_arr = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    train_tokens = train_arr[:train_tokens_for_baselines].astype(np.int64)
    print(f"fitting baselines on {len(train_tokens):,} train tokens")

    baselines = {
        "uniform": UniformBaseline(vocab_size),
        "unigram": UnigramBaseline(vocab_size).fit(train_tokens),
        "trigram": TrigramBackoffBaseline(vocab_size).fit(train_tokens),
    }
    results["baseline_train_tokens"] = int(len(train_tokens))

    for name, baseline in baselines.items():
        doc_nlls = [baseline.nll(d) for d in val_docs]
        point, lo, hi = bootstrap_ppl_ci(doc_nlls, n_boot=n_boot, seed=seed)
        results["models"][name] = {
            "ppl": point,
            "ci_lo": lo,
            "ci_hi": hi,
            "buckets": bucket_perplexity(doc_nlls, doc_lens),
        }
        print(f"{name}: ppl={point:.3f} 95% CI [{lo:.3f}, {hi:.3f}]")

    return results


def write_results_md(results: dict, out_path: Path) -> None:
    lines = [
        "# M6 — Evaluation results\n",
        f"Val set: {results['n_val_docs']:,} documents, {results['n_val_tokens']:,} tokens "
        f"({results['n_truncated_docs']} truncated to block_size). "
        f"Baselines fit on the first {results['baseline_train_tokens']:,} train tokens "
        f"(see `src/baselines.py` for why that's a prefix, not the full corpus). "
        f"95% CI from a {results['n_boot']:,}-replicate document-level bootstrap.\n",
        "| Model | Perplexity | 95% CI |",
        "|---|---|---|",
    ]
    order = ["uniform", "unigram", "trigram", "sol-001"]
    for name in order:
        m = results["models"][name]
        lines.append(f"| {name} | {m['ppl']:.3f} | [{m['ci_lo']:.3f}, {m['ci_hi']:.3f}] |")

    lines.append("\n## Per-length-bucket perplexity (Sol-001)\n")
    lines.append("| Bucket (tokens) | n docs | Perplexity |")
    lines.append("|---|---|---|")
    for label, b in results["models"]["sol-001"]["buckets"].items():
        ppl_str = f"{b['ppl']:.3f}" if b["ppl"] is not None else "—"
        lines.append(f"| {label} | {b['n_docs']} | {ppl_str} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/001_baseline/latest.pt")
    parser.add_argument("--config", default="configs/micro_50m_8gb.yaml")
    parser.add_argument("--data-dir", default="data/tokenized")
    parser.add_argument("--out-dir", default="eval")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-tokens", type=int, default=10_000_000, help="Train token prefix for baseline fitting.")
    parser.add_argument("--batch-size", type=int, default=4, help="Model eval batch size (docs per forward pass). The 32k-vocab logits tensor dominates memory (batch x seq_len x 32000, promoted to float32 inside cross_entropy): measured 64 and 16 both OOM the 8GB card here, 4 does not — matches the training config's own micro-batch size, not a coincidence.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    results = evaluate_all(
        checkpoint=Path(args.checkpoint),
        config_path=args.config,
        data_dir=args.data_dir,
        n_boot=args.n_boot,
        seed=args.seed,
        train_tokens_for_baselines=args.train_tokens,
        batch_size=args.batch_size,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_results_md(results, out_dir / "results.md")
    print(f"wrote {out_dir}/results.json and {out_dir}/results.md")


if __name__ == "__main__":
    main()
