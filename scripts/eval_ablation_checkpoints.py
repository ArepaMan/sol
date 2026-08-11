"""M7: evaluate all 6 ablation checkpoints on the full val set, reusing
src/eval.py's document-level perplexity + bootstrap-CI machinery.

Baselines (uniform/unigram/trigram) are NOT recomputed here — they don't
depend on which Sol checkpoint is being scored, and M6 already measured them
against the same val set (eval/results.json). This script only adds the six
ablation checkpoints' perplexity + CI, all directly comparable to each other
and to Sol-001 (baseline run, ppl 3.719) since they share the same eval
harness, val set, and bootstrap procedure.

Usage:
    python -m scripts.eval_ablation_checkpoints
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.config import load_config
from src.eval import bootstrap_ppl_ci, load_documents, model_doc_nlls
from src.model import GPT
from src.utils import get_device, set_seed

# (run_name, config_path, checkpoint_path)
RUNS = [
    ("002_lr_1e-4", "configs/ablations/002_lr_1e-4.yaml", "checkpoints/ablation_002_lr_1e-4/latest.pt"),
    ("002_lr_3e-4", "configs/ablations/002_lr_3e-4.yaml", "checkpoints/ablation_002_lr_3e-4/latest.pt"),
    ("002_lr_1e-3", "configs/ablations/002_lr_1e-3.yaml", "checkpoints/ablation_002_lr_1e-3/latest.pt"),
    ("003_data_100m", "configs/ablations/003_data_100m.yaml", "checkpoints/ablation_003_data_100m/latest.pt"),
    ("004_seed_43", "configs/ablations/004_seed_43.yaml", "checkpoints/ablation_004_seed_43/latest.pt"),
    ("004_seed_44", "configs/ablations/004_seed_44.yaml", "checkpoints/ablation_004_seed_44/latest.pt"),
]

DATA_DIR = Path("data/tokenized")
BATCH_SIZE = 4  # M6 finding: 16/64 OOM the 8GB card on full-val-set eval; 4 matches training's micro-batch
N_BOOT = 10_000
SEED = 42


def main() -> None:
    device = get_device()
    set_seed(SEED)

    with (DATA_DIR / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    eot_id = meta["eot_id"]

    # block_size is identical (512) across every ablation config, so the
    # documents only need loading once.
    first_cfg = load_config(RUNS[0][1])
    val_docs, n_truncated = load_documents(DATA_DIR / "val.bin", eot_id, first_cfg.model.block_size)
    doc_lens = [len(d) for d in val_docs]
    print(f"val: {len(val_docs)} docs, {sum(doc_lens):,} tokens, {n_truncated} truncated")

    results: dict[str, dict] = {}
    for name, config_path, ckpt_path in RUNS:
        cfg = load_config(config_path)
        ptdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.precision]

        model = GPT(cfg.model, gradient_checkpointing=False).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        iter_num = ckpt.get("iter_num")

        doc_nlls = model_doc_nlls(model, val_docs, device, ptdtype, batch_size=BATCH_SIZE)
        point, lo, hi = bootstrap_ppl_ci(doc_nlls, n_boot=N_BOOT, seed=SEED)
        results[name] = {
            "checkpoint": ckpt_path,
            "iter_num": iter_num,
            "ppl": point,
            "ci_lo": lo,
            "ci_hi": hi,
        }
        print(f"{name} (iter {iter_num}): ppl={point:.4f} 95% CI [{lo:.4f}, {hi:.4f}]")

        del model, ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = Path("experiments/ablation_eval_results.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
