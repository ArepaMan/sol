"""Export a deployable weights-only bundle from a training checkpoint.

A training checkpoint is ~635 MB: fp32 weights plus AdamW's two moment buffers
plus RNG state, all of which exist to *resume training* and none of which
inference needs. This strips it to bf16 weights (~110 MB) and copies the three
small files needed to reconstruct the model: `config.yaml`, `tokenizer.json`,
`meta.json` (for `eot_id`).

The bundle is what gets uploaded to a Hugging Face **model** repo. It is not
committed to git (`.gitignore` excludes `*.pt`) and it is not in the Space repo
— the Space pulls it at startup with `hf_hub_download`.

Usage:
    python -m scripts.export_weights --out export/sol-001
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from src.config import load_config
from src.model import GPT
from src.utils import count_params, human_bytes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default="checkpoints/001_baseline/latest.pt")
    parser.add_argument("--config", default="configs/micro_50m_8gb.yaml")
    parser.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--meta", default="data/tokenized/meta.json")
    parser.add_argument("--out", default="export/sol-001")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"loaded {args.checkpoint} (iter {ckpt.get('iter_num', '?')})")

    # Round-trip through a real GPT so a shape mismatch fails here, at export
    # time, rather than in a Space cold start where the traceback is a log line
    # someone has to go find.
    model = GPT(cfg.model, gradient_checkpointing=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"params: {count_params(model):,}")

    # Preserve weight tying through the dtype cast. `transformer.wte.weight`
    # and `lm_head.weight` are one tensor (src/model.py), and torch.save
    # dedupes shared storage — but a naive `{k: v.to(bfloat16) ...}` allocates
    # two independent tensors and writes the 32000x592 embedding table twice.
    # Measured: 137 MiB naive vs 101 MiB with sharing kept, for identical
    # weights. Worth the four extra lines on a cold-start-sensitive download.
    converted: dict[int, torch.Tensor] = {}
    state = {}
    for k, v in model.state_dict().items():
        ptr = v.data_ptr()
        if ptr not in converted:
            converted[ptr] = v.to(torch.bfloat16)
        state[k] = converted[ptr]
    torch.save(state, out / "model.pt")

    shutil.copy(args.config, out / "config.yaml")
    shutil.copy(args.tokenizer, out / "tokenizer.json")
    shutil.copy(args.meta, out / "meta.json")

    manifest = {
        "source_checkpoint": str(args.checkpoint),
        "iter_num": ckpt.get("iter_num"),
        "best_val_loss": float(ckpt["best_val_loss"]) if "best_val_loss" in ckpt else None,
        "params": count_params(model),
        "dtype": "bfloat16",
        "files": sorted(p.name for p in out.iterdir()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for p in sorted(out.iterdir()):
        print(f"  {p.name:<16} {human_bytes(p.stat().st_size)}")
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"bundle: {human_bytes(total)} -> {out}")


if __name__ == "__main__":
    main()
