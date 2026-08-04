"""Training loop for Sol.

Usage:
    # Gate 1 — overfit a single batch. Loss must reach < 0.1 within 200 iters.
    python -m src.train --config configs/micro_50m_8gb.yaml --overfit-batch --max-iters 200 --no-wandb

    # Gate 3 — short run, then verify resume.
    python -m src.train --config configs/micro_50m_8gb.yaml --max-iters 500 --run-name gate3 --no-wandb
    python -m src.train --config configs/micro_50m_8gb.yaml --max-iters 1000 --run-name gate3 --resume --no-wandb

    # A real run.
    python -m src.train --config configs/micro_50m_8gb.yaml --run-name 001_baseline

No GradScaler anywhere in this file: BF16 (this project's precision — see
configs/micro_50m_8gb.yaml) keeps FP32's exponent range, so it doesn't need
the loss-scaling FP16 requires. That's not an oversight; it's the point of
having picked BF16.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.config import SolConfig, load_config
from src.data import BinDataset
from src.model import GPT
from src.utils import count_params, describe_device, get_device, human_count, set_seed

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def get_lr(
    it: int,
    warmup_iters: int,
    lr_decay_iters: int,
    learning_rate: float,
    min_lr: float,
) -> float:
    """Linear warmup, then cosine decay to min_lr. Pure function — unit-tested
    in tests/test_train.py without needing a GPU, and read directly by the
    training loop so there is exactly one place this schedule is defined."""
    if warmup_iters > 0 and it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it >= lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def _rng_state_dict(dataset: BinDataset) -> dict:
    """Every RNG this run's reproducibility depends on. python/numpy-global
    are seeded once via set_seed() and rarely drift, but capturing them costs
    nothing and closes the gap for anything that happens to touch them."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "dataset": dataset.get_rng_state(),
    }


def _restore_rng_state(state: dict, dataset: BinDataset) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.set_rng_state requires a CPU ByteTensor specifically. load_checkpoint
    # calls torch.load(..., map_location=device) so the whole checkpoint's
    # tensors load onto that device — including this one, if not corrected
    # here — but the RNG state tensor isn't real "data": moving it to CUDA
    # makes torch.set_rng_state reject it outright.
    torch.set_rng_state(state["torch"].cpu())
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in state["torch_cuda"]])
    dataset.set_rng_state(state["dataset"])


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    best_val_loss: float,
    cfg: SolConfig,
    dataset: BinDataset,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "config_name": cfg.name,
            "rng_state": _rng_state_dict(dataset),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    dataset: BinDataset,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    _restore_rng_state(ckpt["rng_state"], dataset)
    return ckpt["iter_num"], ckpt["best_val_loss"]


@torch.no_grad()
def estimate_loss(
    model: GPT,
    dataset: BinDataset,
    splits: list[str],
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
    ptdtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    out = {}
    for split in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dataset.get_batch(split, batch_size, block_size, device)
            with torch.autocast(device_type=device.type, dtype=ptdtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/micro_50m_8gb.yaml")
    parser.add_argument("--data-dir", default="data/tokenized")
    parser.add_argument("--run-name", default=None, help="Defaults to the config's `name` field.")
    parser.add_argument("--max-iters", type=int, default=None, help="Overrides config; also becomes lr_decay_iters unless --lr-decay-iters is set.")
    parser.add_argument("--lr-decay-iters", type=int, default=None)
    parser.add_argument("--overfit-batch", action="store_true", help="Train on one fixed batch on repeat — Gate 1.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--eval-iters", type=int, default=None, help="Overrides config's training.eval_iters (useful to shrink for quick gates).")
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.max_iters is not None:
        cfg.training.max_iters = args.max_iters
        # A shortened run should still complete its cosine decay to min_lr by
        # the end, not spend the whole run near peak LR — sync unless the
        # caller explicitly wants a different decay horizon.
        cfg.training.lr_decay_iters = args.lr_decay_iters or args.max_iters
    elif args.lr_decay_iters is not None:
        cfg.training.lr_decay_iters = args.lr_decay_iters
    if args.eval_iters is not None:
        cfg.training.eval_iters = args.eval_iters
    if args.eval_interval is not None:
        cfg.training.eval_interval = args.eval_interval
    if args.checkpoint_interval is not None:
        cfg.training.checkpoint_interval = args.checkpoint_interval
    if args.log_interval is not None:
        cfg.training.log_interval = args.log_interval

    device = get_device()
    set_seed(cfg.seed)
    ptdtype = _DTYPE_MAP[cfg.precision]

    run_name = args.run_name or cfg.name
    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {describe_device()}")
    print(f"run: {run_name}  precision: {cfg.precision}  max_iters: {cfg.training.max_iters}")

    dataset = BinDataset(args.data_dir, seed=cfg.seed)
    model = GPT(cfg.model, gradient_checkpointing=cfg.gradient_checkpointing).to(device)
    n_params = count_params(model)
    print(f"model: {human_count(n_params)} params ({n_params:,})")

    optimizer = model.configure_optimizers(
        weight_decay=cfg.training.weight_decay,
        learning_rate=cfg.training.learning_rate,
        betas=(0.9, 0.95),
        device_type=device.type,
    )

    iter_num = 0
    best_val_loss = float("inf")
    if args.resume:
        latest = ckpt_dir / "latest.pt"
        if not latest.exists():
            raise SystemExit(f"--resume given but {latest} doesn't exist")
        iter_num, best_val_loss = load_checkpoint(latest, model, optimizer, dataset, device)
        print(f"resumed from {latest} at iter {iter_num}, best_val_loss={best_val_loss:.4f}")

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(
            project="sol",
            name=run_name,
            config={"n_params": n_params, **dataclasses.asdict(cfg)},
        )

    fixed_batch = None
    if args.overfit_batch:
        fixed_batch = dataset.get_batch(
            "train", cfg.training.batch_size, cfg.model.block_size, device
        )
        print("overfit-batch mode: training on one fixed batch every iteration")

    model.train()
    t_start = time.time()
    data_time_total = 0.0
    step_time_total = 0.0
    tokens_per_step = cfg.tokens_per_step

    for it in range(iter_num, cfg.training.max_iters):
        lr = get_lr(
            it,
            cfg.training.warmup_iters,
            cfg.training.lr_decay_iters,
            cfg.training.learning_rate,
            cfg.training.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        step_t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.training.gradient_accumulation_steps):
            data_t0 = time.time()
            x, y = fixed_batch if fixed_batch is not None else dataset.get_batch(
                "train", cfg.training.batch_size, cfg.model.block_size, device
            )
            data_time_total += time.time() - data_t0

            with torch.autocast(device_type=device.type, dtype=ptdtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
                loss = loss / cfg.training.gradient_accumulation_steps
            loss.backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time_total += time.time() - step_t0

        if it % cfg.training.log_interval == 0:
            elapsed = time.time() - t_start
            tok_per_sec = (it - iter_num + 1) * tokens_per_step / max(elapsed, 1e-9)
            data_frac = data_time_total / max(step_time_total, 1e-9)
            peak_mib = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
            print(
                f"iter {it:6d} | loss {accum_loss:.4f} | lr {lr:.2e} | grad_norm {grad_norm:.3f} "
                f"| tok/s {tok_per_sec:,.0f} | data_time_frac {data_frac:.1%} | peak_vram {peak_mib:.0f}MiB"
            )
            if use_wandb:
                wandb.log(
                    {
                        "iter": it,
                        "train/loss": accum_loss,
                        "lr": lr,
                        "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                        "tokens_per_sec": tok_per_sec,
                        "data_time_frac": data_frac,
                        "peak_vram_mib": peak_mib,
                    },
                    step=it,
                )

        if (
            not args.overfit_batch
            and cfg.training.eval_interval > 0
            and it % cfg.training.eval_interval == 0
            and it > iter_num
        ):
            losses = estimate_loss(
                model, dataset, ["train", "val"], cfg.training.eval_iters,
                cfg.training.batch_size, cfg.model.block_size, device, ptdtype,
            )
            print(f"  eval @ iter {it}: train {losses['train']:.4f}  val {losses['val']:.4f}")
            if use_wandb:
                wandb.log({"eval/train_loss": losses["train"], "eval/val_loss": losses["val"]}, step=it)
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, it, best_val_loss, cfg, dataset)
                print(f"  new best val_loss {best_val_loss:.4f} — saved {ckpt_dir / 'best.pt'}")

        if cfg.training.checkpoint_interval > 0 and it % cfg.training.checkpoint_interval == 0 and it > iter_num:
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, it, best_val_loss, cfg, dataset)

    save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, cfg.training.max_iters, best_val_loss, cfg, dataset)
    print(f"done. final checkpoint: {ckpt_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
