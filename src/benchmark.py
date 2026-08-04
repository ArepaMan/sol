"""Gate 2 — VRAM / throughput sweep.

Usage:
    python -m src.benchmark --config configs/micro_50m_8gb.yaml --sweep

Measures peak VRAM and tokens/s for a handful of (batch_size, gradient_checkpointing,
compile) combinations on the real model, and writes the results to
experiments/000_benchmark/results.md. This is what settles torch.compile
on/off and confirms the chosen config's peak VRAM stays under the ~7400 MiB
target (8188 MiB total minus ~800 MiB headroom for Windows WDDM + display) —
by measurement, not by assuming the M3 preview (a single un-optimized
forward+backward) generalizes to the full training loop with an optimizer
attached.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from src.config import ModelConfig, load_config
from src.model import GPT
from src.utils import describe_device, get_device

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class BenchmarkResult:
    batch_size: int
    gradient_checkpointing: bool
    compile: bool
    peak_vram_mib: float
    tokens_per_sec: float
    compile_time_s: float
    ok: bool
    error: str = ""

    @property
    def suspected_shared_memory_spill(self) -> bool:
        """True if reported peak VRAM exceeds this GPU's physical total — CUDA
        on Windows doesn't cleanly OOM in that case, it silently falls back to
        much slower shared/system memory. `ok=True` with this flag set means
        "ran, but not the way you think" — treat as a failure for planning
        purposes even though no exception was raised."""
        if not torch.cuda.is_available():
            return False
        physical_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2
        return self.ok and self.peak_vram_mib > physical_mib


def _run_one(
    model_config: ModelConfig,
    batch_size: int,
    grad_checkpointing: bool,
    use_compile: bool,
    ptdtype: torch.dtype,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
) -> BenchmarkResult:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        model = GPT(model_config, gradient_checkpointing=grad_checkpointing).to(device)
        model.train()
        optimizer = model.configure_optimizers(
            weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95), device_type=device.type
        )

        compile_time = 0.0
        if use_compile:
            t0 = time.time()
            model = torch.compile(model)
            compile_time = time.time() - t0  # actual compile is lazy; real cost shows on first step

        block_size = model_config.block_size
        vocab_size = model_config.vocab_size

        def step():
            x = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
            y = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=ptdtype):
                _, loss = model(x, y)
            loss.backward()
            optimizer.step()

        t_compile0 = time.time()
        for _ in range(warmup_steps):
            step()
        if use_compile:
            torch.cuda.synchronize()
            compile_time += time.time() - t_compile0  # first-call compile cost is here

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # only measure the steady-state steps
        t0 = time.time()
        for _ in range(measure_steps):
            step()
        torch.cuda.synchronize()
        dt = time.time() - t0

        peak_mib = torch.cuda.max_memory_allocated() / 1024**2
        tokens_per_sec = (measure_steps * batch_size * block_size) / dt

        del model, optimizer
        torch.cuda.empty_cache()

        return BenchmarkResult(
            batch_size, grad_checkpointing, use_compile, peak_mib, tokens_per_sec, compile_time, ok=True
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return BenchmarkResult(batch_size, grad_checkpointing, use_compile, float("nan"), float("nan"), 0.0, ok=False, error="OOM")
    except Exception as e:  # torch.compile on Windows/Triton can fail in its own ways
        torch.cuda.empty_cache()
        return BenchmarkResult(batch_size, grad_checkpointing, use_compile, float("nan"), float("nan"), 0.0, ok=False, error=f"{type(e).__name__}: {e}")


def run_sweep(
    model_config: ModelConfig,
    precision: str,
    device: torch.device,
    warmup_steps: int = 3,
    measure_steps: int = 10,
) -> list[BenchmarkResult]:
    ptdtype = _DTYPE_MAP[precision]
    combos = [
        (4, True, False),
        (4, False, False),
        (8, True, False),
        (4, True, True),
        # Headroom probes: the first four rows leave 5+ GB unused at the
        # chosen batch_size, so it's worth knowing where the real ceiling is
        # rather than just clearing the target with room to spare. b=32
        # without checkpointing is the interesting one — it doesn't cleanly
        # OOM, it silently spills into slow shared/system memory instead
        # (peak_vram reported >8188 MiB physical — a real footgun if batch
        # size is ever scaled up without re-running this).
        (16, False, False),
        (32, False, False),
        (16, True, False),
        (32, True, False),
        (64, False, False),
    ]
    results = []
    for batch_size, ckpt, compile_flag in combos:
        label = f"b{batch_size} ckpt={ckpt} compile={compile_flag}"
        print(f"running {label} ...")
        r = _run_one(model_config, batch_size, ckpt, compile_flag, ptdtype, device, warmup_steps, measure_steps)
        if r.ok and r.suspected_shared_memory_spill:
            print(
                f"  peak_vram={r.peak_vram_mib:.0f}MiB (> physical VRAM!) tokens/s={r.tokens_per_sec:,.0f} "
                f"— SUSPECTED SHARED-MEMORY SPILL, not a clean fit"
            )
        elif r.ok:
            print(f"  peak_vram={r.peak_vram_mib:.0f}MiB  tokens/s={r.tokens_per_sec:,.0f}")
        else:
            print(f"  FAILED: {r.error}")
        results.append(r)
    return results


def write_results_md(
    results: list[BenchmarkResult],
    device_desc: str,
    path: Path,
    chosen_batch_size: int | None = None,
    chosen_checkpointing: bool | None = None,
    chosen_compile: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 000 — VRAM / throughput benchmark",
        "",
        f"Device: {device_desc}",
        "",
        "| batch_size | grad_checkpointing | compile | peak VRAM (MiB) | tokens/s | status |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.ok and r.suspected_shared_memory_spill:
            lines.append(
                f"| {r.batch_size} | {r.gradient_checkpointing} | {r.compile} | "
                f"{r.peak_vram_mib:.0f} | {r.tokens_per_sec:,.0f} | ⚠️ shared-memory spill |"
            )
        elif r.ok:
            lines.append(
                f"| {r.batch_size} | {r.gradient_checkpointing} | {r.compile} | "
                f"{r.peak_vram_mib:.0f} | {r.tokens_per_sec:,.0f} | ok |"
            )
        else:
            # Markdown table cells can't contain raw newlines or unescaped
            # pipes — a multi-line exception message (torch.compile errors
            # routinely are) would otherwise corrupt every row after it.
            error_summary = r.error.splitlines()[0].replace("|", "\\|") if r.error else "failed"
            lines.append(
                f"| {r.batch_size} | {r.gradient_checkpointing} | {r.compile} | — | — | {error_summary} |"
            )
    lines.append("")

    failures = [r for r in results if not r.ok]
    if failures:
        lines.append("### Failure details")
        lines.append("")
        for r in failures:
            lines.append(f"**b{r.batch_size} ckpt={r.gradient_checkpointing} compile={r.compile}:**")
            lines.append("```")
            lines.append(r.error)
            lines.append("```")
        lines.append("")

    if chosen_batch_size is not None and chosen_checkpointing is not None:
        chosen = next(
            (
                r for r in results
                if r.ok
                and not r.suspected_shared_memory_spill
                and r.batch_size == chosen_batch_size
                and r.gradient_checkpointing == chosen_checkpointing
                and r.compile == chosen_compile
            ),
            None,
        )
        if chosen:
            lines.append(
                f"**Chosen config** (batch_size={chosen_batch_size}, "
                f"gradient_checkpointing={chosen_checkpointing}, compile={chosen_compile}): "
                f"peak VRAM {chosen.peak_vram_mib:.0f} MiB "
                f"({'under' if chosen.peak_vram_mib < 7400 else 'OVER'} the 7400 MiB target), "
                f"{chosen.tokens_per_sec:,.0f} tokens/s."
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/micro_50m_8gb.yaml")
    parser.add_argument("--sweep", action="store_true", help="Run the full combo sweep (otherwise just prints device info).")
    parser.add_argument("--out", default="experiments/000_benchmark/results.md")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=10)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = get_device()
    device_desc = describe_device()
    print(f"device: {device_desc}")

    if device.type != "cuda":
        raise SystemExit("benchmark requires a CUDA device")

    if not args.sweep:
        return

    results = run_sweep(cfg.model, cfg.precision, device, args.warmup_steps, args.measure_steps)
    write_results_md(
        results,
        device_desc,
        Path(args.out),
        chosen_batch_size=cfg.training.batch_size,
        chosen_checkpointing=cfg.gradient_checkpointing,
        chosen_compile=cfg.compile,
    )


if __name__ == "__main__":
    main()
