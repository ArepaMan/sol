"""Inference for Sol — a reusable generator class plus a CLI.

Two consumers share this module: the CLI below (`python -m src.infer`) and the
Gradio demo (`app/demo.py`). Keeping generation in one place means the demo and
the command line cannot silently drift apart in sampling behaviour.

Three things this does that `GPT.generate` (src/model.py) deliberately does not:

1. **Stops at `<|endoftext|>`.** M6 flagged this as a real gap — every sampled
   generation ran the full `max_new_tokens` and then kept going *past* the end
   of its story, because the training-time generate loop has no stop condition.
   The model does emit EOT at a sensible place; nothing was reading it. Fixed
   here rather than in `src/model.py` so the M3 architecture file stays a clean
   architecture file.
2. **Streams.** Yields decoded text incrementally so the demo can show tokens
   as they arrive instead of freezing for 30 s of CPU sampling.
3. **Re-seeds per call.** `--seed 42` twice gives byte-identical output — an M8
   exit criterion, and the thing that makes a demo bug reproducible at all.

Usage:
    python -m src.infer --prompt "Once upon a time" --seed 42
    python -m src.infer --checkpoint checkpoints/001_baseline/latest.pt --stream
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.config import SolConfig, load_config
from src.model import GPT
from src.utils import get_device, set_seed

DEFAULT_CHECKPOINT = "checkpoints/001_baseline/latest.pt"
DEFAULT_CONFIG = "configs/micro_50m_8gb.yaml"
DEFAULT_TOKENIZER = "data/tokenizer/tokenizer.json"
DEFAULT_META = "data/tokenized/meta.json"

# Fallback only. The real value comes from data/tokenized/meta.json, which is
# committed precisely so this number never has to be guessed.
FALLBACK_EOT_ID = 0


def resolve_dtype(device: torch.device, precision: str) -> torch.dtype:
    """bf16 on CUDA (what the model trained in), fp32 on CPU.

    CPU bf16 is *supported* but runs through slow reference kernels on most
    x86 parts — measurably worse than fp32 for a 52M model, and the HF Space
    this ships to is CPU-only. Sampling quality is unaffected: the weights were
    bf16 during training either way, so fp32 inference is a strict widening.
    """
    if device.type == "cuda":
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[precision]
    return torch.float32


def load_eot_id(meta_path: str | Path | None) -> int:
    if meta_path is None:
        return FALLBACK_EOT_ID
    path = Path(meta_path)
    if not path.exists():
        return FALLBACK_EOT_ID
    with path.open("r", encoding="utf-8") as f:
        return int(json.load(f)["eot_id"])


class SolGenerator:
    """A loaded model + tokenizer, ready to sample from.

    Constructed once (at module import in the demo, per-process in the CLI) —
    never per request. Loading the 52M-param model costs ~1-2 s on CPU, which
    is fine once and unacceptable on every click.
    """

    def __init__(
        self,
        model: GPT,
        tokenizer: Tokenizer,
        device: torch.device,
        eot_id: int,
        block_size: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eot_id = eot_id
        self.block_size = block_size

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        config: str | Path | SolConfig = DEFAULT_CONFIG,
        tokenizer: str | Path = DEFAULT_TOKENIZER,
        meta: str | Path | None = DEFAULT_META,
        device: torch.device | None = None,
    ) -> SolGenerator:
        """Load from a *training* checkpoint (the .pt in `checkpoints/`).

        These carry optimizer state and are ~635 MB. For deployment use
        `from_pretrained` on an exported weights-only bundle instead — see
        `scripts/export_weights.py`.
        """
        cfg = config if isinstance(config, SolConfig) else load_config(config)
        device = device or get_device()
        dtype = resolve_dtype(device, cfg.precision)

        model = GPT(cfg.model, gradient_checkpointing=False)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state)
        model.to(device=device, dtype=dtype)
        model.eval()

        return cls(
            model=model,
            tokenizer=Tokenizer.from_file(str(tokenizer)),
            device=device,
            eot_id=load_eot_id(meta),
            block_size=cfg.model.block_size,
        )

    @classmethod
    def from_pretrained(
        cls, model_dir: str | Path, device: torch.device | None = None
    ) -> SolGenerator:
        """Load from an exported bundle: `model.pt` + `tokenizer.json` +
        `config.yaml` + `meta.json`. This is what the HF Space downloads —
        weights only, bf16, ~110 MB instead of 635 MB.
        """
        model_dir = Path(model_dir)
        return cls.from_checkpoint(
            checkpoint=model_dir / "model.pt",
            config=model_dir / "config.yaml",
            tokenizer=model_dir / "tokenizer.json",
            meta=model_dir / "meta.json",
            device=device,
        )

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 200,
        seed: int | None = None,
        stop_at_eot: bool = True,
    ) -> Iterator[str]:
        """Yield the completion incrementally, one decoded delta per token.

        Yields *only the completion*, not the prompt. The final yielded text
        concatenated with the prompt is what `generate()` returns.
        """
        if seed is not None:
            set_seed(seed)

        prompt_ids = self.tokenizer.encode(prompt).ids
        # A prompt longer than the context window can't be conditioned on in
        # full; keep the tail, which is what the model would attend to anyway.
        if len(prompt_ids) >= self.block_size:
            prompt_ids = prompt_ids[-(self.block_size - 1) :]

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated: list[int] = []
        emitted = ""

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :].float() / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

            if stop_at_eot and next_id == self.eot_id:
                break

            generated.append(next_id)
            idx = torch.cat((idx, torch.tensor([[next_id]], device=self.device)), dim=1)

            # Decode the whole completion each step rather than the single new
            # token: byte-level BPE tokens are not independently decodable
            # (a multi-byte character can straddle two tokens), so per-token
            # decoding produces replacement characters mid-word.
            text = self.tokenizer.decode(generated)
            if len(text) > len(emitted):
                yield text[len(emitted) :]
                emitted = text

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 200,
        seed: int | None = None,
        stop_at_eot: bool = True,
    ) -> str:
        """Full completion, prompt included. Same sampling path as `stream`."""
        parts = list(
            self.stream(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
                stop_at_eot=stop_at_eot,
            )
        )
        return prompt + "".join(parts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--meta", default=DEFAULT_META)
    parser.add_argument("--model-dir", default=None, help="exported bundle dir (overrides the four paths above)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--stream", action="store_true", help="print tokens as they are sampled")
    parser.add_argument("--no-stop-at-eot", action="store_true", help="run the full token budget")
    parser.add_argument("--n", type=int, default=1, help="number of samples")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    device = torch.device(args.device) if args.device else get_device()

    if args.model_dir:
        gen = SolGenerator.from_pretrained(args.model_dir, device=device)
    else:
        gen = SolGenerator.from_checkpoint(
            checkpoint=args.checkpoint,
            config=args.config,
            tokenizer=args.tokenizer,
            meta=args.meta,
            device=device,
        )

    for i in range(args.n):
        # Distinct seeds per sample, still fully determined by --seed: `--n 3
        # --seed 42` reproduces exactly, but doesn't print the same story 3×.
        seed = args.seed + i
        t0 = time.perf_counter()

        if args.stream:
            print(args.prompt, end="", flush=True)
            chunks = []
            for chunk in gen.stream(
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed,
                stop_at_eot=not args.no_stop_at_eot,
            ):
                print(chunk, end="", flush=True)
                chunks.append(chunk)
            print()
            # Not len(chunks): a sampled token whose bytes don't complete a
            # character yields no delta, so chunk count undercounts tokens.
            n_tokens = len(gen.tokenizer.encode("".join(chunks)).ids)
        else:
            text = gen.generate(
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed,
                stop_at_eot=not args.no_stop_at_eot,
            )
            print(text)
            n_tokens = len(gen.tokenizer.encode(text).ids) - len(gen.tokenizer.encode(args.prompt).ids)

        dt = time.perf_counter() - t0
        rate = n_tokens / dt if dt > 0 else float("nan")
        # stderr, not stdout: the M8 exit criterion is that two `--seed 42` runs
        # are byte-identical, and wall-clock never is. Keeping timing off stdout
        # makes `python -m src.infer ... > a.txt` diffable as-is.
        print(
            f"[seed={seed} device={device.type} {n_tokens} tokens in {dt:.2f}s = {rate:.1f} tok/s]",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
