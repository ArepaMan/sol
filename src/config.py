"""Config loading for Sol.

The YAML file in `configs/` is the single source of truth for every number in
this project. Nothing downstream should hardcode a hyperparameter — if you need
one, read it off `SolConfig`. `scripts/export_spec.py` (M9) generates the
portfolio's copy from the same YAML, so there is exactly one place to edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 512
    vocab_size: int = 32000
    dropout: float = 0.0
    bias: bool = False

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.vocab_size > 65535:
            raise ValueError(
                f"vocab_size {self.vocab_size} exceeds uint16 range; data/tokenize.py "
                "stores token ids as uint16"
            )

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    max_iters: int = 40000
    learning_rate: float = 3.0e-4
    min_lr: float = 3.0e-5
    warmup_iters: int = 1000
    lr_decay_iters: int = 40000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 25
    checkpoint_interval: int = 2000


@dataclass
class DataConfig:
    dataset: str = "tinystories"
    max_train_tokens: int = 400_000_000
    val_split: float = 0.05


@dataclass
class SolConfig:
    name: str = "micro_50m_8gb"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    precision: str = "bfloat16"
    gradient_checkpointing: bool = True
    compile: bool = False
    seed: int = 42

    @property
    def tokens_per_step(self) -> int:
        """Tokens consumed by one optimizer step (all accumulation micro-batches)."""
        return (
            self.training.batch_size
            * self.training.gradient_accumulation_steps
            * self.model.block_size
        )

    @property
    def total_train_tokens(self) -> int:
        """Total tokens the model will *process* over the whole run."""
        return self.tokens_per_step * self.training.max_iters

    @property
    def epochs(self) -> float:
        """Passes over the corpus. Distinct from total_train_tokens — the spec
        originally conflated the two (40k iters is ~3.3 epochs over 400M tokens,
        not a 400M-token run)."""
        return self.total_train_tokens / self.data.max_train_tokens


def _build(cls: type, raw: dict[str, Any] | None) -> Any:
    """Instantiate a dataclass from a dict, rejecting unknown keys loudly.

    A silently-ignored typo in a config is the kind of bug that costs a 20-hour
    training run, so this raises instead of shrugging.
    """
    raw = raw or {}
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> SolConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    nested = {"model", "training", "data"}
    top_level = {k: v for k, v in raw.items() if k not in nested}

    known_top = {f.name for f in fields(SolConfig)} - nested
    unknown = set(top_level) - known_top
    if unknown:
        raise ValueError(f"SolConfig: unknown config keys {sorted(unknown)}")

    cfg = SolConfig(
        model=_build(ModelConfig, raw.get("model")),
        training=_build(TrainConfig, raw.get("training")),
        data=_build(DataConfig, raw.get("data")),
        **top_level,
    )

    if cfg.precision not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"unsupported precision: {cfg.precision}")

    return cfg
