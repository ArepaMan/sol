"""Parse src/train.py's console log and plot loss curves.

Shared between M5 (the baseline run's loss curve) and M7 (ablation comparison
plots) — one parser for the one log format src/train.py actually produces,
rather than reimplementing regex-matching per use site.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt

from src.plot_style import CATEGORICAL, CHROME, savefig, use_style

_ITER_RE = re.compile(r"^iter\s+(\d+)\s+\|\s+loss\s+([\d.]+)")
_EVAL_RE = re.compile(r"^\s*eval @ iter (\d+): train\s+([\d.]+)\s+val\s+([\d.]+)")


def parse_train_log(path: str | Path) -> dict[str, list[float]]:
    """Returns {"train_iter": [...], "train_loss": [...], "eval_iter": [...],
    "eval_train_loss": [...], "eval_val_loss": [...]} — the per-step noisy
    train loss and the periodic (usually less noisy, larger-batch) eval loss,
    kept separate since they're different signals plotted differently."""
    train_iter, train_loss = [], []
    eval_iter, eval_train_loss, eval_val_loss = [], [], []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _ITER_RE.match(line)
            if m:
                train_iter.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                continue
            m = _EVAL_RE.match(line)
            if m:
                eval_iter.append(int(m.group(1)))
                eval_train_loss.append(float(m.group(2)))
                eval_val_loss.append(float(m.group(3)))

    return {
        "train_iter": train_iter,
        "train_loss": train_loss,
        "eval_iter": eval_iter,
        "eval_train_loss": eval_train_loss,
        "eval_val_loss": eval_val_loss,
    }


def plot_loss_curve(parsed: dict, title: str, out_path: str | Path) -> None:
    use_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        parsed["train_iter"], parsed["train_loss"],
        color=CATEGORICAL[0], alpha=0.25, linewidth=0.8, label="train (per-step)",
    )
    if parsed["eval_iter"]:
        ax.plot(
            parsed["eval_iter"], parsed["eval_train_loss"],
            color=CATEGORICAL[0], linewidth=2, label="train (eval, 100-batch avg)",
        )
        ax.plot(
            parsed["eval_iter"], parsed["eval_val_loss"],
            color=CATEGORICAL[1], linewidth=2, marker="o", markersize=3, label="val (eval, 100-batch avg)",
        )

    ax.axhline(3.2, color=CHROME["muted"], linestyle="--", linewidth=1)
    ax.text(
        ax.get_xlim()[1] if parsed["train_iter"] else 1, 3.2,
        " target ≤3.2", fontsize=9, color=CHROME["muted"], va="bottom", ha="right",
    )

    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    savefig(fig, out_path)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Training loss")
    args = parser.parse_args()

    parsed = parse_train_log(args.log)
    plot_loss_curve(parsed, args.title, args.out)


if __name__ == "__main__":
    main()
