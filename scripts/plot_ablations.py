"""Plot M7's ablation results against the seed-variance floor.

The whole methodological point of M7 is that an ablation gap only means
something relative to how much two identical runs differ. A bar chart of
perplexities hides that; this plots the seed-noise band explicitly so a reader
can see which gaps clear it and by how much.

Reads `experiments/ablation_eval_results.json` (produced by
`scripts/eval_ablation_checkpoints.py`) — no numbers are typed in here.

Usage:
    python -m scripts.plot_ablations
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from src.plot_style import CATEGORICAL, CHROME, savefig, use_style

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "experiments" / "ablation_eval_results.json"
OUT = REPO_ROOT / "experiments" / "ablation_summary.png"


def main() -> None:
    with RESULTS.open("r", encoding="utf-8") as f:
        r = json.load(f)

    seed_runs = ["002_lr_3e-4", "004_seed_43", "004_seed_44"]
    seed_ppls = [r[k]["ppl"] for k in seed_runs]
    seed_mean = statistics.fmean(seed_ppls)
    seed_sd = statistics.stdev(seed_ppls)

    use_style()
    fig, (ax_lr, ax_data) = plt.subplots(1, 2, figsize=(11, 4.4), width_ratios=[1.35, 1])

    def draw_noise_band(ax, label: bool) -> None:
        """±1 sd around the seed mean — the 'is this even real' threshold.

        Labelled on the left panel only: on the right the band sits right at the
        full-corpus bar's top, so a second label collides with its value.
        """
        ax.axhspan(
            seed_mean - seed_sd, seed_mean + seed_sd,
            color=CATEGORICAL[2], alpha=0.30, zorder=0,
        )
        ax.axhline(seed_mean, color=CATEGORICAL[2], lw=1.2, zorder=1)
        if label:
            ax.text(
                0.02, seed_mean, f"  seed noise ±{seed_sd:.4f}  ",
                transform=ax.get_yaxis_transform(), va="bottom", ha="left",
                fontsize=9, color=CHROME["secondary_ink"],
            )

    # --- Learning rate sweep --------------------------------------------------
    lr_keys = [("1e-4", "002_lr_1e-4"), ("3e-4", "002_lr_3e-4"), ("1e-3", "002_lr_1e-3")]
    labels = [k for k, _ in lr_keys]
    ppls = [r[k]["ppl"] for _, k in lr_keys]
    errs = [
        [r[k]["ppl"] - r[k]["ci_lo"] for _, k in lr_keys],
        [r[k]["ci_hi"] - r[k]["ppl"] for _, k in lr_keys],
    ]
    draw_noise_band(ax_lr, label=True)
    ax_lr.bar(labels, ppls, color=CATEGORICAL[0], width=0.55, zorder=2)
    ax_lr.errorbar(labels, ppls, yerr=errs, fmt="none", ecolor=CHROME["primary_ink"], capsize=4, zorder=3)
    for lbl, p in zip(labels, ppls, strict=True):
        ax_lr.text(lbl, p + 0.06, f"{p:.3f}", ha="center", fontsize=10, fontweight="bold", zorder=4)
    ax_lr.set_title("Learning rate: a large, real effect")
    ax_lr.set_xlabel("peak learning rate")
    ax_lr.set_ylabel("validation perplexity")
    ax_lr.set_ylim(3.9, 5.7)

    # --- Data scale -----------------------------------------------------------
    d_keys = [("100M tokens", "003_data_100m"), ("full corpus\n357.9M", "002_lr_3e-4")]
    d_labels = [k for k, _ in d_keys]
    d_ppls = [r[k]["ppl"] for _, k in d_keys]
    d_errs = [
        [r[k]["ppl"] - r[k]["ci_lo"] for _, k in d_keys],
        [r[k]["ci_hi"] - r[k]["ppl"] for _, k in d_keys],
    ]
    draw_noise_band(ax_data, label=False)
    ax_data.bar(d_labels, d_ppls, color=CATEGORICAL[1], width=0.5, zorder=2)
    ax_data.errorbar(d_labels, d_ppls, yerr=d_errs, fmt="none", ecolor=CHROME["primary_ink"], capsize=4, zorder=3)
    for lbl, p in zip(d_labels, d_ppls, strict=True):
        ax_data.text(lbl, p + 0.012, f"{p:.3f}", ha="center", fontsize=10, fontweight="bold", zorder=4)
    ax_data.set_title("Data scale: real, but small")
    ax_data.set_ylabel("validation perplexity")
    ax_data.set_ylim(4.40, 4.68)

    fig.suptitle(
        "Sol — M7 ablations, read against seed variance (8000 iters, n=3 seeds)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.06,
        f"Seed-to-seed sd is {seed_sd:.4f} ppl. The LR gap is ~{(ppls[0] - ppls[2]) / seed_sd:.0f}x that floor; "
        f"the data-scale gap is ~{(d_ppls[0] - d_ppls[1]) / seed_sd:.0f}x. Error bars are bootstrap 95% CIs.",
        ha="center", fontsize=9.5, color=CHROME["secondary_ink"],
    )
    fig.tight_layout()
    savefig(fig, OUT)


if __name__ == "__main__":
    main()
