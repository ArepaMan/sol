"""Pins the >=1200px-wide export guarantee that M2's figures rely on."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless — no display needed for tests
import matplotlib.pyplot as plt
from PIL import Image

from src.plot_style import savefig, use_style


def test_savefig_meets_minimum_width(tmp_path):
    use_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([1, 2, 3], [1, 4, 9])
    out = tmp_path / "test.png"
    savefig(fig, out, width_px=1200)
    plt.close(fig)

    with Image.open(out) as img:
        assert img.width >= 1200
