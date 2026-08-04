"""CPU-only tests for src/benchmark.py — the GPU sweep itself is exercised
manually (Gate 2), but the results-table writer and result plumbing don't
need a GPU to verify."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.benchmark import BenchmarkResult, write_results_md


def test_suspected_shared_memory_spill_flags_over_physical_vram():
    """The finding that matters most from Gate 2: CUDA on Windows doesn't
    cleanly OOM when a batch exceeds physical VRAM — it silently spills into
    slow shared memory. This property is how that gets caught instead of
    just looking like a big-but-fine peak_vram number."""
    fake_props = MagicMock()
    fake_props.total_memory = 8188 * 1024**2  # bytes, matches the real card

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=fake_props):
        under = BenchmarkResult(4, True, False, 1460.0, 17000.0, 0.0, ok=True)
        over = BenchmarkResult(32, False, False, 11965.0, 4468.0, 0.0, ok=True)
        assert not under.suspected_shared_memory_spill
        assert over.suspected_shared_memory_spill


def test_suspected_shared_memory_spill_false_when_not_ok():
    fake_props = MagicMock()
    fake_props.total_memory = 8188 * 1024**2
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=fake_props):
        failed = BenchmarkResult(64, False, False, float("nan"), float("nan"), 0.0, ok=False, error="OOM")
        assert not failed.suspected_shared_memory_spill


def test_write_results_md_reports_ok_and_failed_rows(tmp_path):
    results = [
        BenchmarkResult(4, True, False, 1668.0, 22000.0, 0.0, ok=True),
        BenchmarkResult(8, True, False, float("nan"), float("nan"), 0.0, ok=False, error="OOM"),
    ]
    out = tmp_path / "results.md"
    write_results_md(results, "fake-gpu", out)

    text = out.read_text(encoding="utf-8")
    assert "1668" in text
    assert "22,000" in text
    assert "OOM" in text
    assert "fake-gpu" in text


def test_write_results_md_flags_chosen_config_under_target(tmp_path):
    results = [BenchmarkResult(4, True, False, 5000.0, 20000.0, 0.0, ok=True)]
    out = tmp_path / "results.md"
    write_results_md(results, "fake-gpu", out, chosen_batch_size=4, chosen_checkpointing=True)
    text = out.read_text(encoding="utf-8")
    assert "under the 7400 MiB target" in text


def test_write_results_md_flags_chosen_config_over_target(tmp_path):
    results = [BenchmarkResult(4, True, False, 8000.0, 20000.0, 0.0, ok=True)]
    out = tmp_path / "results.md"
    write_results_md(results, "fake-gpu", out, chosen_batch_size=4, chosen_checkpointing=True)
    text = out.read_text(encoding="utf-8")
    assert "OVER the 7400 MiB target" in text


def test_write_results_md_omits_chosen_line_when_not_specified(tmp_path):
    results = [BenchmarkResult(4, True, False, 5000.0, 20000.0, 0.0, ok=True)]
    out = tmp_path / "results.md"
    write_results_md(results, "fake-gpu", out)  # no chosen_* kwargs
    text = out.read_text(encoding="utf-8")
    assert "Chosen config" not in text
