"""Parsing correctness against src/train.py's actual log format."""

from __future__ import annotations

from src.plot_curves import parse_train_log

SAMPLE_LOG = """\
device: NVIDIA GeForce RTX 4070 Laptop GPU | 8.0 GiB | sm_89 | bf16=True
run: test  precision: bfloat16  max_iters: 100
model: 52.9M params (52,901,712)
iter      0 | loss 10.4335 | lr 3.00e-07 | grad_norm 13.489 | tok/s 22,281 | data_time_frac 1.8% | peak_vram 1930MiB
iter     25 | loss 8.1234 | lr 7.50e-06 | grad_norm 5.123 | tok/s 21,000 | data_time_frac 2.0% | peak_vram 1930MiB
  eval @ iter 25: train 8.0500  val 8.1200
iter     50 | loss 6.5000 | lr 1.50e-05 | grad_norm 3.200 | tok/s 21,500 | data_time_frac 2.0% | peak_vram 1930MiB
  eval @ iter 50: train 6.4000  val 6.5500
  new best val_loss 6.5500 — saved checkpoints\\test\\best.pt
iter     75 | loss 4.2000 | lr 2.25e-05 | grad_norm 2.100 | tok/s 21,800 | data_time_frac 1.9% | peak_vram 1930MiB
done. final checkpoint: checkpoints\\test\\latest.pt
"""


def test_parses_per_step_train_loss(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(SAMPLE_LOG, encoding="utf-8")
    parsed = parse_train_log(log)
    assert parsed["train_iter"] == [0, 25, 50, 75]
    assert parsed["train_loss"] == [10.4335, 8.1234, 6.5000, 4.2000]


def test_parses_eval_lines(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(SAMPLE_LOG, encoding="utf-8")
    parsed = parse_train_log(log)
    assert parsed["eval_iter"] == [25, 50]
    assert parsed["eval_train_loss"] == [8.0500, 6.4000]
    assert parsed["eval_val_loss"] == [8.1200, 6.5500]


def test_ignores_non_matching_lines(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(SAMPLE_LOG, encoding="utf-8")
    parsed = parse_train_log(log)
    # "new best val_loss ... saved checkpoints\test\best.pt" must not be
    # mistaken for an iter or eval line.
    assert len(parsed["train_iter"]) == 4
    assert len(parsed["eval_iter"]) == 2


def test_empty_log_returns_empty_lists(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("", encoding="utf-8")
    parsed = parse_train_log(log)
    assert parsed["train_iter"] == []
    assert parsed["eval_iter"] == []
