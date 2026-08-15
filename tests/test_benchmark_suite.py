"""Smoke-test the Goal 17 CPU benchmark suite (fast mode)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_benchmark_fast_mode(tmp_path):
    out = tmp_path / "bench.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "benchmark" / "cpu_benchmark.py"),
            "--fast",
            "--output",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data["ocr_stages"]) == {
        "foreground",
        "cc",
        "lattice_generation",
        "normalize",
        "template",
        "tinycnn",
        "decoder",
        "total",
    }
    for stage in data["ocr_stages"].values():
        assert stage["median"] > 0
        assert stage["p95"] >= stage["median"]
    assert set(data["cnn_batch"]) == {"1", "8", "16", "32", "64", "128"}
    assert set(data["charsets"]) == {"10", "100"}
    assert data["optimized_vs_reference"]["argmax_match"] is True
    assert data["optimized_vs_reference"]["max_error"] < 1e-5


def test_goal17_default_charset_matrix_is_10_100_1894():
    """The full run must cover the real 1894-char project charset."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "benchmark" / "cpu_benchmark.py"),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "10,100,1894" in result.stdout
