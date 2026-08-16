from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from fixedfontocr.classifier import TemplateClassifier  # noqa: E402
from fixedfontocr.fontgen import build_templates  # noqa: E402
from tools.benchmark.footprint import fmt_bytes, wgpu_buffers_per_batch  # noqa: E402


def test_compact_feature_dtypes(font_path):
    chars, templates = build_templates(font_path, list("0123456789"), render_size=28)
    clf = TemplateClassifier(templates, chars)
    assert clf._features["ink"].dtype == np.uint16
    for name in ("h", "w", "top", "left", "bottom", "right"):
        assert clf._features[name].dtype == np.uint8
    # numpy >= 2.0 uses bitwise_count and skips the 64 KiB table.
    if hasattr(np, "bitwise_count"):
        assert clf._popcount_table is None


def test_wgpu_buffer_formula():
    per_glyph = wgpu_buffers_per_batch(1)
    assert per_glyph == 24 * 24 + 12 + 12  # input + result + result staging
    assert wgpu_buffers_per_batch(64) == 64 * per_glyph
    # forward_logits adds the full-logits record + its staging buffer.
    assert wgpu_buffers_per_batch(1, classes=1894) == per_glyph + 8 * 1894


def test_fmt_bytes():
    assert fmt_bytes(1023) == "1,023 B"
    assert fmt_bytes(2048).startswith("2.0 KiB")


def test_footprint_script_runs_on_cnn_model():
    model = Path(__file__).parent / "fixtures" / "cnn_digits"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "benchmark" / "footprint.py"),
            str(model),
            "--batch-sizes",
            "1,8",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "cnn weights" in proc.stdout
    assert "batch    1" in proc.stdout
