from __future__ import annotations

import numpy as np
import pytest

from fixedfontocr import defaults
from fixedfontocr.fontgen import _cmap_codepoints, build_templates


def test_font_must_live_inside_fonts(tmp_path):
    outside = tmp_path / "fake.ttf"
    outside.write_bytes(b"not a real font")
    with pytest.raises(ValueError, match="must be inside"):
        defaults.resolve_font(outside)


def test_missing_glyph_is_an_explicit_error(font_path):
    covered = _cmap_codepoints(str(font_path))
    missing_cp = next(cp for cp in range(0xE000, 0xF8FF) if cp not in covered)
    with pytest.raises(ValueError, match="missing glyph"):
        build_templates(font_path, [chr(missing_cp)])


def test_synthetic_dataset_without_font_sha_is_rejected(tmp_path):
    from tools.train.train import build_dataset

    bad = tmp_path / "bad.npz"
    np.savez(
        bad,
        x=np.zeros((1, 24, 24), dtype=np.uint8),
        y=np.zeros(1, dtype=np.int64),
        chars=np.asarray(["0"], dtype="<U8"),
    )
    with pytest.raises(SystemExit, match="font_sha256"):
        build_dataset(bad, None)
