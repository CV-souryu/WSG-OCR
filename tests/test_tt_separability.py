"""T/Т separability: can the CNN ever distinguish Latin T from Cyrillic Т?

The T-23 real-game samples decode as ``Т-23`` (Cyrillic Te).  Before
deciding between "constrain the charset" and "train hard negatives" the
visual evidence must be checked: if the two characters rasterize to the
*identical* pixel mask in the registered font, no CNN can separate them
(they are the same image), and the fix is charset / allowed_chars /
lexicon disambiguation -- not CNN training.

Verdict pinned here: SourceHanSansSC renders T and Т pixel-identically at
14 px (the real-game size), 16 px and 32 px, and their Goal-3 normalized
24x24 forms are identical (Hamming distance 0).  If a future font change
makes them separable, this test fails and CNN training becomes an option.
"""

from __future__ import annotations

import numpy as np
import pytest

from fixedfontocr.oracle import render_glyph_mask
from fixedfontocr.preprocess import normalize

SIZES = (14, 16, 32)


def test_latin_t_and_cyrillic_te_are_pixel_identical(font_path):
    """The registered font renders T and Т as the same raster at every
    tested size: raw masks identical pixel for pixel."""

    for size in SIZES:
        latin = render_glyph_mask("T", font_path, font_size=size)
        cyrillic = render_glyph_mask("Т", font_path, font_size=size)
        assert latin.shape == cyrillic.shape, (
            f"{size}px: shape differs {latin.shape} vs {cyrillic.shape}"
        )
        assert np.array_equal(latin, cyrillic), (
            f"{size}px: T and Т rasters differ -- a CNN could separate "
            "them; the T/Т fix may become a hard-negative problem"
        )


def test_latin_t_and_cyrillic_te_normalized_identical(font_path):
    """The Goal-3 normalized 24x24 forms are identical (Hamming 0/576)."""

    for size in SIZES:
        latin = normalize(
            render_glyph_mask("T", font_path, font_size=size), 24
        )
        cyrillic = normalize(
            render_glyph_mask("Т", font_path, font_size=size), 24
        )
        assert np.array_equal(latin, cyrillic), (
            f"{size}px: normalized T/Т differ ({int(np.count_nonzero(latin != cyrillic))} px)"
        )


def test_cyrillic_te_is_in_model_charset():
    """The model can output Т only because the charset contains it.

    This is what makes the T/Т confusion visible at all; constraining
    allowed characters for game text (no Cyrillic) is the fix.
    """

    from pathlib import Path

    charset_file = Path("model/game_cn/charset.txt")
    if not charset_file.exists():
        pytest.skip("model/game_cn not built")
    charset = charset_file.read_text(encoding="utf-8")
    assert "T" in charset
    assert "Т" in charset  # the lookalike that must be constrained
