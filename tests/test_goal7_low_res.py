"""Goal 7 acceptance: low-resolution training domain.

The Goal 7 contract from ``fonts/goal``:

* training data comes from ``fonts/`` at the real UI sizes 10..18 px,
  not from a single clean 32 px render;
* augmentation simulates ``font -> game render -> UI scale -> GPU
  sampling -> screenshot``: sub-pixel x/y offset, different scale ratios,
  downsampling with bilinear/area-like degradation, slight blur, alpha,
  brightness, background blending and outline changes;
* the regression strings ``Z17`` and ``巴尔的摩`` remain part of the
  low-res dataset domain.

``tools/dataset/generate_font_dataset.py`` owns the generator; the primary
``tools/train/build_model.py`` pipeline and the legacy
``scripts/train_tinycnn.py`` entry point both consume the same low-res
domain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.dataset.generate_font_dataset import (  # noqa: E402
    DEFAULT_RENDER_SIZE_MAX,
    DEFAULT_RENDER_SIZE_MIN,
    DOWNSAMPLE_MODES,
    LOW_RES_SIZES,
    generate_dataset,
)
from tools.dataset.generate_synthetic_samples import render  # noqa: E402

from fixedfontocr.preprocess import find_lines  # noqa: E402
from fixedfontocr.types import default_profile  # noqa: E402


LOW_RES_REGRESSION_CHARSET = list("Z17巴尔的摩")


def test_low_res_size_domain_is_10_to_18_px():
    assert LOW_RES_SIZES == tuple(range(10, 19))
    assert DEFAULT_RENDER_SIZE_MIN == 10
    assert DEFAULT_RENDER_SIZE_MAX == 18
    assert DOWNSAMPLE_MODES == ("bilinear", "area", "random")


def test_generated_dataset_covers_every_low_res_size(font_path, tmp_path):
    out = tmp_path / "lowres.npz"
    generate_dataset(
        font_path,
        LOW_RES_REGRESSION_CHARSET,
        samples_per_char=40,
        output=out,
        seed=7,
    )
    data = np.load(out)

    assert data["x"].shape[1:] == (24, 24)
    assert data["x"].dtype == np.uint8
    assert set(np.unique(data["x"])) <= {0, 255}
    assert set(data["render_sizes"].tolist()) == set(LOW_RES_SIZES)
    assert int(data["render_size_min"]) == 10
    assert int(data["render_size_max"]) == 18
    # Every sample went through a supersampled source before downsampling.
    assert np.all(data["source_sizes"] > data["render_sizes"])


def test_metadata_records_downsample_and_augmentations(font_path, tmp_path):
    out = tmp_path / "aug.npz"
    generate_dataset(
        font_path,
        LOW_RES_REGRESSION_CHARSET,
        samples_per_char=40,
        output=out,
        seed=11,
    )
    data = np.load(out)

    modes = {str(m) for m in data["downsample_modes"]}
    assert modes <= {"bilinear", "area"}
    assert modes == {"bilinear", "area"}

    tags = set()
    for row in data["augmentations"]:
        tags.update(str(row).split(","))
    required = {
        "subpixel",
        "downsample",
        "alpha",
        "brightness",
        "background",
        "scale",
        "outline",
        "blur",
        "shadow",
    }
    assert required <= tags, f"missing augmentation tags: {required - tags}"


@pytest.mark.parametrize("size", LOW_RES_SIZES)
def test_low_res_regression_chars_render_at_every_size(font_path, tmp_path, size):
    """Z17 / 巴尔的摩 must render and normalize at every Goal 7 size."""

    out = tmp_path / f"size_{size}.npz"
    generate_dataset(
        font_path,
        LOW_RES_REGRESSION_CHARSET,
        samples_per_char=1,
        output=out,
        seed=size,
        render_size_min=size,
        render_size_max=size,
    )
    data = np.load(out)
    assert np.all(data["render_sizes"] == size)
    assert np.all(data["x"].any(axis=(1, 2)))
    assert data["x"].shape[0] == len(LOW_RES_REGRESSION_CHARSET)


def test_low_res_regression_strings_form_a_detectable_line(font_path):
    """The full Z17 / 巴尔的摩 strings survive the low-res UI domain."""

    profile = default_profile()
    for text in ("Z17", "巴尔的摩"):
        for size in LOW_RES_SIZES:
            image = render(font_path, text, size)
            lines = find_lines(profile.color_mask(image), profile)
            assert len(lines) == 1, f"{text!r} at {size}px: {len(lines)} lines"


def test_soft_low_res_preserves_intermediate_intensity(font_path, tmp_path):
    out = tmp_path / "soft_lowres.npz"
    generate_dataset(
        font_path,
        list("Z17"),
        samples_per_char=12,
        output=out,
        seed=3,
        soft=True,
    )
    data = np.load(out)
    assert data["x"].dtype == np.uint8
    assert np.any((data["x"] > 0) & (data["x"] < 255))
    assert str(data["input_mode"][0]) == "soft"


def test_generator_source_implements_every_goal7_augmentation():
    src = (ROOT / "tools" / "dataset" / "generate_font_dataset.py").read_text(
        encoding="utf-8"
    )
    needles = (
        "rng.uniform(-0.4, 0.4)",  # sub-pixel x/y offset
        "font_scale",  # different scale ratios
        "Image.Resampling.BILINEAR",  # bilinear degradation
        "Image.Resampling.BOX",  # area-like degradation
        "ImageFilter.GaussianBlur",  # slight blur
        "alpha_composite",  # alpha / background blending
        "brightness",  # brightness
        "stroke_width",  # outline changes
        "scale_min",  # scale variation bounds
    )
    for needle in needles:
        assert needle in src, f"Goal 7 augmentation missing: {needle}"


def test_training_entry_points_default_to_low_res_domain():
    build_src = (ROOT / "tools" / "train" / "build_model.py").read_text(
        encoding="utf-8"
    )
    assert '"10"' in build_src and '"18"' in build_src
    assert "render-size-min" in build_src and "render-size-max" in build_src

    tiny_src = (ROOT / "scripts" / "train_tinycnn.py").read_text(
        encoding="utf-8"
    )
    assert "default=10" in tiny_src
    assert "default=18" in tiny_src
    assert "generate_dataset" in tiny_src
