"""P8: real game + synthetic regression set (tests/game_samples/)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fixedfontocr import FixedFontOCR
from fixedfontocr.defaults import (
    FONT_PATH,
    MODEL_PATH,
    compute_font_sha256,
    resolve_font,
)
from fixedfontocr.types import profile_from_dict


GAME_SAMPLES = Path(__file__).parent / "game_samples"
MANIFEST = GAME_SAMPLES / "manifest.json"

REQUIRED_CATEGORIES = {
    "synthetic/normal-chinese",
    "synthetic/digits",
    "synthetic/mixed-cn-ascii",
    "synthetic/punctuation",
    "synthetic/fragment-glyphs",
    "synthetic/confusable",
    "synthetic/light-background",
    "synthetic/dark-background",
    "synthetic/antialiasing",
    "synthetic/font-size",
    "real-game/level",
}


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sample_profile(manifest: dict, sample: dict):
    base = dict(manifest.get("default_profile", {}))
    base.update(sample.get("profile", {}))
    return profile_from_dict(base)


def test_manifest_font_is_registered():
    if not FONT_PATH.exists():
        pytest.skip("bundled font not present")
    manifest = _manifest()
    rel = manifest["font"]
    path = resolve_font(Path("fonts") / rel)
    assert compute_font_sha256(path) == compute_font_sha256(FONT_PATH)


def test_manifest_covers_required_categories():
    manifest = _manifest()
    categories = {s["category"] for s in manifest["samples"]}
    missing = REQUIRED_CATEGORIES - categories
    assert not missing, f"game samples missing categories: {sorted(missing)}"


def _ocr():
    if not (MODEL_PATH / "config.json").exists():
        pytest.skip("model/game_cn not built")
    return FixedFontOCR(model_path=MODEL_PATH, backend="cpu")


@pytest.mark.parametrize(
    "entry",
    [json.dumps(s, ensure_ascii=False) for s in _manifest()["samples"]],
    ids=[s["file"] for s in _manifest()["samples"]],
)
def test_game_sample_recognizes(entry):
    if not FONT_PATH.exists():
        pytest.skip("bundled font not present")
    sample = json.loads(entry)
    manifest = _manifest()
    path = GAME_SAMPLES / sample["file"]
    if not path.exists():
        pytest.skip(f"sample not committed: {path}")
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    ocr = _ocr()
    ocr.profile = _sample_profile(manifest, sample)
    result = ocr.recognize(image)
    assert result.text == sample["expected"], (
        f"{sample['file']}: expected {sample['expected']!r}, got {result.text!r}"
    )
    assert len(result.chars) == len(sample["expected"])
