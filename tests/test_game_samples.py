"""P8/Goal 18: real game + synthetic regression set (tests/game_samples/)."""

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
    "synthetic/goal14-fragments",
    "synthetic/goal14-lowres",
    "synthetic/goal14-mixed",
    "synthetic/goal14-digits",
    "synthetic/goal14-words",
    "synthetic/goal14-partial",
    "real-game/level",
    "real-game/ship-name",
}

# Goal 18 first milestone: real game crops, not a handful of screenshots.
MIN_REAL_GAME_SAMPLES = 100


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


def test_real_game_corpus_size():
    manifest = _manifest()
    real = [s for s in manifest["samples"] if s["category"].startswith("real-game")]
    assert len(real) >= MIN_REAL_GAME_SAMPLES, (
        f"Goal 18 requires at least {MIN_REAL_GAME_SAMPLES} real game samples, "
        f"manifest has {len(real)}"
    )
    missing = [s["file"] for s in real if not (GAME_SAMPLES / s["file"]).exists()]
    assert not missing, f"real-game samples missing image files: {missing[:5]}"


def test_real_game_manifest_schema():
    manifest = _manifest()
    for sample in manifest["samples"]:
        if not sample["category"].startswith("real-game"):
            continue
        assert sample.get("expected"), sample
        assert (GAME_SAMPLES / sample["file"]).is_file(), sample["file"]
        assert sample["category"] in ("real-game/level", "real-game/ship-name")


def test_known_failures_are_documented():
    manifest = _manifest()
    known = [s for s in manifest["samples"] if s.get("known_failure")]
    for sample in known:
        assert sample.get("expected"), sample["file"]
        assert sample.get("known_failure_note"), sample["file"]


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
    if sample.get("lexicon"):
        result = ocr.recognize(
            image,
            lexicon=sample["lexicon"],
            lexicon_mode=sample.get("lexicon_mode", "prefer"),
        )
    else:
        result = ocr.recognize(image)
    if sample.get("known_failure"):
        # Real regression sample: keep it in the corpus and pin that the
        # current model still misses it. When the model starts reading it,
        # remove the flag and it becomes an ordinary passing assertion.
        assert result.text != sample["expected"], (
            f"{sample['file']}: marked known_failure but now recognized "
            f"{result.text!r}; remove the flag"
        )
        return
    assert result.text == sample["expected"], (
        f"{sample['file']}: expected {sample['expected']!r}, got {result.text!r}"
    )
    assert len(result.chars) == len(sample["expected"])
    if "matched_term" in sample:
        assert result.matched_term == sample["matched_term"], (
            f"{sample['file']}: expected matched_term "
            f"{sample['matched_term']!r}, got {result.matched_term!r}"
        )
        assert result.matched_span == tuple(sample["matched_span"]), (
            f"{sample['file']}: expected matched_span "
            f"{sample['matched_span']!r}, got {result.matched_span!r}"
        )
