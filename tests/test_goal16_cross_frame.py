"""Goal 16 acceptance: cross-frame tracking, decoupled from single-frame OCR.

The Goal 16 contract from ``fonts/goal``:

* ``ocr.recognize(image)`` stays a pure single-frame function;
* a separate ``tracker.update(image)`` API provides ROI change detection,
  result caching, multi-frame logits fusion and stable text voting;
* e.g. ``巴尔的摩 / 巴你的摩 / 巴尔的摩`` stabilizes to ``巴尔的摩``;
* temporal state never pollutes the baseline OCR tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixedfontocr import (
    CharResult,
    Component,
    DecodePath,
    FixedFontOCR,
    FrameTracker,
    OCRResult,
    TrackerConfig,
    VisualCandidate,
    VisualScores,
)

from conftest import render_text


CHARSET = list("A巴你的摩八尔")


def _image(seed: int = 0) -> np.ndarray:
    """Black background with the same white-text ROI and a tiny content diff."""

    img = np.zeros((30, 80, 3), dtype=np.uint8)
    img[5:25, 20:60] = 255
    img[8 + seed % 6, 28 + seed % 6, :] = 0
    return img


def _result(text: str, logs: tuple[dict[str, float], ...]) -> OCRResult:
    chars = tuple(
        CharResult(ch, i * 20, 0, 12, 12, 0.9)
        for i, ch in enumerate(text)
    )
    candidates = []
    for i, (ch, logits) in enumerate(zip(text, logs)):
        ids = tuple(CHARSET.index(c) for c in logits)
        values = tuple(logits.values())
        candidates.append(
            VisualCandidate(
                start=i,
                end=i + 1,
                components=(i,),
                atoms=(i, i + 1),
                segment=Component(
                    np.ones((12, 12), dtype=bool),
                    i * 20,
                    0,
                    12,
                    12,
                ),
                scores=VisualScores(char_ids=ids, logits=values),
            )
        )
    path = DecodePath(candidates=tuple(candidates), text=text)
    return OCRResult(
        text=text,
        confidence=0.9,
        chars=chars,
        path=path,
    )


class _FakeOCR:
    def __init__(self, results, charset=CHARSET):
        self.results = list(results)
        self.charset = list(charset)
        self.i = 0
        self.calls = 0

    def recognize(self, image, **kwargs):
        self.calls += 1
        result = self.results[min(self.i, len(self.results) - 1)]
        self.i += 1
        return result


def test_tracker_config_validation():
    with pytest.raises(ValueError, match="window"):
        TrackerConfig(window=0)
    with pytest.raises(ValueError, match="min_frames"):
        TrackerConfig(min_frames=0)
    with pytest.raises(ValueError, match="vote_ratio"):
        TrackerConfig(vote_ratio=1.5)
    with pytest.raises(ValueError, match="cache_size"):
        TrackerConfig(cache_size=0)


def test_unchanged_roi_is_cached_without_running_ocr():
    fake = _FakeOCR([_result("A", ({"A": 1.0},))])
    tracker = FrameTracker(fake)

    first = tracker.update(_image(1))
    second = tracker.update(_image(1))

    assert first.text == "A"
    assert second.text == "A"
    assert fake.calls == 1
    assert tracker.hits == 1
    assert tracker.misses == 1
    assert tracker.frames == 2
    assert tracker.roi_changed is False


def test_roi_change_detection_flags_different_content():
    fake = _FakeOCR([_result("A", ({"A": 1.0},))])
    tracker = FrameTracker(fake)

    tracker.update(_image(1))
    assert tracker.roi_changed is False  # first frame has no baseline
    tracker.update(_image(2))
    assert tracker.roi_changed is True
    tracker.update(_image(2))
    assert tracker.roi_changed is False


def test_stable_text_voting_suppresses_flicker():
    good = _result(
        "巴尔的摩",
        (
            {"巴": 0.9, "八": 0.7},
            {"尔": 0.9, "你": 0.8},
            {"的": 0.9},
            {"摩": 0.9},
        ),
    )
    bad = _result(
        "巴你的摩",
        (
            {"巴": 0.9, "八": 0.7},
            {"你": 0.95, "尔": 0.94},
            {"的": 0.9},
            {"摩": 0.9},
        ),
    )
    fake = _FakeOCR([good, bad, good])
    tracker = FrameTracker(fake)

    tracker.update(_image(1))
    tracker.update(_image(2))
    third = tracker.update(_image(3))

    assert third.text == "巴尔的摩"
    assert tracker.stable_text == "巴尔的摩"
    assert fake.calls == 3


def test_multiframe_logits_fusion_corrects_before_voting_majority():
    good = _result(
        "巴尔的摩",
        (
            {"巴": 0.9, "八": 0.7},
            {"尔": 0.9, "你": 0.8},
            {"的": 0.9},
            {"摩": 0.9},
        ),
    )
    bad = _result(
        "巴你的摩",
        (
            {"巴": 0.9, "八": 0.7},
            {"你": 0.95, "尔": 0.94},
            {"的": 0.9},
            {"摩": 0.9},
        ),
    )
    fake = _FakeOCR([good, bad])
    # Voting needs more frames than we will provide; fusion only needs 2.
    tracker = FrameTracker(fake, min_frames=10, fusion_min_frames=2)

    tracker.update(_image(1))
    fused = tracker.update(_image(2))

    assert fused.text == "巴尔的摩"
    assert tracker.fused_text == "巴尔的摩"
    assert tracker.stable_text == ""


def test_tracker_reset_clears_temporal_state():
    fake = _FakeOCR([_result("A", ({"A": 1.0},))])
    tracker = FrameTracker(fake)
    tracker.update(_image(1))
    tracker.update(_image(1))
    assert tracker.frames == 2
    assert tracker.hits == 1
    assert tracker.cache_size == 1

    tracker.reset()

    assert tracker.frames == 0
    assert tracker.hits == 0
    assert tracker.misses == 0
    assert tracker.cache_size == 0
    assert tracker.last_result is None
    assert tracker.last_roi_box is None


def test_tracker_accepts_keyword_roi():
    fake = _FakeOCR([_result("A", ({"A": 1.0},))])
    tracker = FrameTracker(fake)
    img = _image(1)
    first = tracker.update(img, roi=(20, 5, 40, 20))
    second = tracker.update(img, roi=(20, 5, 40, 20))
    assert first.text == "A"
    assert second.text == "A"
    assert fake.calls == 1


def test_ocr_tracker_method_and_pure_recognize(font_path, model_dir):
    """``ocr.tracker()`` exists and ``recognize`` is unaffected by tracker state."""

    ocr = FixedFontOCR(model_path=model_dir, backend="cpu", auto_benchmark=False)
    image = render_text("A", font_path, font_size=24)
    baseline = ocr.recognize(image)
    assert baseline.text == "A"

    tracker = ocr.tracker()
    assert isinstance(tracker, FrameTracker)
    tracked = tracker.update(image)
    assert tracked.text == "A"

    after = ocr.recognize(image)
    assert after.text == baseline.text
    assert after.confidence == baseline.confidence
