"""Shared data types for the OCR pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CharResult:
    """A single recognized character and its location in the source image."""

    char: str
    x: int
    y: int
    w: int
    h: int
    confidence: float


@dataclass(frozen=True)
class OCRResult:
    """Full recognition result for one image."""

    text: str
    confidence: float
    chars: tuple[CharResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ClassificationBatch:
    """Top-K classifier output for a batch of glyphs.

    ``ids`` is the top-1 character id, ``top1``/``top2`` are the raw
    classifier scores of the first two candidates and ``margins`` is
    ``top1 - top2``. The API supports ``top_k`` > 2 for future dictionary
    decoding without changing the CNN itself.
    """

    ids: np.ndarray  # int32 [N]
    top1: np.ndarray  # f32 [N]
    top2: np.ndarray  # f32 [N]
    margins: np.ndarray  # f32 [N]


@dataclass(frozen=True)
class CandidateScore:
    """Unified score of one segmentation candidate.

    ``visual_score`` lives on a shared 0..1 scale and is what the
    segmentation DP compares across candidates. ``raw_score`` keeps the
    classifier-specific quantity (template Hamming distance or CNN logit
    margin) and ``score_type`` names the source, so two different units are
    never averaged directly.
    """

    visual_score: float
    raw_score: float
    score_type: str


@dataclass(frozen=True)
class Profile:
    """Per-UI-type configuration used by the preprocessing pipeline."""

    name: str
    # Color extraction: an RGB target plus tolerance (max per-channel distance).
    target_color: tuple[int, int, int] | None = None
    tolerance: int = 40
    use_grayscale: bool = True
    grayscale_threshold: int = 140
    # Character size expectations (used for merging/splitting components).
    char_height_min: int = 8
    char_height_max: int = 64
    char_width_min: int = 4
    char_width_max: int = 64
    stroke_width: int = 1
    char_spacing: int = 1
    # Normalized output size.
    target_size: int = 24
    # Ink polarity: True means text pixels are brighter than the background.
    bright_text: bool = True

    def color_mask(self, image: NDArray[np.uint8]) -> NDArray[np.bool_]:
        """Return a boolean mask of pixels matching this profile's text color."""
        if self.use_grayscale or self.target_color is None:
            gray = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            if self.bright_text:
                return gray >= self.grayscale_threshold
            return gray <= self.grayscale_threshold

        diff = np.abs(image.astype(np.int16) - np.asarray(self.target_color, dtype=np.int16))
        return np.all(diff <= self.tolerance, axis=-1)


def default_profile() -> Profile:
    return Profile(
        name="default",
        target_color=(255, 255, 255),
        tolerance=48,
        use_grayscale=True,
        grayscale_threshold=140,
        char_height_min=8,
        char_height_max=64,
        char_width_min=3,
        char_width_max=72,
        stroke_width=1,
        char_spacing=1,
        target_size=24,
        bright_text=True,
    )


def profile_from_dict(data: dict) -> Profile:
    """Build a :class:`Profile` from a manifest-style dict."""

    return Profile(
        name=str(data.get("name", "default")),
        target_color=tuple(data["target_color"]) if "target_color" in data else None,
        tolerance=int(data.get("tolerance", 40)),
        use_grayscale=bool(data.get("use_grayscale", True)),
        grayscale_threshold=int(data.get("grayscale_threshold", 140)),
        char_height_min=int(data.get("char_height_min", 8)),
        char_height_max=int(data.get("char_height_max", 64)),
        char_width_min=int(data.get("char_width_min", 3)),
        char_width_max=int(data.get("char_width_max", 72)),
        stroke_width=int(data.get("stroke_width", 1)),
        char_spacing=int(data.get("char_spacing", 1)),
        target_size=int(data.get("target_size", 24)),
        bright_text=bool(data.get("bright_text", True)),
    )
