"""Shared data types for the OCR pipeline.

Goal 1 defines the lattice/decoder vocabulary: connected components,
visual candidates, the lattice that holds them, per-candidate Top-K
visual scores, decoded paths and lexicon matches. ``CharResult`` remains
only as a public compatibility projection; the internal pipeline works
with the new structures below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _hsl_lightness_saturation(
    image: NDArray[np.uint8],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Convert an RGB image to HSL lightness/saturation in [0, 1].

    Hue is intentionally not computed: the game text is white (low
    saturation, high lightness) and the surrounding UI is colored, so
    ``(lightness, saturation)`` alone separates ink from background without
    depending on hue.
    """

    rgb = image.astype(np.float32) / 255.0
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    lightness = (maxc + minc) / 2.0
    delta = maxc - minc
    denom = 1.0 - np.abs(2.0 * lightness - 1.0)
    saturation = np.zeros_like(lightness)
    colored = delta > 0
    saturation[colored] = delta[colored] / np.maximum(denom[colored], 1e-6)
    return lightness, saturation


@dataclass(frozen=True)
class CharResult:
    """A single recognized character and its location in the source image."""

    char: str
    x: int
    y: int
    w: int
    h: int
    confidence: float


@dataclass
class Component:
    """One connected component in a text line.

    ``x/y/w/h`` describe the component's tight bounding box in image
    coordinates; ``mask`` is the component bitmap (same shape as the bbox).
    Components are the atomic units of the visual lattice: a candidate can
    cover one component, a consecutive run of components, or a split of a
    single wide component.
    """

    mask: NDArray[np.bool_]
    x: int
    y: int
    w: int
    h: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def ink(self) -> int:
        return int(self.mask.sum())


@dataclass(frozen=True)
class VisualScores:
    """Top-K classifier scores for one visual candidate.

    Top-1 alone is never enough for lattice decoding: the decoder needs the
    competing glyphs and their raw scores so it can let the lexicon or a
    segmentation hypothesis override a marginally-better classifier pick.
    ``char_ids``/``logits`` are the ranked Top-K lists and ``top_k`` mirrors
    ``char_ids`` for callers that prefer that name. Since Goal 10 the object
    also keeps the unified evidence for the chosen character: the raw
    template score (0..1), the raw CNN logit and margin, the geometry
    penalty, and the weighted ``visual_score`` used by the decoder.
    """

    char_ids: tuple[int, ...] = ()
    logits: tuple[float, ...] = ()
    top_k: tuple[int, ...] = ()
    margin: float = 0.0
    raw_score: float = 0.0
    score_type: str = ""
    # Goal 10 unified visual evidence. ``logits`` for the unified scorer are
    # the combined top-K values; the raw classifier quantities are kept in
    # these fields so no scale is ever collapsed into a single number.
    template_raw_score: float = 0.0
    cnn_logit: float = 0.0
    cnn_margin: float = 0.0
    cnn_score: float = 0.0
    geometry_score: float = 0.0
    visual_score: float = 0.0
    confidence: float = 0.0
    geometry_included: bool = False

    @property
    def template_score(self) -> float:
        """Alias for the raw template confidence in [0, 1]."""
        return self.template_raw_score

    def __post_init__(self) -> None:
        if not self.top_k:
            object.__setattr__(self, "top_k", self.char_ids)


@dataclass
class VisualCandidate:
    """One hypothesis that a consecutive component range is one glyph.

    A candidate may be a single component (``C0``), a merged run
    (``C0+C1+C2``), or a split of a wide component (``C0 -> A+B``, where the
    split parts are represented as separate candidates). ``bbox`` is the
    image-space ``(x, y, w, h)`` box; ``glyph`` is the normalized bitmap
    consumed by the template/CNN scorer when available. ``atoms`` records the
    candidate's half-open atom range (atoms are whole components or split
    pieces), which is what the DP covers; ``start``/``end`` keep the original
    component range for compatibility.

    ``score`` is kept as a compatibility view of the unified scorer output;
    the canonical Top-K data lives in ``scores``.
    """

    start: int
    end: int
    bbox: tuple[int, int, int, int] | None = None
    glyph: NDArray[np.uint8] | None = None
    components: tuple[int, ...] = ()
    atoms: tuple[int, int] | None = None
    segment: Component | None = None
    normalize_geometry: tuple[float, float] | None = None
    geometry_score: float = 0.0
    scores: VisualScores | None = None
    score: Any | None = None

    def __post_init__(self) -> None:
        if self.bbox is None and self.segment is not None:
            self.bbox = (self.segment.x, self.segment.y, self.segment.w, self.segment.h)

    @property
    def geometry(self) -> float:
        """Compatibility alias for :attr:`geometry_score`."""
        return self.geometry_score

    @geometry.setter
    def geometry(self, value: float) -> None:
        self.geometry_score = float(value)

    @property
    def total_score(self) -> float:
        """Unified candidate score used by the visual DP.

        The scorer stores the final weighted visual score (including the
        geometry term) directly on ``score.visual_score`` for candidates it
        produced, so the DP must not add ``geometry_score`` a second time.
        Manually-constructed legacy scores (without ``geometry_included``)
        keep the old ``visual + geometry`` behaviour.
        """
        if self.score is not None:
            visual = float(getattr(self.score, "visual_score", self.score))
            if getattr(self.score, "geometry_included", False):
                return visual
            return visual + self.geometry_score
        if self.scores is not None:
            if self.scores.geometry_included:
                return self.scores.visual_score
            if self.scores.visual_score:
                return self.scores.visual_score + self.geometry_score
            return self.scores.margin + self.geometry_score
        return self.geometry_score

    @property
    def atom_span(self) -> tuple[int, int]:
        """Half-open atom range used by the decoder.

        An atom is either one whole connected component or one piece of a
        split component. Candidates created without explicit split metadata
        (legacy/manual constructions) fall back to their component range.
        """

        if self.atoms is not None:
            return self.atoms
        return (self.start, self.end)


@dataclass
class VisualLattice:
    """The full candidate lattice of one line.

    Keeps every original connected component and every generated visual
    candidate, so the decoder can still choose a different merge/split path
    after scoring. Nothing here is an irreversible decision.
    """

    components: tuple[Component, ...] = ()
    candidates: tuple[VisualCandidate, ...] = ()
    width: int = 0
    height: int = 0

    def by_end(self) -> dict[int, list[VisualCandidate]]:
        """Index candidates by their exclusive end atom index."""
        out: dict[int, list[VisualCandidate]] = {}
        for cand in self.candidates:
            out.setdefault(cand.atom_span[1], []).append(cand)
        return out


@dataclass(frozen=True)
class DecodePath:
    """Best (or alternative) decoder path through a visual lattice."""

    candidates: tuple[VisualCandidate, ...] = field(default_factory=tuple)
    mean_score: float = 0.0
    text: str = ""
    confidence: float = 0.0
    alternatives: tuple[str, ...] = field(default_factory=tuple)
    # Goal 13: the charset id chosen for every candidate (``-1`` = unknown).
    # This is what makes the decoder authoritative -- a lexicon-aware path
    # may pick the second-ranked character of a candidate, and the caller
    # must not silently fall back to Top-1.
    char_ids: tuple[int, ...] = field(default_factory=tuple)
    lattice: VisualLattice | None = None


@dataclass(frozen=True)
class LexiconMatch:
    """A dictionary match over a span of visible text.

    ``term`` is the full entity inferred from the lexicon; ``span`` is the
    half-open character range inside ``term`` that the visible text covers.
    For a cropped/partial view this follows the Goal 12 convention
    (``text="尔的摩"`` / ``term="巴尔的摩"`` / ``span=(1, 4)``), while
    ``text_span`` records the same range inside the visible text when the
    term is only a substring of it. The dictionary never invents or
    overwrites visible characters on its own: ``prefer`` may replace a
    character only when that replacement is already among the candidate's
    visual Top-K and the original visual evidence was uncertain. ``kind``
    names the alignment used by the matcher (``exact``, ``term_in_text``,
    ``prefix_crop``, ``suffix_crop``, ``inner_crop`` or ``gap_crop``).
    """

    term: str
    span: tuple[int, int]
    confidence: float = 0.0
    score: float = 0.0
    mode: str = "none"
    text: str = ""
    text_span: tuple[int, int] | None = None
    kind: str = "exact"


@dataclass(frozen=True)
class OCRResult:
    """Full recognition result for one image.

    ``text`` is exactly what is visible on screen. ``matched_term`` (and
    ``lexicon_match``) are dictionary-inferred entities and must never be
    conflated with the visible text; ``matched_span`` is the half-open
    range inside ``matched_term`` that the visible text covers (Goal 12).
    ``alternatives`` holds alternate decoder outputs, and ``path`` retains
    the chosen lattice path for debugging/inspection.
    """

    text: str
    confidence: float
    chars: tuple[CharResult, ...] = field(default_factory=tuple)
    matched_term: str | None = None
    matched_span: tuple[int, int] | None = None
    alternatives: tuple[str, ...] = field(default_factory=tuple)
    lexicon_match: LexiconMatch | None = None
    path: DecodePath | None = None


@dataclass(frozen=True)
class ClassificationBatch:
    """Top-K classifier output for a batch of glyphs.

    ``ids`` is the top-1 character id, ``top1``/``top2`` are the raw
    classifier scores of the first two candidates and ``margins`` is
    ``top1 - top2``. ``second_ids`` carries the second-ranked id so callers
    can build a real Top-K :class:`VisualScores` (never only Top-1).
    When a caller asks for ``top_k`` > 2, ``topk_ids``/``topk_logits``
    carry the full ranked Top-K lists as ``[N, k]`` arrays; the first two
    columns are the same values as the scalar fields.
    """

    ids: np.ndarray  # int32 [N]
    top1: np.ndarray  # f32 [N]
    top2: np.ndarray  # f32 [N]
    margins: np.ndarray  # f32 [N]
    second_ids: np.ndarray | None = None  # int32 [N]
    topk_ids: np.ndarray | None = None  # int32 [N, k]
    topk_logits: np.ndarray | None = None  # f32 [N, k]
    logits: np.ndarray | None = None  # f32 [N, C] full logits (Goal 10)


@dataclass(frozen=True)
class CandidateScore:
    """Unified score of one segmentation candidate.

    ``visual_score`` lives on a shared 0..1 scale and is what the
    segmentation DP compares across candidates. ``raw_score`` keeps the
    classifier-specific quantity (template Hamming distance or CNN logit
    margin), ``score_type`` names the dominant source, and the Goal 10
    fields keep the template/CNN/geometry evidence that produced the
    weighted ``visual_score``.
    """

    visual_score: float
    raw_score: float
    score_type: str
    template_raw_score: float = 0.0
    cnn_logit: float = 0.0
    cnn_margin: float = 0.0
    cnn_score: float = 0.0
    geometry_score: float = 0.0


@dataclass(frozen=True)
class Profile:
    """Per-UI-type configuration used by the preprocessing pipeline."""

    name: str
    # Color extraction: an RGB target plus tolerance (max per-channel distance).
    target_color: tuple[int, int, int] | None = None
    tolerance: int = 40
    use_grayscale: bool = True
    grayscale_threshold: int = 140
    # HSL-based binarization: game text is mostly white, i.e. high
    # lightness and low saturation, while UI frames/backgrounds are
    # colored. When ``use_hsl`` is True the grayscale/RGB branches are
    # ignored.
    use_hsl: bool = False
    hsl_lightness_min: float = 0.55
    hsl_saturation_max: float = 0.70
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
        if self.use_hsl:
            lightness, saturation = _hsl_lightness_saturation(image)
            sat_ok = saturation <= self.hsl_saturation_max
            if self.bright_text:
                return (lightness >= self.hsl_lightness_min) & sat_ok
            return (lightness <= self.hsl_lightness_min) & sat_ok
        if self.use_grayscale or self.target_color is None:
            gray = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            if self.bright_text:
                return gray >= self.grayscale_threshold
            return gray <= self.grayscale_threshold

        diff = np.abs(image.astype(np.int16) - np.asarray(self.target_color, dtype=np.int16))
        return np.all(diff <= self.tolerance, axis=-1)

    def soft_foreground(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Return a 0..255 foreground-strength map without a hard cutoff.

        ``color_mask`` keeps its hard decision for segmentation and the
        template path; this map is the *soft* twin that preserves
        anti-aliasing, alpha, edge gray and low-resolution information for
        the TinyCNN. Values are 0 (definitely background) .. 255
        (definitely text-colored), with anti-aliased edge pixels retaining
        their intermediate intensity instead of being snapped to a
        threshold.
        """

        if self.use_hsl:
            lightness, saturation = _hsl_lightness_saturation(image)
            sat_atten = 1.0 - np.clip(
                saturation / max(self.hsl_saturation_max, 1e-6), 0.0, 1.0
            )
            if self.bright_text:
                soft = lightness * sat_atten
            else:
                soft = (1.0 - lightness) * sat_atten
            return (np.clip(soft, 0.0, 1.0) * 255.0).astype(np.uint8)

        if self.use_grayscale or self.target_color is None:
            gray = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            if self.bright_text:
                return np.clip(gray, 0, 255).astype(np.uint8)
            return np.clip(255.0 - gray, 0, 255).astype(np.uint8)

        diff = np.abs(
            image.astype(np.int16) - np.asarray(self.target_color, dtype=np.int16)
        ).max(axis=-1)
        return np.clip(255 - diff, 0, 255).astype(np.uint8)


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
        use_hsl=bool(data.get("use_hsl", False)),
        hsl_lightness_min=float(data.get("hsl_lightness_min", 0.55)),
        hsl_saturation_max=float(data.get("hsl_saturation_max", 0.70)),
        char_height_min=int(data.get("char_height_min", 8)),
        char_height_max=int(data.get("char_height_max", 64)),
        char_width_min=int(data.get("char_width_min", 3)),
        char_width_max=int(data.get("char_width_max", 72)),
        stroke_width=int(data.get("stroke_width", 1)),
        char_spacing=int(data.get("char_spacing", 1)),
        target_size=int(data.get("target_size", 24)),
        bright_text=bool(data.get("bright_text", True)),
    )
