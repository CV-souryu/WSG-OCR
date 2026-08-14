"""Unified candidate scoring for the segmentation candidate lattice.

The segmentation DP must compare candidates across two classifiers whose
raw scores live on different scales (template distance in ``[0,1]``,
TinyCNN logit margin unbounded). This module converts every candidate to a
shared 0..1 ``visual_score`` and keeps the raw score + source type in a
:class:`CandidateScore` so the two units are never averaged directly.

The hybrid rule follows the plan: use the template when it is both
confident *and* has a non-trivial top-1/top-2 margin; otherwise fall back to
the TinyCNN. A template with a high score but a tiny margin (e.g. 0.96 vs
0.95) is treated as ambiguous, exactly like the margin-gated hybrid design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .classifier import (
    DOWNSAMPLE_CLEAN,
    TemplateClassifier,
    TemplateV2Classifier,
)
from .cnn import prepare_weights
from .model import OCRModel
from .postprocess import second_ids as topk_second_ids
from .postprocess import top2
from .preprocess import (
    Component,
    NormalizeSpec,
    glyph_normalize_geometry,
    normalize,
    normalize_grayscale,
)
from .types import CandidateScore, ClassificationBatch, VisualScores


@dataclass(frozen=True)
class SegmentScore:
    """Score of one segmentation candidate plus its best character."""

    char_id: int
    visual_score: float
    raw_score: float
    score_type: str  # "template" | "cnn"
    visual_scores: VisualScores | None = None

    @property
    def candidate_score(self) -> CandidateScore:
        """View of this score as the plan's unified CandidateScore."""

        return CandidateScore(
            visual_score=self.visual_score,
            raw_score=self.raw_score,
            score_type=self.score_type,
        )


def cnn_visual(margin: float, scale: float = 1.0) -> float:
    """Map a CNN logit margin to a shared 0..1 visual score.

    Uses the sigmoid of the top-1/top-2 logit margin, i.e. the binary
    softmax probability that the top-1 class beats the top-2 class. A tie
    maps to 0.5 (not 0): a small-margin but correct CNN pick still
    contributes meaningfully to the segmentation path, while large margins
    approach 1 like a confident template match.
    """

    return float(1.0 / (1.0 + np.exp(-margin / scale)))


def to_confidence(raw: float, score_type: str) -> float:
    """Map a classifier raw score to the public 0..1 confidence."""

    if score_type == "template":
        return float(np.clip(raw, 0.0, 1.0))
    if score_type == "cnn":
        return cnn_visual(raw)
    return 0.0


def _template_visual_scores(tb, i: int) -> VisualScores:
    second_id = int(tb.second_ids[i]) if tb.second_ids is not None else -1
    return VisualScores(
        char_ids=(int(tb.ids[i]), second_id),
        logits=(float(tb.scores[i]), float(tb.second_scores[i])),
        top_k=(int(tb.ids[i]), second_id),
        margin=float(tb.margins[i]),
        raw_score=float(tb.dists[i]),
        score_type="template",
    )


def _cnn_visual_scores(batch: ClassificationBatch, i: int) -> VisualScores:
    second_id = int(batch.second_ids[i]) if batch.second_ids is not None else -1
    return VisualScores(
        char_ids=(int(batch.ids[i]), second_id),
        logits=(float(batch.top1[i]), float(batch.top2[i])),
        top_k=(int(batch.ids[i]), second_id),
        margin=float(batch.margins[i]),
        raw_score=float(batch.margins[i]),
        score_type="cnn",
    )


def _soft_glyph_batch(
    segments: list[Segment],
    soft: NDArray[np.uint8],
    target: int,
    geometries: list[tuple[float, float] | None] | None = None,
    spec: NormalizeSpec | None = None,
) -> np.ndarray:
    """Crop + normalize soft ROIs for every candidate segment."""

    if not segments:
        return np.empty((0, target, target), dtype=np.uint8)
    geoms = geometries or [None] * len(segments)
    row = spec.baseline_row if spec is not None else 18.0
    return np.stack(
        [
            (
                normalize_grayscale(
                    soft[int(s.y) : int(s.y + s.h), int(s.x) : int(s.x + s.w)],
                    target,
                )
                if g is None
                else normalize_grayscale(
                    soft[int(s.y) : int(s.y + s.h), int(s.x) : int(s.x + s.w)],
                    target,
                    baseline_offset=g[0],
                    scale=g[1],
                    baseline_row=row,
                )
            )
            for s, g in zip(segments, geoms)
        ]
    )


class SegmentScorer:
    """Batch-scorer used by the segmentation DP for one model."""

    def __init__(self, model: OCRModel):
        self.model = model
        self.input_size = model.input_size
        self.normalize_spec = model.normalize_spec
        if model.templates_v2 is not None:
            self.template = TemplateV2Classifier(
                data=model.templates_v2,
                charset=model.charset,
                input_size=model.input_size,
                normalize_spec=model.normalize_spec,
            )
        elif model.templates is not None:
            self.template = TemplateClassifier(
                templates=model.templates,
                charset=model.charset,
                input_size=model.input_size,
                normalize_spec=model.normalize_spec,
            )
        else:
            self.template = None
        self.weights = prepare_weights(model.weights) if model.weights else None
        self.template_threshold = float(model.config.get("template_threshold", 0.90))
        self.template_margin_threshold = float(
            model.config.get("template_margin_threshold", 0.04)
        )
        self.cnn_threshold = float(model.config.get("cnn_threshold", 0.0))
        self.cnn_input_mode = model.input_mode
        self.geometry = model.geometry
        # CNN-only models cannot lean on template confidence; a merged
        # candidate must look like a real character before the DP may prefer
        # it over its components (see segmentation._drop_weak_merges).
        self.cnn_merge_threshold = (
            0.0 if self.template is not None else float(model.config.get("cnn_merge_threshold", 0.7))
        )

    def score(
        self,
        segments: list[Segment],
        allowed_ids: set[int] | None = None,
        soft: NDArray[np.uint8] | None = None,
        geometries: list[tuple[float, float] | None] | None = None,
    ) -> list[SegmentScore]:
        """Score every candidate segment in one batched pass.

        ``soft`` is the Visual Frontend's ``soft_foreground`` map. When the
        model's ``input_mode`` is ``"soft"`` the TinyCNN consumes
        soft-normalized glyphs (which preserve anti-aliasing/edge
        intensity); the template path always consumes binary-normalized
        glyphs. Binary-trained models (``input_mode="binary"``) ignore the
        soft map, so old checkpoints keep their exact training-domain
        inputs.
        """

        if not segments:
            return []
        spec = self.normalize_spec
        if geometries is None and spec is not None:
            geometries = [
                glyph_normalize_geometry(
                    Component(mask=s.mask, x=s.x, y=s.y, w=s.w, h=s.h),
                    spec,
                    self.input_size,
                )
                for s in segments
            ]
        geoms = geometries or [None] * len(segments)
        if spec is None:
            glyphs = np.stack([normalize(s.mask, self.input_size) for s in segments])
        else:
            glyphs = np.stack(
                [
                    (
                        normalize(s.mask, self.input_size)
                        if g is None
                        else normalize(
                            s.mask,
                            self.input_size,
                            baseline_offset=g[0],
                            scale=g[1],
                            baseline_row=spec.baseline_row,
                        )
                    )
                    for s, g in zip(segments, geoms)
                ]
            )
        soft_batch = (
            _soft_glyph_batch(segments, soft, self.input_size, geoms, spec)
            if soft is not None and self.cnn_input_mode == "soft"
            else None
        )
        n = len(segments)

        if self.template is not None:
            tb = self.template.match_batch(glyphs, allowed_ids)
            if self.weights is None:
                return [
                    SegmentScore(
                        char_id=int(tb.ids[i]),
                        visual_score=float(tb.scores[i]),
                        raw_score=float(tb.dists[i]),
                        score_type="template",
                        visual_scores=_template_visual_scores(tb, i),
                    )
                    for i in range(n)
                ]
            # Trust a template when it is confident AND either it is an
            # exact match (conf == 1.0, always reliable) or its top-1/top-2
            # margin is non-trivial. Small punctuation (！？；：) often has
            # conf 1.0 with a tiny normalized margin because the glyphs are
            # tiny, so exact matches bypass the margin floor.
            needs_cnn = (tb.scores < self.template_threshold) | (
                (tb.scores < 1.0)
                & (tb.margins < self.template_margin_threshold)
            )
            if isinstance(self.template, TemplateV2Classifier):
                # A pixel-exact low-res prototype can be shared by two
                # different characters (e.g. '.' and '*' both rasterize to
                # the same tiny blob at 11 px), so an exact match with zero
                # margin is only trustworthy when it came from the clean
                # high-res prototype; everything else goes to the CNN.
                needs_cnn = needs_cnn | (
                    (tb.scores >= 1.0)
                    & (tb.margins <= 0.0)
                    & (tb.prototype_downsample_modes != DOWNSAMPLE_CLEAN)
                )
            if np.any(needs_cnn):
                cnn = self._cnn_batch(glyphs, allowed_ids, soft_batch)
            out: list[SegmentScore] = []
            for i in range(n):
                if needs_cnn[i]:
                    out.append(
                        SegmentScore(
                            char_id=int(cnn.ids[i]),
                            visual_score=cnn_visual(float(cnn.margins[i])),
                            raw_score=float(cnn.margins[i]),
                            score_type="cnn",
                            visual_scores=_cnn_visual_scores(cnn, i),
                        )
                    )
                else:
                    out.append(
                        SegmentScore(
                            char_id=int(tb.ids[i]),
                            visual_score=float(tb.scores[i]),
                            raw_score=float(tb.dists[i]),
                            score_type="template",
                            visual_scores=_template_visual_scores(tb, i),
                        )
                    )
            return out

        if self.weights is not None:
            cnn = self._cnn_batch(glyphs, allowed_ids, soft_batch)
            return [
                SegmentScore(
                    char_id=int(cnn.ids[i]),
                    visual_score=cnn_visual(float(cnn.margins[i])),
                    raw_score=float(cnn.margins[i]),
                    score_type="cnn",
                    visual_scores=_cnn_visual_scores(cnn, i),
                )
                for i in range(n)
            ]
        raise ValueError("model has neither templates nor CNN weights")

    def _cnn_batch(
        self,
        glyphs: np.ndarray,
        allowed_ids: set[int] | None,
        soft_batch: np.ndarray | None = None,
    ) -> ClassificationBatch:
        from .cnn import forward

        if allowed_ids is not None and not allowed_ids:
            n = soft_batch.shape[0] if soft_batch is not None else glyphs.shape[0]
            return ClassificationBatch(
                ids=np.full(n, -1, dtype=np.int32),
                top1=np.full(n, -np.inf, dtype=np.float32),
                top2=np.full(n, -np.inf, dtype=np.float32),
                margins=np.full(n, 0.0, dtype=np.float32),
                second_ids=np.full(n, -1, dtype=np.int32),
            )
        cnn_input = soft_batch if soft_batch is not None else glyphs
        x = cnn_input.astype(np.float32)[:, None, :, :] * (1.0 / 255.0)
        logits = forward(x, self.weights)
        if allowed_ids is not None:
            masked = np.full_like(logits, -np.inf)
            idx = np.fromiter(sorted(allowed_ids), dtype=np.int64)
            masked[:, idx] = logits[:, idx]
            logits = masked
        ids, top1, top2v, margins = top2(logits)
        return ClassificationBatch(
            ids=ids,
            top1=top1,
            top2=top2v,
            margins=margins,
            second_ids=topk_second_ids(logits, ids),
        )
