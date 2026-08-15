"""Unified candidate scoring for the segmentation candidate lattice.

Goal 10 moves past the old "pick one classifier" hybrid gate: every
candidate carries the raw template score, raw CNN logit/margin and geometry
score, and the decoder consumes one weighted ``visual_score``:

    visual_score = a * cnn_score + b * template_score + c * geometry_score

The weights live in ``visual_weights`` (model config, tunable on a real test
set) and the public confidence is mapped to ``[0, 1]`` through an optional
per-model calibration. The old margin-aware template trust rule is retained
only as a *label* for the dominant source (``score_type``) and for the
unknown-char threshold; it no longer decides which classifier is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .backends import Backend, CPUBackend
from .classifier import (
    DOWNSAMPLE_CLEAN,
    TemplateClassifier,
    TemplateV2Classifier,
)
from .cnn import prepare_weights
from .model import OCRModel
from .postprocess import second_ids as topk_second_ids
from .postprocess import top2, topk
from .preprocess import (
    Component,
    NormalizeSpec,
    glyph_normalize_geometry,
    normalize,
    normalize_grayscale,
)
from .types import CandidateScore, ClassificationBatch, VisualCandidate, VisualScores


@dataclass(frozen=True)
class VisualWeights:
    """Goal 10 weighted visual scoring coefficients.

    ``visual_score = cnn_weight * cnn_score + template_weight *
    template_score + geometry_weight * geometry_score``. For a hybrid model
    all three are active; for a template-only or CNN-only model the missing
    classifier's term is zero and the available classifier score is used
    unchanged.
    """

    cnn: float = 0.45
    template: float = 0.45
    geometry: float = 0.10

    @classmethod
    def from_config(cls, config: dict | None) -> "VisualWeights":
        raw = (config or {}).get("visual_weights", {})
        return cls(
            cnn=float(raw.get("cnn", cls.cnn)),
            template=float(raw.get("template", cls.template)),
            geometry=float(raw.get("geometry", cls.geometry)),
        )


@dataclass(frozen=True)
class VisualCalibration:
    """Monotone piecewise-linear visual-score calibration.

    ``points`` is a list of ``(visual_score, confidence)`` pairs in
    increasing order. Values outside the fitted range clamp to the end
    points; the default empty calibration is the identity on ``[0, 1]``.
    """

    points: tuple[tuple[float, float], ...] = ()

    @classmethod
    def from_config(cls, config: dict | None) -> "VisualCalibration":
        raw = (config or {}).get("visual_calibration")
        if not raw:
            return cls()
        return cls(tuple(sorted((float(a), float(b)) for a, b in raw)))

    def __call__(self, visual_score: float) -> float:
        return calibrate_visual(visual_score, self)


def calibrate_visual(
    visual_score: float,
    calibration: VisualCalibration | None = None,
) -> float:
    """Map a unified visual score to a calibrated public confidence."""

    score = float(np.clip(visual_score, 0.0, 1.0))
    cal = calibration or VisualCalibration()
    if not cal.points:
        return score
    xs = [p[0] for p in cal.points]
    ys = [p[1] for p in cal.points]
    if score <= xs[0]:
        return float(np.clip(ys[0], 0.0, 1.0))
    if score >= xs[-1]:
        return float(np.clip(ys[-1], 0.0, 1.0))
    for (x0, y0), (x1, y1) in zip(cal.points, cal.points[1:]):
        if x0 <= score <= x1:
            t = (score - x0) / max(x1 - x0, 1e-12)
            return float(np.clip(y0 + t * (y1 - y0), 0.0, 1.0))
    return score


def unified_visual_score(
    template_score: float,
    cnn_score: float,
    geometry_score: float,
    weights: VisualWeights | None = None,
) -> float:
    """Goal 10 formula: ``a*cnn + b*template + c*geometry`` in ``[0, 1]``."""

    w = weights or VisualWeights()
    return float(
        np.clip(
            w.cnn * cnn_score
            + w.template * template_score
            + w.geometry * geometry_score,
            0.0,
            1.0,
        )
    )


@dataclass(frozen=True)
class SegmentScore:
    """Score of one segmentation candidate plus its best character."""

    char_id: int
    visual_score: float
    raw_score: float
    score_type: str  # "template" | "cnn"
    visual_scores: VisualScores | None = None
    template_raw_score: float = 0.0
    cnn_logit: float = 0.0
    cnn_margin: float = 0.0
    cnn_score: float = 0.0
    geometry_score: float = 0.0
    confidence: float = 0.0
    geometry_included: bool = False
    geometry_contribution: float = 0.0

    @property
    def candidate_score(self) -> CandidateScore:
        """View of this score as the plan's unified CandidateScore."""

        return CandidateScore(
            visual_score=self.visual_score,
            raw_score=self.raw_score,
            score_type=self.score_type,
            template_raw_score=self.template_raw_score,
            cnn_logit=self.cnn_logit,
            cnn_margin=self.cnn_margin,
            cnn_score=self.cnn_score,
            geometry_score=self.geometry_score,
        )

    @property
    def classifier_visual_score(self) -> float:
        """Classifier-only visual score (before the geometry term)."""
        if self.geometry_included:
            return self.visual_score - self.geometry_contribution
        return self.visual_score

    @property
    def public_confidence(self) -> float:
        """Calibrated public confidence for this candidate.

        A trusted template is already a high-confidence visual match, so its
        raw template score is the public confidence (the unified weighted
        score may be lower only because the other classifier is uncertain).
        Every other path uses the calibrated unified visual score.
        """
        if self.score_type == "template" and self.template_raw_score > self.confidence:
            return float(np.clip(self.template_raw_score, 0.0, 1.0))
        return self.confidence

    @property
    def template_score(self) -> float:
        """Alias for the raw template confidence in ``[0, 1]``."""
        return self.template_raw_score


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


def _template_entries(
    tb,
    i: int,
) -> list[tuple[int, float]]:
    """Top-K ``(char_id, template_score)`` entries for one candidate."""
    if tb.top_k_ids is not None:
        return [
            (int(cid), float(score))
            for cid, score in zip(tb.top_k_ids[i], tb.top_k_scores[i])
            if int(cid) >= 0
        ]
    out = [(int(tb.ids[i]), float(tb.scores[i]))]
    if tb.second_ids is not None and int(tb.second_ids[i]) >= 0:
        out.append((int(tb.second_ids[i]), float(tb.second_scores[i])))
    return out


def _cnn_entries(
    batch: ClassificationBatch,
    i: int,
) -> list[tuple[int, float]]:
    """Top-K ``(char_id, raw_logit)`` entries for one candidate."""
    if batch.topk_ids is not None and batch.topk_logits is not None:
        return [
            (int(cid), float(logit))
            for cid, logit in zip(batch.topk_ids[i], batch.topk_logits[i])
            if int(cid) >= 0
        ]
    out: list[tuple[int, float]] = [(int(batch.ids[i]), float(batch.top1[i]))]
    if batch.second_ids is not None and int(batch.second_ids[i]) >= 0:
        out.append((int(batch.second_ids[i]), float(batch.top2[i])))
    return out


def _cnn_char_score(logit: float, second_logit: float) -> float:
    """Normalized CNN score for one character in ``[0, 1]``.

    Uses the sigmoid of ``logit - second_logit``: the top-1 character maps
    its top-1/top-2 margin to ``(0.5, 1)``, the second character maps to
    ``0.5``, and every lower-ranked character maps below ``0.5`` without
    being collapsed to zero.
    """

    if not np.isfinite(logit):
        return 0.0
    if not np.isfinite(second_logit):
        return 1.0
    return cnn_visual(logit - second_logit)


def _fuse_scores(
    template_entries: list[tuple[int, float]],
    cnn_entries: list[tuple[int, float]],
    cnn_top2_logit: float,
    full_logits: np.ndarray | None,
    weights: VisualWeights,
) -> list[tuple[float, int, float, float, float, float]]:
    """Fuse template Top-K with CNN Top-K into one ranked character list.

    Returns ``(combined, char_id, template_raw, cnn_logit, cnn_margin,
    cnn_score)`` sorted by combined score descending. Characters visible to
    only one classifier use zero for the other classifier's term.
    """

    template_by_id = {cid: score for cid, score in template_entries}
    cnn_by_id = dict(cnn_entries)
    ids = sorted(set(template_by_id) | set(cnn_by_id))
    scored: list[tuple[float, int, float, float, float, float]] = []
    for cid in ids:
        template_score = template_by_id.get(cid, 0.0)
        raw_logit = cnn_by_id.get(cid)
        if raw_logit is None and full_logits is not None and 0 <= cid < full_logits.shape[0]:
            raw_logit = float(full_logits[cid])
        if raw_logit is None:
            cnn_score = 0.0
            logit = 0.0
            margin = 0.0
        else:
            logit = float(raw_logit)
            margin = (
                logit - float(cnn_top2_logit)
                if np.isfinite(cnn_top2_logit)
                else 1.0
            )
            cnn_score = _cnn_char_score(logit, float(cnn_top2_logit))
        combined = float(
            np.clip(
                weights.cnn * cnn_score + weights.template * template_score,
                0.0,
                1.0,
            )
        )
        scored.append((combined, cid, template_score, logit, margin, cnn_score))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return scored


def _fuse_trusted_template(
    template_entries: list[tuple[int, float]],
    cnn_entries: list[tuple[int, float]],
    cnn_top2_logit: float,
    full_logits: np.ndarray | None,
    weights: VisualWeights,
    template_cid: int,
    template_score: float,
) -> list[tuple[float, int, float, float, float, float]]:
    """Ranked evidence for a template-trusted candidate.

    The template is authoritative for the character identity (its margin
    gate passed), but the CNN evidence is still stored and can only boost
    the unified visual score -- it can never lower a confident template
    match below the template's own confidence.
    """

    cnn_by_id = dict(cnn_entries)
    raw_logit = cnn_by_id.get(template_cid)
    if raw_logit is None and full_logits is not None and 0 <= template_cid < full_logits.shape[0]:
        raw_logit = float(full_logits[template_cid])
    if raw_logit is None:
        logit = 0.0
        margin = 0.0
        cnn_score = 0.0
    else:
        logit = float(raw_logit)
        margin = (
            logit - float(cnn_top2_logit)
            if np.isfinite(cnn_top2_logit)
            else 1.0
        )
        cnn_score = _cnn_char_score(logit, float(cnn_top2_logit))
    combined = max(
        template_score,
        float(
            np.clip(
                weights.cnn * cnn_score + weights.template * template_score,
                0.0,
                1.0,
            )
        ),
    )
    best = (combined, template_cid, template_score, logit, margin, cnn_score)

    others: dict[int, tuple[float, int, float, float, float, float]] = {}
    for cid, score in template_entries:
        if cid == template_cid:
            continue
        raw_logit = cnn_by_id.get(cid)
        if raw_logit is None and full_logits is not None and 0 <= cid < full_logits.shape[0]:
            raw_logit = float(full_logits[cid])
        if raw_logit is None:
            logit, margin, cnn_score = 0.0, 0.0, 0.0
        else:
            logit = float(raw_logit)
            margin = (
                logit - float(cnn_top2_logit)
                if np.isfinite(cnn_top2_logit)
                else 1.0
            )
            cnn_score = _cnn_char_score(logit, float(cnn_top2_logit))
        others[cid] = (
            float(
                np.clip(
                    weights.cnn * cnn_score + weights.template * score,
                    0.0,
                    1.0,
                )
            ),
            cid,
            score,
            logit,
            margin,
            cnn_score,
        )
    for cid, logit in cnn_entries:
        if cid == template_cid or cid in others:
            continue
        margin = (
            logit - float(cnn_top2_logit)
            if np.isfinite(cnn_top2_logit)
            else 1.0
        )
        cnn_score = _cnn_char_score(logit, float(cnn_top2_logit))
        others[cid] = (
            float(np.clip(weights.cnn * cnn_score, 0.0, 1.0)),
            cid,
            0.0,
            logit,
            margin,
            cnn_score,
        )
    ranked = sorted(others.values(), key=lambda row: (-row[0], row[1]))
    return [best, *ranked]


def _unified_visual_scores(
    fused: list[tuple[float, int, float, float, float, float]],
    raw_score: float,
    score_type: str,
    calibration: VisualCalibration,
) -> VisualScores | None:
    """Project fused evidence into the Goal 1 Top-K ``VisualScores``.

    The full Top-5 is retained (not just Top-2): at low resolution the
    correct character is frequently the third/fourth-ranked alternative of
    a merged candidate (e.g. 潜 at 17 px, 摩 at 18 px), and the Goal 13
    decoder or a Goal 11/12 lexicon can only correct the path when that
    alternative is still visible in ``VisualScores``.
    """
    if not fused:
        return None
    top = fused[:5]
    char_ids = tuple(int(row[1]) for row in top)
    logits = tuple(float(row[0]) for row in top)
    best = top[0]
    second = top[1] if len(top) > 1 else None
    margin = best[0] - (second[0] if second is not None else 0.0)
    return VisualScores(
        char_ids=char_ids,
        logits=logits,
        top_k=char_ids,
        margin=float(margin),
        raw_score=float(raw_score),
        score_type=score_type,
        template_raw_score=best[2],
        cnn_logit=best[3],
        cnn_margin=best[4],
        cnn_score=best[5],
        visual_score=best[0],
        confidence=calibration(best[0]),
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
    """Batch-scorer used by the segmentation DP for one model.

    ``cnn_backend`` is the CNN executor for TinyCNN/hybrid models. The CPU
    reference passes :class:`CPUBackend`; WGPU runs use :class:`WGPUBackend`
    (or :class:`AutoBackend`) with the exact same OCR algorithm. This class
    never imports/calls ``cnn.forward`` directly -- all CNN evidence enters
    through the backend boundary.
    """

    def __init__(self, model: OCRModel, cnn_backend: Backend | None = None):
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
        if self.weights is not None:
            self.cnn_backend: Backend | None = cnn_backend or CPUBackend(
                model.weights, model.input_size
            )
        else:
            self.cnn_backend = None
        self.template_threshold = float(model.config.get("template_threshold", 0.90))
        self.template_margin_threshold = float(
            model.config.get("template_margin_threshold", 0.04)
        )
        self.cnn_threshold = float(model.config.get("cnn_threshold", 0.0))
        self.cnn_input_mode = model.input_mode
        self.geometry = model.geometry
        self.visual_weights = VisualWeights.from_config(model.config)
        self.calibration = VisualCalibration.from_config(model.config)
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
                out: list[SegmentScore] = []
                for i in range(n):
                    fused = _fuse_scores(
                        _template_entries(tb, i),
                        [],
                        -np.inf,
                        None,
                        VisualWeights(cnn=0.0, template=1.0, geometry=self.visual_weights.geometry),
                    )
                    if not fused:
                        out.append(
                            SegmentScore(
                                char_id=-1,
                                visual_score=0.0,
                                raw_score=0.0,
                                score_type="template",
                            )
                        )
                        continue
                    best = fused[0]
                    raw = float(tb.dists[i])
                    vs = _unified_visual_scores(
                        fused, raw, "template", self.calibration
                    )
                    out.append(
                        SegmentScore(
                            char_id=best[1],
                            visual_score=best[0],
                            raw_score=raw,
                            score_type="template",
                            visual_scores=vs,
                            template_raw_score=best[2],
                            cnn_logit=best[3],
                            cnn_margin=best[4],
                            cnn_score=best[5],
                            confidence=self.calibration(best[0]),
                        )
                    )
                return out
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
            cnn = self._cnn_batch(glyphs, allowed_ids, soft_batch)
            out: list[SegmentScore] = []
            for i in range(n):
                template_entries = _template_entries(tb, i)
                cnn_entries = _cnn_entries(cnn, i)
                full_logits = cnn.logits[i] if cnn.logits is not None else None
                if needs_cnn[i]:
                    # The template margin gate failed, so the CNN remains
                    # the decision source; but the template Top-K evidence
                    # still participates in the unified visual score. For
                    # every character the final score is the max of its
                    # CNN-only evidence and its weighted template+CNN
                    # evidence: a confident CNN pick is never diluted by an
                    # absent/weak template, while a correct low-margin
                    # low-res match (潜 at 17 px, 鲃 at 15 px) gets the
                    # corroborating template boost it needs to beat
                    # look-alike fragments (Goal 14).
                    cnn_only = _fuse_scores(
                        [],
                        cnn_entries,
                        float(cnn.top2[i]),
                        full_logits,
                        VisualWeights(
                            cnn=1.0,
                            template=0.0,
                            geometry=self.visual_weights.geometry,
                        ),
                    )
                    if template_entries:
                        fused_full = _fuse_scores(
                            template_entries,
                            cnn_entries,
                            float(cnn.top2[i]),
                            full_logits,
                            self.visual_weights,
                        )
                        cnn_only_by_id = {row[1]: row for row in cnn_only}
                        fused = []
                        for row in fused_full:
                            base = cnn_only_by_id.get(row[1])
                            if base is None:
                                fused.append(row)
                            else:
                                fused.append(
                                    (max(base[0], row[0]), *row[1:])
                                )
                        fused.sort(key=lambda r: (-r[0], r[1]))
                    else:
                        fused = cnn_only
                    score_type = "cnn"
                    raw = float(cnn.margins[i])
                else:
                    template_cid = int(tb.ids[i])
                    template_score = float(tb.scores[i])
                    fused = _fuse_trusted_template(
                        template_entries,
                        cnn_entries,
                        float(cnn.top2[i]),
                        full_logits,
                        self.visual_weights,
                        template_cid,
                        template_score,
                    )
                    score_type = "template"
                    raw = float(tb.dists[i])
                if not fused:
                    out.append(
                        SegmentScore(
                            char_id=-1,
                            visual_score=0.0,
                            raw_score=0.0,
                            score_type=score_type,
                        )
                    )
                    continue
                best = fused[0]
                vs = _unified_visual_scores(
                    fused, raw, score_type, self.calibration
                )
                out.append(
                    SegmentScore(
                        char_id=best[1],
                        visual_score=best[0],
                        raw_score=raw,
                        score_type=score_type,
                        visual_scores=vs,
                        template_raw_score=best[2],
                        cnn_logit=best[3],
                        cnn_margin=best[4],
                        cnn_score=best[5],
                        confidence=self.calibration(best[0]),
                    )
                )
            return out

        if self.weights is not None:
            cnn = self._cnn_batch(glyphs, allowed_ids, soft_batch)
            out = []
            for i in range(n):
                fused = _fuse_scores(
                    [],
                    _cnn_entries(cnn, i),
                    float(cnn.top2[i]),
                    cnn.logits[i] if cnn.logits is not None else None,
                    VisualWeights(cnn=1.0, template=0.0, geometry=self.visual_weights.geometry),
                )
                if not fused:
                    out.append(
                        SegmentScore(
                            char_id=-1,
                            visual_score=0.0,
                            raw_score=0.0,
                            score_type="cnn",
                        )
                    )
                    continue
                best = fused[0]
                raw = float(cnn.margins[i])
                vs = _unified_visual_scores(
                    fused, raw, "cnn", self.calibration
                )
                out.append(
                    SegmentScore(
                        char_id=best[1],
                        visual_score=best[0],
                        raw_score=raw,
                        score_type="cnn",
                        visual_scores=vs,
                        template_raw_score=best[2],
                        cnn_logit=best[3],
                        cnn_margin=best[4],
                        cnn_score=best[5],
                        confidence=self.calibration(best[0]),
                    )
                )
            return out
        raise ValueError("model has neither templates nor CNN weights")

    def score_fused(
        self,
        segments: list[Segment],
        allowed_ids: set[int] | None = None,
        soft: NDArray[np.uint8] | None = None,
        geometries: list[tuple[float, float] | None] | None = None,
        visual_weights: VisualWeights | None = None,
    ) -> list[SegmentScore]:
        """Goal 10 fusion variant used for weight tuning.

        Unlike :meth:`score`, this always fuses template and CNN evidence
        with the supplied (or configured) weights -- no trust gate overrides
        the character choice. The normal decoder keeps the compatibility
        trust gate, while tuning and offline evaluation use this method to
        find the real-test-set weights.
        """

        if not segments:
            return []
        weights = visual_weights or self.visual_weights
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

        if self.template is not None and self.weights is not None:
            tb = self.template.match_batch(glyphs, allowed_ids)
            cnn = self._cnn_batch(glyphs, allowed_ids, soft_batch)
            out: list[SegmentScore] = []
            for i in range(n):
                template_entries = _template_entries(tb, i)
                cnn_entries = _cnn_entries(cnn, i)
                full_logits = cnn.logits[i] if cnn.logits is not None else None
                fused = _fuse_scores(
                    template_entries,
                    cnn_entries,
                    float(cnn.top2[i]),
                    full_logits,
                    weights,
                )
                if not fused:
                    out.append(
                        SegmentScore(
                            char_id=-1,
                            visual_score=0.0,
                            raw_score=0.0,
                            score_type="cnn",
                        )
                    )
                    continue
                best = fused[0]
                template_part = weights.template * best[2]
                cnn_part = weights.cnn * best[5]
                score_type = (
                    "template" if template_part >= cnn_part else "cnn"
                )
                raw = (
                    float(tb.dists[i])
                    if score_type == "template"
                    else float(cnn.margins[i])
                )
                vs = _unified_visual_scores(
                    fused, raw, score_type, self.calibration
                )
                out.append(
                    SegmentScore(
                        char_id=best[1],
                        visual_score=best[0],
                        raw_score=raw,
                        score_type=score_type,
                        visual_scores=vs,
                        template_raw_score=best[2],
                        cnn_logit=best[3],
                        cnn_margin=best[4],
                        cnn_score=best[5],
                        confidence=self.calibration(best[0]),
                    )
                )
            return out
        return self.score(
            segments,
            allowed_ids,
            soft=soft,
            geometries=geometries,
        )

    def _cnn_batch(
        self,
        glyphs: np.ndarray,
        allowed_ids: set[int] | None,
        soft_batch: np.ndarray | None = None,
    ) -> ClassificationBatch:
        if allowed_ids is not None and not allowed_ids:
            n = soft_batch.shape[0] if soft_batch is not None else glyphs.shape[0]
            return ClassificationBatch(
                ids=np.full(n, -1, dtype=np.int32),
                top1=np.full(n, -np.inf, dtype=np.float32),
                top2=np.full(n, -np.inf, dtype=np.float32),
                margins=np.full(n, 0.0, dtype=np.float32),
                second_ids=np.full(n, -1, dtype=np.int32),
                topk_ids=np.empty((n, 0), dtype=np.int32),
                topk_logits=np.empty((n, 0), dtype=np.float32),
                logits=None,
            )
        if self.cnn_backend is None:
            raise ValueError("CNN backend is required for a TinyCNN/hybrid model")
        cnn_input = soft_batch if soft_batch is not None else glyphs
        logits = self.cnn_backend.forward_logits(cnn_input)
        if allowed_ids is not None:
            masked = np.full_like(logits, -np.inf)
            idx = np.fromiter(sorted(allowed_ids), dtype=np.int64)
            masked[:, idx] = logits[:, idx]
            logits = masked
        ids, top1, top2v, margins = top2(logits)
        topk_ids, topk_logits = topk(logits, 5)
        return ClassificationBatch(
            ids=ids,
            top1=top1,
            top2=top2v,
            margins=margins,
            second_ids=topk_second_ids(logits, ids),
            topk_ids=topk_ids,
            topk_logits=topk_logits,
            logits=logits,
        )

    def finalize_score(
        self,
        score: SegmentScore,
        geometry_score: float,
    ) -> SegmentScore:
        """Add the Goal 10 geometry term and calibrate the confidence.

        The classifier half of the scorer already produced
        ``a * cnn_score + b * template_score``; this method adds
        ``c * geometry_score`` and maps the result through the model's
        calibration so the decoder and public API consume the same unified
        value.
        """

        visual = float(
            np.clip(
                score.visual_score
                + self.visual_weights.geometry * float(geometry_score),
                0.0,
                1.0,
            )
        )
        confidence = self.calibration(visual)
        geometry_contribution = float(
            self.visual_weights.geometry * float(geometry_score)
        )
        vs = (
            replace(
                score.visual_scores,
                geometry_score=float(geometry_score),
                visual_score=visual,
                confidence=confidence,
                geometry_included=True,
            )
            if score.visual_scores is not None
            else None
        )
        return replace(
            score,
            geometry_score=float(geometry_score),
            visual_score=visual,
            confidence=confidence,
            geometry_included=True,
            geometry_contribution=geometry_contribution,
            visual_scores=vs,
        )

    def classifier_score_for(
        self,
        candidate: VisualCandidate,
        char_id: int,
    ) -> float:
        """Classifier-only fused score of one chosen ``(candidate, char_id)``.

        This is the Top-K entry from ``candidate.scores`` (before the
        geometry term), so choosing the decoder's Top-2/Top-3 character
        reports that character's own visual score -- never the Top-1 score
        with a different label attached.
        """

        scores = candidate.scores
        if scores is not None and scores.char_ids:
            logits = scores.logits or ()
            for i, cid in enumerate(scores.char_ids):
                if int(cid) == int(char_id):
                    if i < len(logits):
                        return float(logits[i])
                    # Manual VisualScores may only carry the finalized Top-1
                    # scalar; remove the geometry contribution before the
                    # caller adds the chosen character's geometry back.
                    if i == 0:
                        visual = float(scores.visual_score)
                        if scores.geometry_included:
                            visual -= (
                                self.visual_weights.geometry
                                * float(scores.geometry_score)
                            )
                        return visual
                    return 0.0
        score = candidate.score
        if score is not None and getattr(score, "char_id", -1) == int(char_id):
            return float(score.classifier_visual_score)
        return 0.0

    def finalize_char_score(
        self,
        candidate: VisualCandidate,
        char_id: int,
        geometry_score: float,
    ) -> tuple[float, float]:
        """Return ``(visual_score, confidence)`` for a chosen character.

        The decoder owns *which* character was chosen; this method only adds
        the geometry term and calibration to that character's own Top-K
        visual score, without running any classifier a second time.
        """

        classifier_score = self.classifier_score_for(candidate, char_id)
        visual = float(
            np.clip(
                classifier_score
                + self.visual_weights.geometry * float(geometry_score),
                0.0,
                1.0,
            )
        )
        return visual, self.calibration(visual)
