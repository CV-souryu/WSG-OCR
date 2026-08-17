"""Probabilistic joint decoder (v2): local ranker + Viterbi/k-best.

This is the ``probabilistic/v2`` decoder. The legacy Goal 13 decoder
(``decode_dp`` / ``decode_beam``) stays untouched as ``legacy/v1``; the
model config selects between them:

* no ``decoder`` block, or ``decoder.version < 2``, or any validation
  failure -> the engine uses v1 and its output is byte-identical to the
  pre-v2 pipeline;
* ``decoder.version == 2`` with a fully valid schema -> v2.

The v2 path score is additive in log space:

.. math::

    S(\\pi) = \\log\\sum_z \\exp\\Big[
        \\log P(z)
        + \\sum_i \\big(\\log P(I_i\\mid c_i, z)
                             + \\log P(G_i\\mid c_i)
                             + \\log P(B_i)\\big)
        + \\log P(c_{1:n}\\mid domain)
    \\Big]

with

* ``log P(I_i | c_i, z)`` -- the local ranker's logit (visual block only)
  over the candidate's Top-K options, temperature-scaled and normalized by
  the per-candidate log-softmax, conditioned on the line render state
  ``z`` (template state features);
* ``log P(G_i | c_i)`` -- a learned linear log-evidence over the
  geometry-residual block (clipped, so it can never dominate a clear
  visual match);
* ``log P(B_i)`` -- a learned linear log-evidence over the boundary block
  (merge/split type, seam approximation, width ratio, ...);
* ``log P(c_{1:n} | domain)`` -- the calibrated :class:`DomainPrior`
  (open text / allowed chars / lexicon / unigram).

The best path is found by a deterministic k-best Viterbi over the atom
sequence for every retained render state, the state is marginalized with
``logsumexp``, and the score is length-normalized by
``(n + beta) ** alpha`` with ``alpha``/``beta`` chosen on the validation
split (see ``docs/probabilistic_decoder.md`` for why this does not
systematically bias path length). Ties are broken by (1) fewer anomalous
split/merge candidates, (2) smaller candidate spans, (3) lexicographic
text -- never by Python object identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .domain_prior import DomainPrior, OpenTextPrior
from .model import OCRModel
from .prob_features import (
    BOUNDARY_FEATURE_IDX,
    FEATURE_SCHEMA_VERSION,
    GEOMETRY_FEATURE_IDX,
    N_FEATURES,
    VISUAL_FEATURE_IDX,
    LineContext,
    LocalFeatureExtractor,
)
from .prob_math import calibrate_monotone, logsumexp
from .render_state import RenderStateModel
from .types import DecodePath, Profile, VisualCandidate, VisualLattice

UNKNOWN_CHAR = "?"
DECODER_VERSION = 2
SCORE_TYPE = "log_probability"
N_VISUAL_FEATURES = len(VISUAL_FEATURE_IDX)  # 26
N_GEOMETRY_FEATURES = len(GEOMETRY_FEATURE_IDX)  # 9
N_BOUNDARY_FEATURES = len(BOUNDARY_FEATURE_IDX)  # 14


@dataclass(frozen=True)
class ProbDecoderConfig:
    """Validated v2 decoder configuration (mirrors ``config.json``)."""

    version: int = DECODER_VERSION
    score_type: str = SCORE_TYPE
    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    local_ranker_coef: tuple[float, ...] = ()  # N_VISUAL_FEATURES
    local_ranker_intercept: float = 0.0
    feature_mean: tuple[float, ...] = ()  # N_FEATURES
    feature_std: tuple[float, ...] = ()  # N_FEATURES
    feature_clip: float = 4.0
    geometry_coef: tuple[float, ...] = ()  # N_GEOMETRY_FEATURES
    boundary_coef: tuple[float, ...] = ()  # N_BOUNDARY_FEATURES
    temperature: float = 1.0
    length_alpha: float = 0.0
    length_beta: float = 0.0
    render_states: dict = None  # type: ignore[assignment]
    confidence_calibration: tuple[tuple[float, float], ...] = ()
    reject_enabled: bool = False
    reject_confidence: float = 0.0
    reject_margin: float = 0.0
    k_best: int = 8
    num_alternatives: int = 8
    unknown_log_prob: float = -8.0
    # Open-set local normalization: the per-candidate log-softmax also
    # competes against a "not a known character" background option with
    # this logit. A candidate whose Top-K options are all weak then yields
    # genuinely low log-probabilities (it cannot saturate the softmax),
    # which is what stops a garbage merge/split candidate from winning a
    # path. Learned in the structured training stage; validated finite.
    background_logit: float = 0.0
    max_state_approx: bool = False
    domain_prior_cfg: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.render_states is None:
            object.__setattr__(self, "render_states", {})
        if self.domain_prior_cfg is None:
            object.__setattr__(self, "domain_prior_cfg", {})

    @property
    def render_state_model(self) -> RenderStateModel:
        return RenderStateModel.from_config(self.render_states)

    def validate(self) -> list[str]:
        """Return every schema violation (empty list == valid)."""

        errors: list[str] = []
        if self.version != DECODER_VERSION:
            errors.append(f"decoder.version must be {DECODER_VERSION}")
        if self.score_type != SCORE_TYPE:
            errors.append(f"decoder.score_type must be {SCORE_TYPE!r}")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            errors.append(
                f"decoder.feature_schema_version must be "
                f"{FEATURE_SCHEMA_VERSION}"
            )
        for name, arr, expect in (
            ("local_ranker.coef", self.local_ranker_coef, N_VISUAL_FEATURES),
            ("feature_normalization.mean", self.feature_mean, N_FEATURES),
            ("feature_normalization.std", self.feature_std, N_FEATURES),
            ("geometry_weights.coef", self.geometry_coef, N_GEOMETRY_FEATURES),
            ("boundary_weights.coef", self.boundary_coef, N_BOUNDARY_FEATURES),
        ):
            a = np.asarray(arr, dtype=np.float64)
            if a.size != expect:
                errors.append(f"{name} has {a.size} values, expected {expect}")
            elif not np.all(np.isfinite(a)):
                errors.append(f"{name} contains NaN/Inf")
        if not np.isfinite(self.local_ranker_intercept):
            errors.append("local_ranker.intercept is NaN/Inf")
        if not (np.isfinite(self.temperature) and self.temperature > 0.0):
            errors.append("decoder.temperature must be finite and > 0")
        if not (np.isfinite(self.length_alpha) and 0.0 <= self.length_alpha <= 2.0):
            errors.append("decoder.length_alpha must be finite and in [0, 2]")
        if not (np.isfinite(self.length_beta) and self.length_beta >= 0.0):
            errors.append("decoder.length_beta must be finite and >= 0")
        if not (np.isfinite(self.unknown_log_prob) and self.unknown_log_prob <= 0.0):
            errors.append("decoder.unknown_log_prob must be finite and <= 0")
        if not np.isfinite(self.background_logit):
            errors.append("decoder.background_logit must be finite")
        if self.k_best < 1 or self.num_alternatives < 1:
            errors.append("decoder.k_best/num_alternatives must be >= 1")
        if self.feature_std:
            std = np.asarray(self.feature_std, dtype=np.float64)
            if np.any(std < 0.0):
                errors.append("feature_normalization.std must be >= 0")
        if not (np.isfinite(self.feature_clip) and self.feature_clip > 0.0):
            errors.append("feature_normalization.clip must be finite and > 0")
        pts = np.asarray(
            [(a, b) for a, b in self.confidence_calibration], dtype=np.float64
        )
        if pts.size:
            if pts.shape != (pts.shape[0], 2):
                errors.append("confidence_calibration must be [[x, y], ...]")
            else:
                if np.any(pts[:, 0] < 0.0) or np.any(pts[:, 0] > 1.0):
                    errors.append("confidence_calibration x outside [0, 1]")
                if np.any(np.diff(pts[:, 0]) <= 0.0):
                    errors.append("confidence_calibration x not increasing")
                if np.any(np.diff(pts[:, 1]) < 0.0):
                    errors.append("confidence_calibration y not monotone")
                if np.any(pts[:, 1] < 0.0) or np.any(pts[:, 1] > 1.0):
                    errors.append("confidence_calibration y outside [0, 1]")
        if not (0.0 <= self.reject_confidence <= 1.0):
            errors.append("reject.confidence_threshold must be in [0, 1]")
        if not (0.0 <= self.reject_margin <= 10.0):
            errors.append("reject.margin_threshold must be in [0, 10]")
        try:
            self.render_state_model
        except ValueError as exc:
            errors.append(f"render_states invalid: {exc}")
        return errors

    @classmethod
    def from_dict(cls, cfg: dict) -> "ProbDecoderConfig":
        """Parse + validate a ``config.json["decoder"]`` block.

        Raises :class:`ValueError` on any invalid field (NaN/Inf, wrong
        dimensions, unsupported values). The engine catches this and falls
        back to v1.
        """

        lr = cfg.get("local_ranker", {}) or {}
        fn = cfg.get("feature_normalization", {}) or {}
        gw = cfg.get("geometry_weights", {}) or {}
        bw = cfg.get("boundary_weights", {}) or {}
        rs = cfg.get("render_states", {}) or {}
        cal = cfg.get("confidence_calibration", []) or []
        rej = cfg.get("reject", {}) or {}
        out = cls(
            version=int(cfg.get("version", DECODER_VERSION)),
            score_type=str(cfg.get("score_type", SCORE_TYPE)),
            feature_schema_version=int(
                cfg.get("feature_schema_version", FEATURE_SCHEMA_VERSION)
            ),
            local_ranker_coef=tuple(float(x) for x in lr.get("coef", ())),
            local_ranker_intercept=float(lr.get("intercept", 0.0)),
            feature_mean=tuple(float(x) for x in fn.get("mean", ())),
            feature_std=tuple(float(x) for x in fn.get("std", ())),
            feature_clip=float(fn.get("clip", 4.0)),
            geometry_coef=tuple(float(x) for x in gw.get("coef", ())),
            boundary_coef=tuple(float(x) for x in bw.get("coef", ())),
            temperature=float(cfg.get("temperature", 1.0)),
            length_alpha=float(cfg.get("length_alpha", 0.0)),
            length_beta=float(cfg.get("length_beta", 0.0)),
            render_states=dict(rs),
            confidence_calibration=tuple(
                (float(a), float(b)) for a, b in cal
            ),
            reject_enabled=bool(rej.get("enabled", False)),
            reject_confidence=float(rej.get("confidence_threshold", 0.0)),
            reject_margin=float(rej.get("margin_threshold", 0.0)),
            k_best=int(cfg.get("k_best", 8)),
            num_alternatives=int(cfg.get("num_alternatives", 8)),
            unknown_log_prob=float(cfg.get("unknown_log_prob", -8.0)),
            background_logit=float(cfg.get("background_logit", 0.0)),
            max_state_approx=bool(cfg.get("max_state_approx", False)),
            domain_prior_cfg=dict(cfg.get("domain_prior", {}) or {}),
        )
        errors = out.validate()
        if errors:
            raise ValueError("invalid decoder config: " + "; ".join(errors))
        return out

    @classmethod
    def load(cls, config: dict) -> "ProbDecoderConfig | None":
        """Safe loader: ``None`` (v1) when absent/legacy; raises on invalid.

        ``version < 2`` or a missing block returns ``None`` (v1, no
        change); a version-2 block that fails validation raises so the
        caller can record the reason and fall back to v1.
        """

        block = (config or {}).get("decoder")
        if not isinstance(block, dict):
            return None
        try:
            version = int(block.get("version", 1))
        except (TypeError, ValueError):
            return None
        if version < 2:
            return None
        return cls.from_dict(block)


class LocalRanker:
    """Logistic local ranker over the normalized visual feature block."""

    def __init__(self, cfg: ProbDecoderConfig):
        if not cfg.local_ranker_coef:
            raise ValueError("local_ranker.coef is empty")
        self.coef = np.asarray(cfg.local_ranker_coef, dtype=np.float64)
        self.intercept = float(cfg.local_ranker_intercept)
        self.temperature = float(cfg.temperature)

    def logit(self, normalized_visual: np.ndarray) -> float:
        """Raw logit of one normalized visual feature vector."""

        f = np.asarray(normalized_visual, dtype=np.float64)
        return float(self.coef @ f + self.intercept)


@dataclass(frozen=True)
class _CharOption:
    """One (candidate, char, z) alternative with its score pieces."""

    char_id: int
    logp_visual: float
    logp_geometry: float


@dataclass(frozen=True)
class _PathEntry:
    """One k-best partial/full path."""

    score: float  # raw additive log score (before length normalization)
    n_anomalies: int
    span: int
    text: str
    candidates: tuple[VisualCandidate, ...] = ()
    char_ids: tuple[int, ...] = ()
    prev: "_PathEntry | None" = None

    @property
    def ident(self) -> tuple:
        """Hashable path identity (candidate ids, not objects)."""

        return (tuple(id(c) for c in self.candidates), self.char_ids)

    def key(self) -> tuple:
        return (-self.score, self.n_anomalies, self.span, self.text)


class PathDiagnostics:
    """Diagnostics of the best v2 path (serialized into DecodePath)."""

    def __init__(
        self,
        total_log_prob: float,
        normalized_score: float,
        visual_sum: float,
        geometry_sum: float,
        boundary_sum: float,
        domain_sum: float,
        line_render_state: str,
        states: tuple[str, ...],
        runner_up_margin: float,
        k_best_searched: int,
        reject_reason: str,
        per_char: tuple[dict, ...],
    ):
        self.total_log_prob = float(total_log_prob)
        self.normalized_score = float(normalized_score)
        self.visual_sum = float(visual_sum)
        self.geometry_sum = float(geometry_sum)
        self.boundary_sum = float(boundary_sum)
        self.domain_sum = float(domain_sum)
        self.line_render_state = line_render_state
        self.states = tuple(states)
        self.runner_up_margin = float(runner_up_margin)
        self.k_best_searched = int(k_best_searched)
        self.reject_reason = reject_reason
        self.per_char = tuple(per_char)

    def to_dict(self) -> dict:
        return {
            "total_log_prob": self.total_log_prob,
            "normalized_score": self.normalized_score,
            "decomposition": {
                "visual": self.visual_sum,
                "geometry": self.geometry_sum,
                "boundary": self.boundary_sum,
                "domain": self.domain_sum,
            },
            "line_render_state": self.line_render_state,
            "states": list(self.states),
            "runner_up_margin": self.runner_up_margin,
            "k_best_searched": self.k_best_searched,
            "reject_reason": self.reject_reason,
            "per_char": [
                {
                    "char": str(d.get("char", "")),
                    "logp_visual": float(d.get("logp_visual", 0.0)),
                    "logp_geometry": float(d.get("logp_geometry", 0.0)),
                    "logp_boundary": float(d.get("logp_boundary", 0.0)),
                    "logp_domain": float(d.get("logp_domain", 0.0)),
                }
                for d in self.per_char
            ],
        }


class ProbabilisticDecoder:
    """v2 decoder: local ranker + per-state k-best Viterbi + diagnostics."""

    def __init__(
        self,
        cfg: ProbDecoderConfig,
        model: OCRModel,
        profile: Profile | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.profile = profile
        self.ranker = LocalRanker(cfg)
        self.extractor = LocalFeatureExtractor(
            model,
            profile=profile,
            template_v2=self._build_template_v2(model),
        )
        self.render_state = cfg.render_state_model
        self._mean = np.asarray(cfg.feature_mean, dtype=np.float64)
        self._std = np.asarray(cfg.feature_std, dtype=np.float64)
        self._gcoef = np.asarray(cfg.geometry_coef, dtype=np.float64)
        self._bcoef = np.asarray(cfg.boundary_coef, dtype=np.float64)

    @staticmethod
    def _build_template_v2(model: OCRModel):
        if model.templates_v2 is None:
            return None
        from .classifier import TemplateV2Classifier

        return TemplateV2Classifier(
            data=model.templates_v2,
            charset=model.charset,
            input_size=model.input_size,
            normalize_spec=model.normalize_spec,
        )

    @classmethod
    def try_build(
        cls,
        model: OCRModel,
        profile: Profile | None = None,
    ) -> tuple["ProbabilisticDecoder | None", str | None]:
        """Build the v2 decoder or explain the v1 fallback."""

        try:
            cfg = ProbDecoderConfig.load(model.config)
        except ValueError as exc:
            return None, f"decoder config invalid, falling back to v1: {exc}"
        if cfg is None:
            return None, None
        return cls(cfg, model, profile), None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        return self.extractor.normalize(
            features, self._mean, self._std, self.cfg.feature_clip
        )

    def _char_logits(
        self,
        cand: VisualCandidate,
        char_id: int,
        ctx: LineContext,
        z: str,
    ) -> tuple[float, float]:
        """``(temperature-scaled visual logit, geometry log-evidence)``."""

        f = self.extractor.char_features(cand, char_id, ctx, z=z)
        fn = self._normalize(f)
        visual = fn[list(VISUAL_FEATURE_IDX)]
        geo = fn[list(GEOMETRY_FEATURE_IDX)]
        logit = self.ranker.logit(visual)
        logp_geometry = float(np.clip(self._gcoef @ geo, -6.0, 0.0))
        return logit / self.cfg.temperature, logp_geometry

    def _options(
        self,
        cand: VisualCandidate,
        ctx: LineContext,
        z: str,
        allowed_ids: set[int] | None,
    ) -> list[_CharOption]:
        """All character options of one candidate under state ``z``.

        The visual log-probability is the temperature-scaled local logit
        normalized over the candidate's own option set (log-softmax), so a
        candidate with a clear visual winner contributes a high log
        probability while a tied candidate stays near-uniform. Characters
        outside ``allowed_ids`` are excluded; when nothing survives the
        ``?`` option carries ``unknown_log_prob``.
        """

        scores = cand.scores
        if scores is None or not scores.char_ids:
            return [_CharOption(-1, self.cfg.unknown_log_prob, 0.0)]
        logits: list[float] = []
        opts: list[_CharOption] = []
        for cid in scores.char_ids:
            cid = int(cid)
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            if not (0 <= cid < len(self.model.charset)):
                continue
            logit, logp_geo = self._char_logits(cand, cid, ctx, z)
            logits.append(logit)
            opts.append(_CharOption(cid, 0.0, logp_geo))
        if not opts:
            return [_CharOption(-1, self.cfg.unknown_log_prob, 0.0)]
        # Open-set normalization: the character options compete against a
        # "not a known character" background option (background_logit), so
        # a candidate whose every Top-K option is weak cannot saturate its
        # local softmax. The background itself is never a path option.
        lse = logsumexp(
            np.asarray(logits + [self.cfg.background_logit], dtype=np.float64)
        )
        return [
            _CharOption(opt.char_id, float(logit - lse), opt.logp_geometry)
            for opt, logit in zip(opts, logits)
        ]

    def _boundary_logp(self, cand: VisualCandidate, ctx: LineContext) -> float:
        cache = self.extractor._candidate_cache(cand, ctx, None)
        idx = list(BOUNDARY_FEATURE_IDX)
        f = self.extractor.normalize(
            cache.boundary, self._mean[idx], self._std[idx], self.cfg.feature_clip
        )
        return float(np.clip(self._bcoef @ f, -6.0, 0.0))

    def _is_anomalous(self, cand: VisualCandidate, ctx: LineContext) -> int:
        cache = self.extractor._candidate_cache(cand, ctx, None)
        b = cache.boundary
        return int(b[1] > 0.5 or b[2] > 0.5)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _kbest_for_state(
        self,
        lattice: VisualLattice,
        ctx: LineContext,
        z: str,
        allowed_ids: set[int] | None,
        domain: DomainPrior,
        n_atoms: int,
        k: int,
    ) -> list[_PathEntry]:
        """Deterministic k-best Viterbi over the atom sequence for one z.

        The DP keeps the top-``k`` partial paths per atom position ordered
        by the *length-normalized look-ahead key* (the same
        ``(n + beta) ** alpha`` normalization the final ranking uses), so
        a long path of individually strong characters is not pruned by a
        short path whose one candidate saturates the local softmax.
        """

        by_end: dict[int, list[tuple[VisualCandidate, list[_CharOption], float]]] = {}
        for cand in lattice.candidates:
            start, end = cand.atom_span
            if end <= start:
                continue
            bnd = self._boundary_logp(cand, ctx)
            opts = self._options(cand, ctx, z, allowed_ids)
            by_end.setdefault(end, []).append((cand, opts, bnd))

        def path_score(opt: _CharOption, bnd: float, domain: DomainPrior) -> float:
            if opt.char_id >= 0:
                return (
                    opt.logp_visual
                    + opt.logp_geometry
                    + bnd
                    + domain.log_prior(opt.char_id)
                )
            # The ``?`` option's logp_visual already carries the full
            # unknown-log-prob (no separate domain prior, no double count).
            return opt.logp_visual + bnd

        alpha = self.cfg.length_alpha
        beta = self.cfg.length_beta

        def partial_key(e: _PathEntry) -> tuple:
            n = len(e.candidates)
            norm = e.score / ((n + beta) ** alpha)
            return (-norm, e.n_anomalies, e.span, e.text)

        dp: list[list[_PathEntry]] = [[] for _ in range(n_atoms + 1)]
        dp[0] = [_PathEntry(0.0, 0, 0, "")]
        for end in range(1, n_atoms + 1):
            merged: list[_PathEntry] = []
            for cand, opts, bnd in by_end.get(end, ()):
                start = cand.atom_span[0]
                if not dp[start]:
                    continue
                for opt in opts:
                    score = path_score(opt, bnd, domain)
                    ch = (
                        self.model.charset[opt.char_id]
                        if 0 <= opt.char_id < len(self.model.charset)
                        else UNKNOWN_CHAR
                    )
                    anom = 1 if self._is_anomalous(cand, ctx) else 0
                    for prev in dp[start]:
                        merged.append(
                            _PathEntry(
                                score=prev.score + score,
                                n_anomalies=prev.n_anomalies + anom,
                                span=prev.span + (end - start),
                                text=prev.text + ch,
                                candidates=prev.candidates + (cand,),
                                char_ids=prev.char_ids + (opt.char_id,),
                                prev=prev,
                            )
                        )
            dp[end] = self._top_k(merged, k, key=partial_key)
        return dp[n_atoms]

    @staticmethod
    def _top_k(
        entries: list[_PathEntry], k: int, key=None
    ) -> list[_PathEntry]:
        """Deterministic top-k by a total-order path key."""

        if not entries:
            return []
        key = key or _PathEntry.key
        entries = sorted(entries, key=key)
        out: list[_PathEntry] = []
        seen: set[tuple] = set()
        for e in entries:
            if e.ident in seen:
                continue
            seen.add(e.ident)
            out.append(e)
            if len(out) >= k:
                break
        return out

    def decode_lattice(
        self,
        lattice: VisualLattice,
        domain: DomainPrior | None = None,
        allowed_ids: set[int] | None = None,
    ) -> DecodePath:
        """Decode one scored lattice with the probabilistic v2 decoder."""

        domain = domain or OpenTextPrior(tuple(self.model.charset))
        if not lattice.candidates:
            return DecodePath(candidates=(), mean_score=0.0, lattice=lattice)
        self.extractor.reset()
        ctx = self.extractor.line_context(lattice)
        states = self.render_state.select_states(ctx)
        n_atoms = max(c.atom_span[1] for c in lattice.candidates)
        k = self.cfg.k_best

        per_state: dict[str, list[_PathEntry]] = {}
        for z in states:
            per_state[z] = self._kbest_for_state(
                lattice, ctx, z, allowed_ids, domain, n_atoms, k
            )

        # Merge k-best across states, dedupe by path identity.
        merged: list[_PathEntry] = []
        seen: set[tuple] = set()
        for z in states:
            for e in per_state[z]:
                if e.ident in seen:
                    continue
                seen.add(e.ident)
                merged.append(e)
        if not merged:
            return DecodePath(candidates=(), mean_score=0.0, lattice=lattice)

        # Per-path raw scores per state (for the z marginalization).
        per_path_state: dict[tuple, dict[str, float]] = {}
        for z in states:
            for e in per_state[z]:
                per_path_state.setdefault(e.ident, {})[z] = e.score

        def norm_of(ident: tuple) -> float:
            raw = per_path_state[ident]
            total, _argmax = self.render_state.marginalize(
                raw, use_max=self.cfg.max_state_approx
            )
            n = len(ident[0])
            return total / ((n + self.cfg.length_beta) ** self.cfg.length_alpha)

        def final_key(e: _PathEntry) -> tuple:
            return (-norm_of(e.ident), e.n_anomalies, e.span, e.text)

        merged.sort(key=final_key)
        best = merged[0]
        total_log_prob, argmax_z = self.render_state.marginalize(
            per_path_state[best.ident], use_max=self.cfg.max_state_approx
        )
        norm = norm_of(best.ident)
        norm_scores = [norm_of(e.ident) for e in merged]
        lse_all = logsumexp(np.asarray(norm_scores, dtype=np.float64))
        p_best = math.exp(float(np.clip(norm - lse_all, -30.0, 0.0)))
        confidence = float(
            calibrate_monotone(
                np.asarray([p_best], dtype=np.float64),
                self.cfg.confidence_calibration,
            )[0]
        )
        runner_up = norm_scores[1] if len(norm_scores) > 1 else -np.inf
        margin = float(norm - runner_up) if np.isfinite(runner_up) else 0.0

        text = best.text
        alternatives: list[str] = []
        for e in merged[1:]:
            if e.text and e.text != text and e.text not in alternatives:
                alternatives.append(e.text)
            if len(alternatives) >= self.cfg.num_alternatives:
                break

        # Per-char decomposition under the argmax state.
        per_char: list[dict] = []
        for cand, cid in zip(best.candidates, best.char_ids):
            opts = self._options(cand, ctx, argmax_z, allowed_ids)
            ch = (
                self.model.charset[cid]
                if 0 <= cid < len(self.model.charset)
                else UNKNOWN_CHAR
            )
            info = {
                "char": ch,
                "logp_visual": 0.0,
                "logp_geometry": 0.0,
                "logp_boundary": self._boundary_logp(cand, ctx),
                "logp_domain": (
                    domain.log_prior(cid) if cid >= 0 else self.cfg.unknown_log_prob
                ),
            }
            for opt in opts:
                if opt.char_id == cid:
                    info["logp_visual"] = opt.logp_visual
                    info["logp_geometry"] = opt.logp_geometry
            per_char.append(info)

        reject_reason = ""
        if self.cfg.reject_enabled:
            if confidence < self.cfg.reject_confidence:
                reject_reason = (
                    f"confidence {confidence:.3f} < "
                    f"{self.cfg.reject_confidence}"
                )
            elif margin < self.cfg.reject_margin:
                reject_reason = (
                    f"runner-up margin {margin:.3f} < {self.cfg.reject_margin}"
                )

        diag = PathDiagnostics(
            total_log_prob=float(total_log_prob),
            normalized_score=float(norm),
            visual_sum=float(sum(d["logp_visual"] for d in per_char)),
            geometry_sum=float(sum(d["logp_geometry"] for d in per_char)),
            boundary_sum=float(sum(d["logp_boundary"] for d in per_char)),
            domain_sum=float(sum(d["logp_domain"] for d in per_char)),
            line_render_state=argmax_z,
            states=tuple(states),
            runner_up_margin=margin,
            k_best_searched=len(merged),
            reject_reason=reject_reason,
            per_char=tuple(per_char),
        )
        return DecodePath(
            candidates=best.candidates,
            text=text,
            char_ids=best.char_ids,
            mean_score=float(total_log_prob),
            confidence=confidence,
            alternatives=tuple(alternatives),
            lattice=lattice,
            prob_diagnostics=diag.to_dict(),
        )
