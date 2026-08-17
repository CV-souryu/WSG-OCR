"""Raw feature extraction for the probabilistic decoder (v2).

The v2 decoder never consumes the hand-fused ``visual_score`` as its main
evidence. Instead, for every ``(candidate, char_id)`` choice it extracts an
un-fused feature vector from the *frozen* raw evidence the scorer attached
(:class:`~fixedfontocr.types.RawEvidence`), the Goal 8 font-geometry
database and the candidate's own boundary geometry:

* template block -- per-character raw template score (best prototype over
  the character's own prototypes), rank/margin inside the template Top-K,
  exact-match flag, winner prototype metadata (render size, sub-pixel
  phase, downsample mode), coarse-filter hit, and the per-render-state
  score table used by the line-level render state ``z``;
* CNN block -- raw logit of the character, top-1/top-2 logits, margin,
  fused rank, Top-K entropy approximation, evidence-missing flags;
* geometry block -- residuals of the candidate bbox/aspect/ink/component/
  baseline against the character's font-geometry database entry;
* boundary block -- candidate type (single/merge/split), atom/component
  counts, approximate seam evidence (cut deviation from the ideal advance
  and ink crossing the cut), width ratio, piece balance, internal gaps,
  vertical alignment, small-punctuation flag;
* one legacy feature -- the fused ``visual_score`` *as one feature among
  many* (never the only one), so v2 can learn when v1's fusion is
  trustworthy without re-fitting it.

The legacy fused score is never the sole training feature. All features
are deterministic NumPy computations; the only classifier work is the
per-character prototype re-scan (:meth:`TemplateV2Classifier
.char_template_evidence`), which is a pure function of the frozen template
data and identical on CPU and WGPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .classifier import (
    DOWNSAMPLE_AREA,
    DOWNSAMPLE_BILINEAR,
    DOWNSAMPLE_CLEAN,
    TemplateV2Classifier,
)
from .geometry import FontGeometryDatabase
from .model import OCRModel
from .preprocess import Component, glyph_normalize_geometry, normalize
from .types import Profile, VisualCandidate, VisualLattice, default_profile

FEATURE_SCHEMA_VERSION = 1

# Ordered state keys for the line render state z: every (size, mode) pair
# of the Goal 9 grid plus the clean high-res prototype. The order is fixed
# so argmax indices and state ranking are deterministic across models.
_STATE_SIZES = (11, 12, 13, 14, 15, 16)
STATE_KEYS: tuple[str, ...] = tuple(
    f"{size}_{mode}"
    for size in _STATE_SIZES
    for mode in (DOWNSAMPLE_BILINEAR, DOWNSAMPLE_AREA)
) + ("clean",)
CLEAN_STATE = "clean"

FEATURE_NAMES: tuple[str, ...] = (
    # template (unconditional, per char)
    "tpl_score",
    "tpl_rank_one",
    "tpl_margin",
    "tpl_exact",
    "tpl_winner_size",
    "tpl_winner_phase_x",
    "tpl_winner_phase_y",
    "tpl_winner_mode",
    "tpl_coarse_hit",
    "tpl_state_max",
    "tpl_state_argmax",
    # state-conditional (per z)
    "tpl_state_score_z",
    "tpl_state_delta_z",
    "tpl_state_is_clean",
    # cnn
    "cnn_logit_c",
    "cnn_logit_c_present",
    "cnn_top1_logit",
    "cnn_top2_logit",
    "cnn_margin_c",
    "cnn_rank_c",
    "cnn_entropy_top5",
    "cnn_top1_prob_top5",
    "cnn_missing",
    "cnn_input_mode_soft",
    # geometry residuals (per char, vs database entry)
    "geo_advance_resid",
    "geo_width_resid",
    "geo_height_resid",
    "geo_aspect_resid",
    "geo_ink_resid",
    "geo_comp_resid",
    "geo_baseline_resid",
    "geo_type_match",
    "geo_multi_comp_match",
    # boundary (candidate-level, char-independent)
    "bnd_single",
    "bnd_merge",
    "bnd_split",
    "bnd_atoms",
    "bnd_components",
    "bnd_seam_ink",
    "bnd_seam_dev",
    "bnd_width_ratio",
    "bnd_balance",
    "bnd_gap_max",
    "bnd_align_offset",
    "bnd_punct_small",
    "bnd_seg_width_penalty",
    # legacy fused evidence (one feature among many)
    "legacy_fused",
    "legacy_rank",
)

N_FEATURES = len(FEATURE_NAMES)

# Feature index ranges used by the two training stages / diagnostics.
VISUAL_FEATURE_IDX = tuple(range(0, 24)) + (N_FEATURES - 2, N_FEATURES - 1)
GEOMETRY_FEATURE_IDX = tuple(range(24, 33))
BOUNDARY_FEATURE_IDX = tuple(range(33, 46))


@dataclass(frozen=True)
class LineContext:
    """Per-line quantities shared by every candidate feature vector."""

    expected_width: float
    median_center_y: float
    median_h: float
    # atom_index -> (component_index, atom_index_within_component,
    #                atoms_in_component) for split-piece detection.
    atom_runs: tuple[tuple[int, int, int], ...] = ()
    # Atoms of the line (frozen segmentation output), for cut geometry.
    atoms: tuple = ()
    # Deterministic candidate state ranking: (state_key, support).
    state_candidates: tuple[tuple[str, float], ...] = ()


@dataclass
class _CandCache:
    """Per-candidate extracted pieces (reused across characters/states)."""

    boundary: np.ndarray
    glyph: np.ndarray | None
    template_evidence: dict[int, dict] = field(default_factory=dict)
    # Raw CNN logits per char id (used when no RawEvidence was attached,
    # e.g. offline training on v1 models). ``None`` means "no CNN pass".
    cnn_logits: dict[int, float] | None = None
    # 46-dim base feature vector per char id (z-independent part cached).
    base_features: dict[int, np.ndarray] = field(default_factory=dict)


class LocalFeatureExtractor:
    """Extracts the schema-v1 feature vector for ``(candidate, char, z)``.

    ``template_v2`` is an optional CPU :class:`TemplateV2Classifier` built
    from the model's V2 data; it is used for the per-character prototype
    re-scans (state tables / raw per-char template scores). When absent
    (V1 template models), the per-char template block falls back to the
    raw scores stored in :class:`RawEvidence` and the state table is empty.
    """

    def __init__(
        self,
        model: OCRModel,
        profile: Profile | None = None,
        template_v2: TemplateV2Classifier | None = None,
    ):
        self.model = model
        self.profile = profile or default_profile()
        self.geometry: FontGeometryDatabase | None = model.geometry
        self.charset = list(model.charset)
        self.template_v2 = template_v2 or self._auto_template_v2(model)
        self.input_size = model.input_size
        self.normalize_spec = model.normalize_spec
        self._caches: dict[int, _CandCache] = {}

    @staticmethod
    def _auto_template_v2(model: OCRModel):
        """Build the CPU Template V2 matcher when the model has V2 data."""

        if model.templates_v2 is None:
            return None
        from .classifier import TemplateV2Classifier

        return TemplateV2Classifier(
            data=model.templates_v2,
            charset=model.charset,
            input_size=model.input_size,
            normalize_spec=model.normalize_spec,
        )

    def reset(self) -> None:
        """Drop per-line caches (call once per decoded line)."""

        self._caches.clear()

    # ------------------------------------------------------------------
    # Line-level context
    # ------------------------------------------------------------------

    def line_context(
        self,
        lattice: VisualLattice,
        comps: list | None = None,
    ) -> LineContext:
        """Compute expected width, alignment band and state candidates."""

        from .segmentation import _estimate_expected_width, expand_atoms

        comps = comps if comps is not None else list(lattice.components)
        expected = float(
            _estimate_expected_width(
                [c for c in comps if getattr(c, "mask", None) is not None],
                self.profile,
                self.geometry,
            )
        )
        centers = [c.y + c.h / 2 for c in comps]
        median_center = float(np.median(centers)) if centers else 0.0
        heights = [c.h for c in comps]
        median_h = float(np.median(heights)) if heights else 1.0

        runs: list[tuple[int, int, int]] = []
        atoms: tuple = ()
        try:
            atoms, _ = expand_atoms(
                comps,
                self.profile,
                4,
                split_wide=True,
                geometry=self.geometry,
            )
            per_comp: dict[int, list[int]] = {}
            for ai, atom in enumerate(atoms):
                per_comp.setdefault(atom.component_index, []).append(ai)
            for ai, atom in enumerate(atoms):
                idxs = per_comp[atom.component_index]
                runs.append((atom.component_index, idxs.index(ai), len(idxs)))
        except Exception:
            # Manual lattices without pipeline atoms: no split metadata.
            runs = []

        states = self._line_states(lattice)
        return LineContext(
            expected_width=expected,
            median_center_y=median_center,
            median_h=median_h,
            atom_runs=tuple(runs),
            atoms=atoms,
            state_candidates=states,
        )

    def _candidate_glyph(self, cand: VisualCandidate) -> np.ndarray | None:
        """The normalized binary glyph of a candidate (raw evidence or
        re-normalized from the segment; byte-identical)."""

        raw = cand.scores.raw_evidence if cand.scores is not None else None
        if raw is not None and raw.glyph is not None:
            return raw.glyph
        seg = cand.segment
        if seg is None:
            return None
        geom = cand.normalize_geometry
        if geom is None and self.normalize_spec is not None:
            geom = glyph_normalize_geometry(
                Component(mask=seg.mask, x=seg.x, y=seg.y, w=seg.w, h=seg.h),
                self.normalize_spec,
                self.input_size,
            )
        if self.normalize_spec is not None and geom is not None:
            return normalize(
                seg.mask,
                self.input_size,
                baseline_offset=geom[0],
                scale=geom[1],
                baseline_row=self.normalize_spec.baseline_row,
            )
        return normalize(seg.mask, self.input_size)

    def _line_states(
        self, lattice: VisualLattice,
    ) -> tuple[tuple[str, float], ...]:
        """Line-level render-state candidates from the candidates' winners.

        Every candidate's fused top-1 character's best prototype votes for
        its ``(render_size, downsample_mode)`` state; states are ranked by
        total support (ties by state key order). ``clean`` is always
        included as the high-resolution fallback. The vote is a pure
        function of (glyph, template data), so runtime (RawEvidence) and
        offline training (re-normalized glyphs) see identical states.
        """

        support: dict[str, float] = {CLEAN_STATE: 0.0}
        for cand in lattice.candidates:
            scores = cand.scores
            if scores is None or not scores.char_ids:
                continue
            glyph = self._candidate_glyph(cand)
            if glyph is None or self.template_v2 is None:
                continue
            top_cid = int(scores.char_ids[0])
            ev = self.template_v2.char_template_evidence(glyph, [top_cid])[0]
            size, _dx, _dy, mode = ev.get("winner_meta", (-1, -1, -1, -1))
            if size > 0 and mode in (DOWNSAMPLE_BILINEAR, DOWNSAMPLE_AREA):
                key = f"{int(size)}_{int(mode)}"
                support[key] = support.get(key, 0.0) + 1.0
        ranked = sorted(
            support.items(), key=lambda kv: (-kv[1], kv[0])
        )
        return tuple(ranked)

    # ------------------------------------------------------------------
    # Candidate-level pieces
    # ------------------------------------------------------------------

    def _candidate_cache(
        self,
        cand: VisualCandidate,
        ctx: LineContext,
        comps: list | None,
    ) -> _CandCache:
        key = id(cand)
        cached = self._caches.get(key)
        if cached is not None:
            return cached
        glyph = self._candidate_glyph(cand)
        boundary = self._boundary_features(cand, ctx, comps)
        cache = _CandCache(boundary=boundary, glyph=glyph)
        self._caches[key] = cache
        return cache

    def _boundary_features(
        self,
        cand: VisualCandidate,
        ctx: LineContext,
        comps: list | None,
    ) -> np.ndarray:
        """Candidate-level boundary block (schema positions 33..45)."""

        out = np.zeros(13, dtype=np.float64)
        seg = cand.segment
        start, end = cand.atom_span
        n_atoms = end - start
        n_components = max(1, len(cand.components))
        # Candidate type from the atom/component structure.
        run = None
        if start < len(ctx.atom_runs):
            run = ctx.atom_runs[start]
        is_split_piece = (
            n_atoms == 1 and run is not None and run[2] > 1
        )
        is_merge = n_components > 1
        is_split_whole = n_atoms > 1 and n_components == 1
        single = int(not is_split_piece and not is_merge and not is_split_whole)
        merge = int(is_merge)
        split = int(is_split_piece or is_split_whole)
        out[0], out[1], out[2] = single, merge, split
        out[3] = float(n_atoms)
        out[4] = float(n_components)

        if seg is None:
            return out
        h = max(seg.h, 1)
        w = max(seg.w if seg is not None else 1, 1)
        exp_w = max(ctx.expected_width, 1.0)
        out[7] = float(np.clip(w / exp_w, 0.0, 4.0))
        out[10] = float(
            np.clip(abs((seg.y + h / 2) - ctx.median_center_y) / max(h, 1), 0.0, 4.0)
        )
        out[11] = float(
            1.0 if h < max(2, self.profile.char_height_min) else 0.0
        )
        out[12] = float(getattr(cand, "segmentation_width_penalty", 0.0))

        if is_merge and comps is not None:
            idxs = list(cand.components)
            gaps = [
                comps[k + 1].x - (comps[k].x + comps[k].w)
                for k in range(idxs[0], idxs[-1])
                if k + 1 < len(comps)
            ]
            if gaps:
                out[8] = float(np.clip(max(gaps) / max(h, 1), 0.0, 4.0))

        if split and ctx.atoms:
            # Approximate seam evidence from the atoms adjacent to the cut.
            pieces = self._cut_pieces(ctx, start, end)
            if pieces is not None:
                (l_seg, r_seg), ink_ratio, cut_x = pieces
                out[5] = float(np.clip(ink_ratio, 0.0, 1.0))
                ideal = max(1.0, round(cut_x / exp_w) * exp_w)
                out[6] = float(np.clip(abs(cut_x - ideal) / exp_w, 0.0, 4.0))
                out[9] = float(
                    min(l_seg.w, r_seg.w) / max(max(l_seg.w, r_seg.w), 1.0)
                )
        return out

    def _cut_pieces(
        self,
        ctx: LineContext,
        start: int,
        end: int,
    ) -> tuple[tuple, float, float] | None:
        """Best (lowest crossing-ink) cut inside/at the edges of the span.

        Returns ``((left_seg, right_seg), ink_ratio, cut_x_image)`` or
        ``None`` when no adjacent atom pair is available. For a single
        split piece (span length 1) the candidate cut is the boundary
        between that piece and its neighbour (left or right, whichever has
        less crossing ink); for a split-whole span it is the best internal
        cut. Uses only the atom geometry recorded by the frozen
        segmentation output, so the exact seam path/cost is approximated
        (documented limitation).
        """

        if not ctx.atoms or end > len(ctx.atoms):
            return None
        atoms = ctx.atoms

        def pair_ink(a: int) -> tuple[tuple, float, float] | None:
            if a < 0 or a + 1 >= len(atoms):
                return None
            la, ra = atoms[a].segment, atoms[a + 1].segment
            if la is None or ra is None:
                return None
            # Adjacent atoms may belong to different components with
            # different heights: compare per-column ink *fractions*.
            def col_frac(seg, col: int) -> float:
                if seg.mask.shape[1] <= col or seg.h < 1:
                    return 0.0
                return float(np.count_nonzero(seg.mask[:, col])) / seg.h

            left_frac = col_frac(la, la.mask.shape[1] - 1)
            right_frac = col_frac(ra, 0)
            ink = (left_frac + right_frac) / 2.0
            cut_x = (la.x + la.w + ra.x) / 2.0
            return ((la, ra), ink, cut_x)

        if end - start == 1:
            # Single split piece: the cut is at one of its two edges.
            left = pair_ink(start - 1)
            right = pair_ink(start)
            candidates = [p for p in (left, right) if p is not None]
            if not candidates:
                return None
            return min(candidates, key=lambda p: p[1])
        best = None
        best_ink = float("inf")
        for a in range(start, end - 1):
            p = pair_ink(a)
            if p is not None and p[1] < best_ink:
                best_ink = p[1]
                best = p
        return best

    # ------------------------------------------------------------------
    # Per-character features
    # ------------------------------------------------------------------

    def char_features(
        self,
        cand: VisualCandidate,
        char_id: int,
        ctx: LineContext,
        z: str | None = None,
        comps: list | None = None,
        template_evidence: dict | None = None,
    ) -> np.ndarray:
        """Full schema-v1 feature vector for one ``(cand, char, z)``.

        The 46 z-independent features are cached per ``(cand, char)``;
        only the three z-conditional template features are recomputed per
        state, which keeps the per-state decode cost low.
        """

        cache = self._candidate_cache(cand, ctx, comps)
        boundary = cache.boundary
        raw = cand.scores.raw_evidence if cand.scores is not None else None

        base = cache.base_features.get(int(char_id))
        if base is None:
            base = self._base_features(cand, int(char_id), ctx, cache, raw)
            cache.base_features[int(char_id)] = base
        f = base.copy()
        ev = cache.template_evidence.get(int(char_id))
        tpl_score = float(f[0])
        z_key = z if z is not None else CLEAN_STATE
        if z_key not in STATE_KEYS and z is not None:
            z_key = CLEAN_STATE
        if ev is not None and ev.get("states"):
            z_score = float(ev["states"].get(z_key, 0.0))
            f[11] = z_score
            f[12] = float(np.clip(z_score - tpl_score, -1.0, 0.0))
        f[13] = 1.0 if z_key == CLEAN_STATE else 0.0
        return f

    def _base_features(
        self,
        cand: VisualCandidate,
        char_id: int,
        ctx: LineContext,
        cache: _CandCache,
        raw,
    ) -> np.ndarray:
        """The 46 z-independent features of one (cand, char)."""

        boundary = cache.boundary
        # ---- template block (positions 0..10, z-features 11..13 later) --
        tpl = np.zeros(11, dtype=np.float64)
        ev = cache.template_evidence.get(int(char_id))
        tpl_score = 0.0
        if self.template_v2 is not None:
            if ev is None:
                ev = self._template_evidence(cand, char_id)
                cache.template_evidence[int(char_id)] = ev
            tpl_score = float(ev.get("best_score", 0.0))
            meta = ev.get("winner_meta", (-1, -1, -1, -1))
            tpl[0] = tpl_score
            tpl[4] = float(meta[0]) / 32.0 if meta[0] > 0 else 0.0
            tpl[5] = float(meta[1]) / 8.0 if meta[1] >= 0 else 0.0
            tpl[6] = float(meta[2]) / 8.0 if meta[2] >= 0 else 0.0
            tpl[7] = float(meta[3]) / 2.0 if meta[3] >= 0 else 0.0
            tpl[8] = float(ev.get("coarse_hit", False))
            states = ev.get("states", {})
            if states:
                vals = np.asarray(
                    [states.get(k, 0.0) for k in STATE_KEYS], dtype=np.float64
                )
                tpl[9] = float(np.max(vals))
                tpl[10] = float(int(np.argmax(vals)))
            else:
                tpl[9] = tpl_score
        elif raw is not None and raw.template_scores:
            ids = list(raw.char_ids)
            if int(char_id) in ids:
                tpl_score = float(raw.template_scores[ids.index(int(char_id))])
            tpl[0] = tpl_score
            tpl[9] = tpl_score
        # Rank/margin/exact inside the template Top-K (from RawEvidence).
        if raw is not None:
            ids = [int(c) for c in raw.char_ids]
            tpl_scores = [float(s) for s in raw.template_scores]
            if int(char_id) in ids:
                k = ids.index(int(char_id))
                top1 = max(tpl_scores) if tpl_scores else 0.0
                tpl[1] = 1.0 if tpl_score > 0.0 and tpl_score >= top1 else 0.0
                tpl[2] = float(np.clip(tpl_score - top1, -1.0, 0.0))
                tpl[3] = 1.0 if tpl_score >= 1.0 - 1e-9 else 0.0

        # ---- z-features (positions 11..13, filled by char_features) ----
        zfeat = np.zeros(3, dtype=np.float64)

        # ---- CNN block (positions 14..23) ------------------------------
        cnn = np.zeros(10, dtype=np.float64)
        has_cnn = bool(self.model.weights is not None)
        if has_cnn and raw is not None:
            ids = [int(c) for c in raw.char_ids]
            logits = [float(l) for l in raw.cnn_logits]
            top1 = float(raw.cnn_top1_logit)
            top2 = float(raw.cnn_top2_logit)
            if int(char_id) in ids:
                k = ids.index(int(char_id))
                lc = logits[k]
                cnn[0] = float(np.clip(lc, -30.0, 30.0))
                cnn[1] = 1.0 if np.isfinite(lc) else 0.0
                cnn[5] = float(k + 1)
                if np.isfinite(lc) and np.isfinite(top2):
                    cnn[4] = float(np.clip(lc - top2, -30.0, 30.0))
            else:
                cnn[5] = 6.0
            cnn[2] = float(np.clip(top1, -30.0, 30.0)) if np.isfinite(top1) else 0.0
            cnn[3] = float(np.clip(top2, -30.0, 30.0)) if np.isfinite(top2) else 0.0
            finite = [l for l in logits if np.isfinite(l)]
            if len(finite) >= 2:
                arr = np.asarray(finite, dtype=np.float64)
                arr = arr - np.max(arr)
                p = np.exp(arr)
                p = p / np.sum(p)
                entropy = -float(np.sum(p * np.log(p + 1e-12)))
                cnn[6] = float(entropy / math.log(max(len(p), 2)))
                cnn[7] = float(np.max(p))
        elif has_cnn:
            cache_logits = cache.cnn_logits if cache is not None else None
            if cache_logits:
                vals = [
                    (float(v), int(c))
                    for c, v in cache_logits.items()
                    if np.isfinite(v)
                ]
                vals.sort(reverse=True)
                if vals:
                    top1 = vals[0][0]
                    top2 = vals[1][0] if len(vals) > 1 else -np.inf
                    lc = cache_logits.get(int(char_id))
                    cnn[0] = float(np.clip(lc, -30.0, 30.0)) if lc is not None else 0.0
                    cnn[1] = 1.0 if lc is not None and np.isfinite(lc) else 0.0
                    cnn[2] = float(np.clip(top1, -30.0, 30.0))
                    cnn[3] = float(np.clip(top2, -30.0, 30.0)) if np.isfinite(top2) else 0.0
                    if lc is not None and np.isfinite(lc) and np.isfinite(top2):
                        cnn[4] = float(np.clip(lc - top2, -30.0, 30.0))
                    rank = next(
                        (i + 1 for i, (_v, c) in enumerate(vals) if c == int(char_id)),
                        6,
                    )
                    cnn[5] = float(rank)
                    finite = [v for v, _c in vals]
                    if len(finite) >= 2:
                        arr = np.asarray(finite, dtype=np.float64)
                        arr = arr - np.max(arr)
                        p = np.exp(arr)
                        p = p / np.sum(p)
                        entropy = -float(np.sum(p * np.log(p + 1e-12)))
                        cnn[6] = float(entropy / math.log(max(len(p), 2)))
                        cnn[7] = float(np.max(p))
        cnn[8] = 1.0 if not has_cnn else 0.0
        cnn[9] = 1.0 if self.model.input_mode == "soft" else 0.0

        # ---- geometry block (positions 24..32) -------------------------
        geo = self._geometry_features(cand, char_id)

        # ---- legacy fused feature (positions 46..47) -------------------
        fused = np.zeros(2, dtype=np.float64)
        if cand.scores is not None and cand.scores.char_ids:
            ids = [int(c) for c in cand.scores.char_ids]
            logits = list(cand.scores.logits or ())
            if int(char_id) in ids:
                k = ids.index(int(char_id))
                fused[0] = (
                    float(np.clip(logits[k], 0.0, 1.0))
                    if k < len(logits)
                    else 0.0
                )
                fused[1] = float(k + 1)
            else:
                fused[1] = 6.0

        return np.concatenate([tpl, zfeat, cnn, geo, boundary, fused])

    def _geometry_features(
        self,
        cand: VisualCandidate,
        char_id: int,
    ) -> np.ndarray:
        """Geometry residuals of ``(cand, char_id)`` (positions 24..32)."""

        out = np.zeros(9, dtype=np.float64)
        seg = cand.segment
        if self.geometry is None or seg is None or seg.h < 1 or seg.w < 1:
            return out
        entry = self.geometry.get(int(char_id))
        if entry is None:
            return out
        h = max(seg.h, 1)
        w = max(seg.w, 1)
        em = max(
            h / max(entry.bbox_height, 1e-6),
            w / max(entry.bbox_width, 1e-6),
            1.0,
        )
        w_ratio = w / em
        h_ratio = h / em
        out[0] = float(
            np.clip(
                (w_ratio - entry.advance) / max(entry.advance, 1e-3), -4.0, 4.0
            )
        )
        out[1] = float(
            np.clip(
                (w_ratio - entry.bbox_width) / max(entry.bbox_width, 1e-3),
                -4.0,
                4.0,
            )
        )
        out[2] = float(
            np.clip(
                (h_ratio - entry.bbox_height) / max(entry.bbox_height, 1e-3),
                -4.0,
                4.0,
            )
        )
        aspect = w / h
        out[3] = float(
            np.clip(
                math.log(aspect / max(entry.aspect_ratio, 1e-3)), -4.0, 4.0
            )
        )
        ink_ratio = float(seg.mask.sum()) / float(w * h)
        out[4] = float(
            np.clip(
                (ink_ratio - entry.ink_ratio) / max(entry.ink_ratio, 0.05),
                -4.0,
                4.0,
            )
        )
        actual_cc = max(1, len(cand.components))
        expected_cc = max(1, entry.component_count)
        out[5] = float(np.clip(actual_cc - expected_cc, -8.0, 8.0))
        ng = cand.normalize_geometry
        if ng is not None and ng[0] > 0 and 0.0 < entry.baseline_ratio <= 1.5:
            out[6] = float(
                np.clip(ng[0] / h - entry.baseline_ratio, -2.0, 2.0)
            )
        out[7] = 1.0 if (aspect < 0.8) == (entry.aspect_ratio < 0.8) else 0.0
        out[8] = 1.0 if (actual_cc > 1) == (expected_cc > 1) else 0.0
        return out

    def _template_evidence(self, cand: VisualCandidate, char_id: int) -> dict:
        """Per-character template re-scan (pure function of glyph + data)."""

        cache = self._caches.get(id(cand))
        glyph = cache.glyph if cache is not None else None
        if glyph is None or self.template_v2 is None:
            return {"best_score": 0.0, "states": {}, "coarse_hit": False}
        if not cache.template_evidence:
            # Batch-scan the candidate's fused Top-K chars in one call.
            scores = cand.scores
            if scores is not None and scores.char_ids:
                cids = [int(c) for c in scores.char_ids]
                evs = self.template_v2.char_template_evidence_batch(glyph, cids)
                for ev in evs:
                    cache.template_evidence[int(ev["char_id"])] = ev
        return cache.template_evidence.get(
            int(char_id),
            {"best_score": 0.0, "states": {}, "coarse_hit": False},
        )

    # ------------------------------------------------------------------
    # Feature normalization
    # ------------------------------------------------------------------

    def normalize(
        self,
        features: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        clip: float = 4.0,
    ) -> np.ndarray:
        """Standardize features; zero std maps to 0."""

        f = np.asarray(features, dtype=np.float64)
        m = np.asarray(mean, dtype=np.float64)
        s = np.asarray(std, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(s > 1e-12, (f - m) / np.where(s > 1e-12, s, 1.0), 0.0)
        return np.clip(out, -clip, clip)
