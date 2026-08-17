#!/usr/bin/env python3
"""Two-stage probabilistic decoder training (v2).

Builds the ``probabilistic/v2`` decoder of ``model/game_cn_prob``:

1. **Corpus**: the frozen real game samples (``tests/game_samples``,
   grouped by source screenshot/session) plus freshly rendered synthetic
   lines (each render is its own session, so no adjacent-crop leakage).
2. **Harvest**: for every line the frozen pipeline runs once; the scored
   lattice, the ground-truth path (oracle boxes for synthetic renders,
   GT-text DP for real crops) and the per-``(candidate, char)`` raw
   feature vectors (schema v1) are stored. Segmentation output is treated
   as frozen input; oracle/coverage failures are recorded, never silently
   skipped.
3. **Grouped split**: train/validation/test by session (fixed seed),
   saved as a JSON manifest.
4. **Stage 1 (local ranker)**: pairwise logistic over the visual block --
   the correct character vs the candidate's Top-K wrong characters
   (hard negatives), with the line's render state as the z-condition.
5. **Stage 2 (structured)**: with the local ranker frozen, learn the
   geometry/boundary log-evidence weights, the local temperature and the
   background logit by pairwise logistic over the k-best paths (GT path
   positive, competing paths negative); ``(alpha, beta)`` length
   normalization is selected on the validation split.
6. **Calibration**: temperature + monotone confidence calibration and the
   reject thresholds, fitted on validation only.
7. **Export**: ``model/game_cn_prob`` (self-contained model copy with the
   versioned ``decoder`` block in both ``config.json`` and ``model.json``)
   and a full metrics report (oracle recall, Top-K coverage, local top-1,
   exact match, CER, NLL, Brier, ECE, reject rate, wrong-association
   rate) per split.

Usage:
    python tools/train/train_probabilistic_decoder.py \
        --model model/game_cn --out out/prob --seed 0

Runtime is pure NumPy (no torch/scikit-learn); the exported decoder block
is a small JSON coefficient set.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fixedfontocr.defaults import (  # noqa: E402
    FONT_PATH,
    MODEL_PATH,
    resolve_font,
)
from fixedfontocr.model import load_model  # noqa: E402
from fixedfontocr.oracle import (  # noqa: E402
    GroundTruthBox,
    render_ground_truth,
)
from fixedfontocr.prob_features import (  # noqa: E402
    BOUNDARY_FEATURE_IDX,
    CLEAN_STATE,
    FEATURE_SCHEMA_VERSION,
    GEOMETRY_FEATURE_IDX,
    N_FEATURES,
    STATE_KEYS,
    VISUAL_FEATURE_IDX,
    LocalFeatureExtractor,
)
from fixedfontocr.prob_math import (  # noqa: E402
    calibrate_monotone,
    calibration_metrics,
    logsumexp,
    sequence_cer,
)
from fixedfontocr.render_state import RenderStateModel  # noqa: E402

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

SPECIAL_LINES = (
    "鲃鱼。",
    "小",
    "潜甲",
    "潜乙",
    "巴尔的摩",
    "塞瓦斯托波尔",
    "Z17",
    "获得金币1000",
    "甲申",
    "1234",
    "未",
    "末",
    "舰船Lv.99",
    "初雪",
    "乌戈里尼·维瓦尔迪",
    "T-23",
    "Lv.40",
    "金币+1000",
    # hard real-game shapes: touching glyphs, U-boats, level badges
    "U-156",
    "U-96",
    "K-21",
    "LV.26",
    "LV.24",
    "Z28",
    "Z1",
    "47工程",
    "4+3",
    "甲申Z17",
    "塞瓦斯托波尔。",
    "华盛顿",
    "初雪",
    "乌戈里尼",
    "维瓦尔迪",
    "金币1000",
    "Lv.12",
    "Lv.99",
    "Z17 Z28",
    "0/",
    "试制四联41厘米主炮",
    "双联20厘米炮(AH)",
)


def _render_line(
    text: str,
    size: int,
    mode: str,
    phase: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Supersampled low-res render (bilinear/area) or clean render.

    Returns an RGB uint8 image of the text line. ``mode`` is
    ``"bilinear"`` / ``"area"`` for the Goal 7 degradation domain and
    ``"clean"`` for the high-res render.
    """

    from PIL import Image, ImageDraw, ImageFont

    font = resolve_font(FONT_PATH)
    if mode == "clean":
        f = ImageFont.truetype(str(font), size)
        bbox = f.getbbox(text)
        pad = 8
        w = bbox[2] - bbox[0] + pad * 2
        h = bbox[3] - bbox[1] + pad * 2
        img = Image.new("RGB", (max(1, w), max(1, h)), (0, 0, 0))
        ImageDraw.Draw(img).text(
            (pad - bbox[0], pad - bbox[1]), text, font=f, fill=(255, 255, 255)
        )
        return np.asarray(img, dtype=np.uint8)
    ss = 3
    f = ImageFont.truetype(str(font), size * ss)
    bbox = f.getbbox(text)
    pad = 8 * ss
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    img = Image.new("RGB", (max(1, w), max(1, h)), (0, 0, 0))
    ImageDraw.Draw(img).text(
        (pad - bbox[0] + phase[0] * ss, pad - bbox[1] + phase[1] * ss),
        text,
        font=f,
        fill=(255, 255, 255),
    )
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    tw = max(1, int(round(w / ss)))
    th = max(1, int(round(h / ss)))
    if mode == "bilinear":
        small = Image.fromarray(gray).resize(
            (tw, th), Image.BILINEAR
        )
    else:  # area-like (BOX)
        small = Image.fromarray(gray).resize((tw, th), Image.BOX)
    small = np.asarray(small, dtype=np.uint8)
    rgb = np.repeat(small[:, :, None], 3, axis=2)
    return rgb


def _synthetic_corpus(
    words_dir: Path,
    n_words: int = 140,
    rng=None,
) -> list[dict]:
    """Synthetic line records: text, size, mode, phase, session."""

    rng = rng or np.random.default_rng(0)
    lines: list[str] = list(SPECIAL_LINES)
    for filename in ("ship_names.txt", "equipment_names.txt", "ui_texts.txt"):
        p = words_dir / filename
        if not p.is_file():
            continue
        terms = [
            t.strip()
            for t in p.read_text(encoding="utf-8").splitlines()
            if t.strip()
        ]
        if terms:
            idx = rng.choice(len(terms), size=min(n_words, len(terms)), replace=False)
            lines.extend(terms[int(i)] for i in idx)
    lines = list(dict.fromkeys(lines))

    out: list[dict] = []
    for i, text in enumerate(lines):
        if len(text) > 14:
            continue
        size = int(rng.integers(11, 19))
        mode = "bilinear" if rng.random() < 0.5 else "area"
        phase = (float(rng.integers(0, 3)) / 3.0, 0.0)
        out.append(
            {
                "text": text,
                "size": size,
                "mode": mode,
                "phase": phase,
                "session": f"synth_{i:04d}",
            }
        )
    # A clean-render slice for the fallback state.
    for i, text in enumerate(lines[:40]):
        out.append(
            {
                "text": text,
                "size": 32,
                "mode": "clean",
                "phase": (0.0, 0.0),
                "session": f"synth_clean_{i:04d}",
            }
        )
    return out


def _real_corpus(manifest: dict, samples_dir: Path) -> list[dict]:
    """Real game sample records (session = source screenshot)."""

    import re

    out: list[dict] = []
    for s in manifest["samples"]:
        f = s["file"]
        path = samples_dir / f
        if not path.is_file():
            continue
        m = re.match(r"real-game/ship-name/vp(\d+)_", f)
        if m:
            session = f"vp{m.group(1)}"
        elif f.startswith("real-game/level/"):
            session = f.split("/")[-1].split("_level")[0]
        else:
            session = f.split("/")[0] + "/" + Path(f).stem
        out.append(
            {
                "text": s["expected"],
                "image": path,
                "session": session,
                "profile": s.get("profile", {}),
                "lexicon": s.get("lexicon"),
                "matched_term": s.get("matched_term"),
                "category": s["category"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Harvest
# ---------------------------------------------------------------------------


@dataclass
class LineRecord:
    """Everything the two training stages need for one line."""

    line_id: int
    session: str
    text: str
    split: str = ""
    n_atoms: int = 0
    oracle: bool = False
    oracle_reason: str = ""
    coverage_failures: tuple[int, ...] = ()
    gt_candidates: tuple[int, ...] = ()  # candidate index per GT position
    gt_char_ids: tuple[int, ...] = ()
    # per candidate: (span, components, char_ids, [48-dim base features
    # per char], [13-dim state table per char], boundary features)
    candidates: tuple = ()
    image: object = None
    gt_z: str = ""
    state_candidates: tuple = ()
    category: str = ""


LineRecord.__module__ = "train_probabilistic_decoder"


def _text_path_dp(lattice, gt_text, model, extractor, ctx):
    """Best tiling of the lattice whose text equals ``gt_text``.

    Returns the candidate indices per GT position (or None). Maximizes
    the sum of the GT character's raw template score along the path, so
    among ambiguous tilings the visually best one wins deterministically.
    """

    n = max(c.atom_span[1] for c in lattice.candidates)
    by_start: dict[int, list[tuple[int, object]]] = {}
    for i, cand in enumerate(lattice.candidates):
        by_start.setdefault(cand.atom_span[0], []).append((i, cand))
    NEG = -1e18
    tables: list[list[tuple[float, object]]] = []
    dp: list[tuple[float, object]] = [(NEG, None)] * (n + 1)
    dp[0] = (0.0, None)
    for pos, ch in enumerate(gt_text):
        cid = model.charset.index(ch) if ch in model.charset else -1
        if cid < 0:
            return None
        nxt: list[tuple[float, object]] = [(NEG, None)] * (n + 1)
        for end in range(1, n + 1):
            best = NEG
            best_prev = None
            for start in range(0, end):
                if dp[start][0] <= NEG / 2:
                    continue
                for i, cand in by_start.get(start, ()):
                    if cand.atom_span[1] != end:
                        continue
                    if cand.scores is None:
                        continue
                    ids = [int(c) for c in cand.scores.char_ids]
                    if cid in ids:
                        k = ids.index(cid)
                        score = float(cand.scores.logits[k])
                    else:
                        ev = extractor._template_evidence(cand, cid)
                        score = float(ev.get("best_score", 0.0)) - 2.0
                    tot = dp[start][0] + score
                    if tot > best:
                        best = tot
                        best_prev = (start, i)
            nxt[end] = (best, best_prev)
        tables.append(dp)
        dp = nxt
    if dp[n][0] <= NEG / 2:
        return None
    chosen: list[int] = []
    pos = n
    for k in range(len(gt_text) - 1, -1, -1):
        prev = dp[pos][1]
        if prev is None:
            return None
        start, i = prev
        chosen.append(i)
        pos = start
        dp = tables[k]
    chosen.reverse()
    return tuple(chosen)


def harvest_line(
    line_id: int,
    rec: dict,
    scorer,
    model,
    profile,
    extractor: LocalFeatureExtractor,
    oracle_boxes=None,
) -> LineRecord:
    """Run the frozen pipeline on one line and harvest training data."""

    from fixedfontocr.frontend import extract_frontend
    from fixedfontocr.preprocess import find_lines
    from fixedfontocr.segmentation import segment_line
    from fixedfontocr.types import profile_from_dict

    if "image" in rec and rec["image"] is not None:
        from PIL import Image as _PILImage

        image = np.asarray(
            _PILImage.open(rec["image"]).convert("RGB"), dtype=np.uint8
        )
    else:
        image = _render_line(
            rec["text"], rec["size"], rec["mode"], rec.get("phase", (0.0, 0.0))
        )
    prof = profile_from_dict(rec.get("profile") or {}) if rec.get("profile") else profile
    frontend = extract_frontend(image, prof)
    lines = find_lines(frontend.binary_mask, prof)
    if len(lines) != 1:
        return LineRecord(
            line_id=line_id,
            session=rec["session"],
            text=rec["text"],
            oracle=False,
            oracle_reason="pipeline_found_no_single_line",
        )
    line = lines[0]
    path = segment_line(line, prof, scorer)
    lattice = path.lattice
    if not lattice.candidates:
        return LineRecord(
            line_id=line_id,
            session=rec["session"],
            text=rec["text"],
            oracle=False,
            oracle_reason="empty_lattice",
        )
    n_atoms = max(c.atom_span[1] for c in lattice.candidates)

    # --- ground-truth path ---------------------------------------------
    gt_cands: tuple[int, ...] = ()
    oracle = True
    reason = ""
    if oracle_boxes is not None:
        from fixedfontocr.oracle import oracle_lattice_recall

        res = oracle_lattice_recall(
            line,
            prof,
            oracle_boxes,
            geometry=scorer.geometry,
            candidates=list(lattice.candidates),
        )
        if res.recall:
            gt_cands = tuple(id(c) for c in res.path)
        else:
            oracle = False
            reason = res.reason
    if not gt_cands:
        extractor.reset()
        ctx = extractor.line_context(lattice)
        found = _text_path_dp(lattice, rec["text"], model, extractor, ctx)
        if found is not None:
            gt_cands = found
            if not oracle:
                reason = f"text_dp ({reason})"
        elif not oracle:
            pass
        else:
            oracle = False
            reason = "text_dp_missing"
    if not gt_cands:
        return LineRecord(
            line_id=line_id,
            session=rec["session"],
            text=rec["text"],
            oracle=False,
            oracle_reason=reason or "no_gt_path",
            category=rec.get("category", ""),
        )

    gt_char_ids = tuple(
        model.charset.index(ch) if ch in model.charset else -1
        for ch in rec["text"]
    )
    coverage_failures: list[int] = []
    # Attach the exact RawEvidence snapshot the runtime sees: template
    # top-K scores + CNN top-K logits (lossy by design -- a character the
    # CNN does not rank in its top-5 has logit -inf, exactly like the
    # production scorer's _raw_evidence).
    from dataclasses import replace as _replace
    from fixedfontocr.scorer import (
        _cnn_entries,
        _raw_evidence,
        _template_entries,
    )

    _glyphs = []
    _ci_map = []
    for ci, cand in enumerate(lattice.candidates):
        _cache = extractor._candidate_cache(cand, ctx, None)
        if _cache.glyph is not None:
            _glyphs.append(_cache.glyph)
            _ci_map.append(ci)
    if _glyphs:
        _g = np.stack(_glyphs, axis=0)
        tb = scorer.template.match_batch(_g, None) if scorer.template else None
        if model.weights is not None:
            cnn_batch = scorer._cnn_batch(_g, None)
        else:
            cnn_batch = None
        for j, ci in enumerate(_ci_map):
            cand = lattice.candidates[ci]
            tpl_entries = _template_entries(tb, j) if tb is not None else []
            cnn_entries = _cnn_entries(cnn_batch, j) if cnn_batch is not None else []
            cnn_top1 = (
                float(cnn_batch.top1[j]) if cnn_batch is not None else -np.inf
            )
            cnn_top2 = (
                float(cnn_batch.top2[j]) if cnn_batch is not None else -np.inf
            )
            re = _raw_evidence(
                cand.scores,
                tpl_entries,
                cnn_entries,
                cnn_top1,
                cnn_top2,
                tb,
                j,
                _g[j],
                cand.normalize_geometry,
            )
            if re is not None and cand.scores is not None:
                cand.scores = _replace(cand.scores, raw_evidence=re)

    # Per-(candidate, char) feature rows (schema v1, runtime-identical:
    # the extractor now reads the attached RawEvidence exactly like the
    # production decoder).
    cand_records: list[tuple] = []
    for ci, cand in enumerate(lattice.candidates):
        if cand.scores is None or not cand.scores.char_ids:
            continue
        ids = [int(c) for c in cand.scores.char_ids]
        cache = extractor._candidate_cache(cand, ctx, None)
        base_rows: list[list[float]] = []
        state_rows: list[list[float]] = []
        for cid in ids:
            ev = extractor._template_evidence(cand, cid)
            cache.template_evidence[int(cid)] = ev
            f = extractor.char_features(cand, cid, ctx, z=None)
            states = ev.get("states", {})
            state_vec = [states.get(k, 0.0) for k in STATE_KEYS]
            base_rows.append([float(v) for v in f])
            state_rows.append(state_vec)
        cand_records.append(
            (
                cand.atom_span,
                tuple(cand.components),
                tuple(ids),
                base_rows,
                state_rows,
            )
        )

    # Coverage: every GT char must be inside its candidate's fused Top-K.
    gt_cands_list = list(gt_cands)
    for pos, cand_idx in enumerate(gt_cands_list):
        cand = lattice.candidates[cand_idx]
        ids = [int(c) for c in cand.scores.char_ids] if cand.scores else []
        if gt_char_ids[pos] not in ids:
            coverage_failures.append(pos)

    state_candidates = ctx.state_candidates
    gt_z = _gt_z(rec)
    if not gt_z or gt_z == CLEAN_STATE and state_candidates:
        # Real crops: use the evidence-inferred dominant state.
        gt_z = state_candidates[0][0] if state_candidates else CLEAN_STATE
    rec_out = LineRecord(
        line_id=line_id,
        session=rec["session"],
        text=rec["text"],
        n_atoms=n_atoms,
        oracle=oracle,
        oracle_reason=reason,
        coverage_failures=tuple(coverage_failures),
        gt_candidates=gt_cands_list,
        gt_char_ids=gt_char_ids,
        candidates=tuple(cand_records),
        gt_z=gt_z,
        state_candidates=state_candidates,
        category=rec.get("category", ""),
    )
    return rec_out


def _gt_z(rec: dict) -> str:
    if rec.get("mode") == "clean":
        return CLEAN_STATE
    size = int(rec.get("size", 0))
    mode = rec.get("mode", "")
    key = f"{size}_{0 if mode == 'bilinear' else 1}"
    return key if key in STATE_KEYS else CLEAN_STATE


# ---------------------------------------------------------------------------
# Stage 1: local ranker
# ---------------------------------------------------------------------------


def _stage1_rows(records: list[LineRecord], model, extractor):
    """Build (visual features, label) rows for pairwise training.

    Returns X (normalized later), y (1 = correct char), line_idx, and the
    mean/std computed over the training rows.
    """

    rows: list[np.ndarray] = []
    labels: list[int] = []
    for rec in records:
        if not rec.gt_candidates:
            continue
        for pos, cand_idx in enumerate(rec.gt_candidates):
            if pos >= len(rec.gt_char_ids) or pos >= len(rec.candidates):
                continue
            gt_cid = rec.gt_char_ids[pos]
            if gt_cid < 0:
                continue
            _span, _comps, ids, base_rows, _states = rec.candidates[cand_idx]
            ids = list(ids)
            if gt_cid not in ids:
                # Coverage failure: recorded in the report; the character
                # is not selectable at decode time, so no training row.
                continue
            k = ids.index(gt_cid)
            z = rec.gt_z
            zi = STATE_KEYS.index(z) if z in STATE_KEYS else STATE_KEYS.index(CLEAN_STATE)
            # Positive: the correct char under the line's z.
            f_pos = _apply_z(base_rows[k], rec.candidates[cand_idx][4][k], zi)
            rows.append(f_pos)
            labels.append(1)
            # Negatives: the other fused chars of the same candidate.
            for j, cid in enumerate(ids):
                if j == k:
                    continue
                f_neg = _apply_z(base_rows[j], rec.candidates[cand_idx][4][j], zi)
                rows.append(f_neg)
                labels.append(0)
    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return X, y


def _apply_z(base: list[float], states: list[float], zi: int) -> np.ndarray:
    """The 48-dim vector with the z-conditional features filled in."""

    f = np.asarray(base, dtype=np.float64).copy()
    z_score = states[zi] if states else 0.0
    f[11] = z_score
    f[12] = float(np.clip(z_score - f[0], -1.0, 0.0))
    f[13] = 1.0 if zi == STATE_KEYS.index(CLEAN_STATE) else 0.0
    return f


def train_local_ranker(
    X: np.ndarray,
    y: np.ndarray,
    visual_idx: tuple[int, ...],
    epochs: int = 6,
    lr: float = 0.5,
    l2: float = 1e-4,
    seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, dict]:
    """Pairwise logistic over (correct char, Top-K wrong chars).

    Loss: ``sum log(1 + exp(w . (f_neg - f_pos))) + l2/2 |w|^2``.
    Deterministic full-batch gradient descent in float64.
    """

    Xv = X[:, list(visual_idx)]
    mean = Xv.mean(axis=0)
    std = Xv.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Xn = (Xv - mean) / std
    Xn = np.clip(Xn, -4.0, 4.0)
    # Build pairwise differences: positive rows vs negative rows of the
    # same candidate. Rows are stored (pos, negs...) per candidate; we
    # reconstruct the pairing by the running candidate id.
    pos_mask = y > 0.5
    pos_idx = np.flatnonzero(pos_mask)
    neg_idx = np.flatnonzero(~pos_mask)
    # pair each positive with the negatives that follow it before the next
    # positive.
    pairs: list[tuple[int, int]] = []
    for pi, p in enumerate(pos_idx):
        nxt = pos_idx[pi + 1] if pi + 1 < len(pos_idx) else len(Xn)
        pairs.extend((p, n) for n in neg_idx if p < n < nxt)
    if not pairs:
        raise ValueError("no (positive, negative) pairs in training data")
    P = np.asarray(pairs, dtype=np.int64)
    d = Xn[P[:, 0]] - Xn[P[:, 1]]
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(Xn.shape[1]) * 0.01
    b = 0.0
    n_pairs = len(P)
    history: list[float] = []
    for epoch in range(epochs):
        # margin = s_pos - s_neg; the loss log(1+exp(-margin)) is
        # minimized when the positive scores above the negative.
        margin = d @ w + b
        margin = np.clip(margin, -30.0, 30.0)
        sig_neg = 1.0 / (1.0 + np.exp(margin))  # sigma(-margin)
        grad_w = -(sig_neg @ d) / n_pairs + l2 * w
        grad_b = -float(sig_neg.mean())
        lr_cur = lr * (0.8 ** epoch)
        w = w - lr_cur * grad_w
        b = b - lr_cur * grad_b
        loss = float(
            np.logaddexp(0.0, -margin).mean() + 0.5 * l2 * float(w @ w)
        )
        history.append(loss)
    # Accuracy of the implied ranking: for each candidate, is the correct
    # char's logit the max among its options?
    logits = Xn @ w + b
    correct = 0
    total = 0
    for pi, p in enumerate(pos_idx):
        nxt = pos_idx[pi + 1] if pi + 1 < len(pos_idx) else len(Xn)
        group = np.concatenate([[p], neg_idx[(neg_idx > p) & (neg_idx < nxt)]])
        if logits[p] >= np.max(logits[group]):
            correct += 1
        total += 1
    info = {
        "epochs": epochs,
        "n_pairs": n_pairs,
        "local_top1": correct / max(total, 1),
        "loss_history": history,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    return w, b, mean, std, info


# ---------------------------------------------------------------------------
# Stage 2: structured path training
# ---------------------------------------------------------------------------


class HarvestReplay:
    """Replays the v2 path scoring from harvested per-(cand, char) data.

    Mirrors ``ProbabilisticDecoder`` exactly (same log-softmax with the
    background option, same k-best key, same length normalization and
    state marginalization); the training script asserts the replay
    reproduces the runtime decoder on sample lines.
    """

    def __init__(self, config, charset_len: int):
        self.cfg = config
        self.n_charset = charset_len
        self.visual_idx = list(VISUAL_FEATURE_IDX)
        self.geo_idx = list(GEOMETRY_FEATURE_IDX)
        self.bnd_idx = list(BOUNDARY_FEATURE_IDX)
        self.mean = np.asarray(config["feature_normalization"]["mean"])
        self.std = np.asarray(config["feature_normalization"]["std"])
        self.w = np.asarray(config["local_ranker"]["coef"])
        self.b = float(config["local_ranker"]["intercept"])
        self.wg = np.asarray(config["geometry_weights"]["coef"])
        self.wb = np.asarray(config["boundary_weights"]["coef"])
        self.T = float(config["temperature"])
        self.bg = float(config.get("background_logit", 0.0))
        self.alpha = float(config["length_alpha"])
        self.beta = float(config["length_beta"])
        self.k = int(config.get("k_best", 8))
        self.top_m = int((config.get("render_states") or {}).get("top_m", 3))

    def states(self, rec: LineRecord) -> list[str]:
        """Deterministic state list (mirrors RenderStateModel)."""

        keys = [k for k, _s in rec.state_candidates if k in STATE_KEYS]
        out: list[str] = []
        for k in keys:
            if k not in out:
                out.append(k)
            if len(out) >= self.top_m:
                break
        if CLEAN_STATE not in out:
            out.append(CLEAN_STATE)
        return out

    def _normalize(self, f: np.ndarray) -> np.ndarray:
        fn = (f - self.mean) / np.where(self.std > 1e-12, self.std, 1.0)
        return np.clip(fn, -4.0, 4.0)

    def line_options(self, rec: LineRecord, z: str):
        """Per-candidate option tables under state ``z``.

        Returns ``(opts, bnd)``: ``opts[ci]`` = list of
        ``(char_id, logp_visual, logp_geometry, raw_logit, lse, p_bg)``
        and ``bnd[ci]`` = boundary log-evidence.
        """

        zi = (
            STATE_KEYS.index(z)
            if z in STATE_KEYS
            else STATE_KEYS.index(CLEAN_STATE)
        )
        opts: dict[int, list[tuple]] = {}
        bnd: dict[int, float] = {}
        for ci, (span, comps, ids, base_rows, state_rows) in enumerate(
            rec.candidates
        ):
            f0 = np.zeros(N_FEATURES, dtype=np.float64)
            f0[33:46] = base_rows[0][33:46]
            fn0 = self._normalize(f0)
            bnd[ci] = float(
                np.clip(self.wb @ fn0[self.bnd_idx], -6.0, 0.0)
            )
            entries: list[tuple] = []
            for j, cid in enumerate(ids):
                f = _apply_z(base_rows[j], state_rows[j], zi)
                fn = self._normalize(f)
                logit = float(self.w @ fn[self.visual_idx] + self.b)
                geo = float(np.clip(self.wg @ fn[self.geo_idx], -6.0, 0.0))
                entries.append((cid, logit, geo))
            if entries:
                lse = logsumexp(
                    np.asarray([e[1] / self.T for e in entries] + [self.bg])
                )
                out_entries = []
                for (cid, logit, geo) in entries:
                    logp = float(logit / self.T - lse)
                    out_entries.append(
                        (cid, logp, geo, logit, float(lse),
                         float(math.exp(self.bg - lse)))
                    )
                opts[ci] = out_entries
        return opts, bnd

    def kbest(self, rec: LineRecord, z: str, domain_logp=None):
        """Deterministic k-best paths (same key as the runtime decoder).

        Returns a list of ``(score, cand_idxs, char_ids, n_anom, span)``.
        """

        opts, bnd = self.line_options(rec, z)
        by_end: dict[int, list[tuple[int, list, float]]] = {}
        for ci, (span, comps, ids, _b, _s) in enumerate(rec.candidates):
            by_end.setdefault(span[1], []).append(
                (ci, opts.get(ci, []), bnd.get(ci, 0.0))
            )
        domain = domain_logp if domain_logp is not None else (
            -math.log(max(self.n_charset, 1))
        )

        def path_score(opt, b: float) -> float:
            cid, logp, geo, _raw, _lse, _pbg = opt
            return logp + geo + b + domain

        def partial_key(e) -> tuple:
            n = len(e[1])
            norm = e[0] / ((n + self.beta) ** self.alpha)
            return (-norm, e[3], e[4], _entry_text(rec, e[2]))

        dp: dict[int, list] = {0: [(0.0, (), (), 0, 0)]}
        n_atoms = rec.n_atoms
        for end in range(1, n_atoms + 1):
            merged: list = []
            for ci, entries, b in by_end.get(end, ()):
                span = rec.candidates[ci][0]
                start = span[0]
                if start not in dp:
                    continue
                anom = (
                    1
                    if len(rec.candidates[ci][1]) > 1
                    or span[1] - span[0] > 1
                    else 0
                )
                for opt in entries:
                    score = path_score(opt, b)
                    for prev in dp[start]:
                        merged.append(
                            (
                                prev[0] + score,
                                prev[1] + (ci,),
                                prev[2] + (opt[0],),
                                prev[3] + anom,
                                prev[4] + (span[1] - span[0]),
                            )
                        )
            dp[end] = self._topk(merged, partial_key)
        return dp.get(n_atoms, [])

    @staticmethod
    def _topk(entries, key):
        entries = sorted(entries, key=key)
        out = []
        seen = set()
        for e in entries:
            ident = (e[1], e[2])
            if ident in seen:
                continue
            seen.add(ident)
            out.append(e)
            if len(out) >= 8:
                break
        return out

    def decode_texts(self, rec: LineRecord, domain_logp=None):
        """All k-best texts across states, ranked (deterministic)."""

        merged: list = []
        seen: set = set()
        for z in self.states(rec):
            for e in self.kbest(rec, z, domain_logp):
                if (e[1], e[2]) in seen:
                    continue
                seen.add((e[1], e[2]))
                merged.append(e)
        return merged

    def path_score_of(self, rec, cand_idxs, char_ids, domain_logp=None):
        """Marginalized score of one specific path."""

        scores: list[float] = []
        for z in self.states(rec):
            for e in self.kbest(rec, z, domain_logp):
                if e[1] == cand_idxs and e[2] == char_ids:
                    scores.append(e[0])
        if not scores:
            return None
        return float(logsumexp(np.asarray(scores)))

    def path_grads(self, rec, cand_idxs, char_ids, domain_logp=None):
        """Exact per-path gradient pieces.

        Returns ``(sum_geo, sum_bnd, d_logT, d_bg)`` of the path score:
        the score is linear in (wg, wb) with the summed feature vectors,
        and its log-T / background derivatives follow from the option
        tables.
        """

        sum_geo = 0.0
        sum_bnd = 0.0
        d_logT = 0.0
        d_bg = 0.0
        for z in self.states(rec):
            opts, bnd = self.line_options(rec, z)
            for e in self.kbest(rec, z, domain_logp):
                if e[1] != cand_idxs or e[2] != char_ids:
                    continue
                for ci, cid in zip(e[1], e[2]):
                    entries = opts.get(ci, [])
                    if not entries:
                        continue
                    geo = 0.0
                    for (ccid, _logp, g, raw, lse, pbg) in entries:
                        if ccid == cid:
                            geo = g
                            # d logp / d logT = (E_p[raw] - raw) / T
                            exp_raw = [
                                math.exp(r / self.T - lse)
                                for (_c, _p, _g, r, _l, _pb) in entries
                            ]
                            # softmax weights including the background
                            total = sum(exp_raw) + math.exp(self.bg - lse)
                            e_raw = sum(
                                r * er / total
                                for (_c, _p, _g, r, _l, _pb), er in zip(entries, exp_raw)
                            )
                            d_logT += (e_raw - raw) / self.T
                            d_bg += -math.exp(self.bg - lse) / total
                            sum_geo += geo
                            break
                sum_bnd += bnd.get(ci, 0.0)
        return (sum_geo, sum_bnd, d_logT, d_bg)


def _entry_text(rec: LineRecord, char_ids) -> str:
    from fixedfontocr.prob_decoder import UNKNOWN_CHAR

    return "".join(
        _CHARSET_CACHE[int(cid)] if 0 <= int(cid) < len(_CHARSET_CACHE) else UNKNOWN_CHAR
        for cid in char_ids
    )


_CHARSET_CACHE: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "prob")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs1", type=int, default=6)
    parser.add_argument("--epochs2", type=int, default=8)
    parser.add_argument("--n-words", type=int, default=140)
    parser.add_argument("--limit", type=int, default=0,
                        help="truncate the corpus (smoke runs)")
    parser.add_argument("--skip-harvest", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not (args.model / "config.json").exists():
        raise SystemExit(f"model {args.model} not built; run tools/train/build_model.py")
    model = load_model(args.model)
    charset = model.charset
    global _CHARSET_CACHE
    _CHARSET_CACHE = list(charset)

    # ------------------------------------------------------------------
    # 1. Corpus
    # ------------------------------------------------------------------
    manifest_path = ROOT / "tests" / "game_samples" / "manifest.json"
    samples_dir = ROOT / "tests" / "game_samples"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    real = _real_corpus(manifest, samples_dir)
    synth = _synthetic_corpus(ROOT / "charsets" / "words", args.n_words, rng)
    corpus = [dict(r, source="real") for r in real] + [
        dict(s, source="synth") for s in synth
    ]
    if args.limit:
        corpus = corpus[: args.limit]
    print(f"[corpus] {len(real)} real + {len(synth)} synthetic lines "
          f"(using {len(corpus)})")

    # ------------------------------------------------------------------
    # 2. Harvest
    # ------------------------------------------------------------------
    harvest_pkl = out / "harvest.pkl"
    records: list[LineRecord] = []
    if args.skip_harvest and harvest_pkl.exists():
        records = pickle.loads(harvest_pkl.read_bytes())
        print(f"[harvest] loaded {len(records)} records from cache")
    else:
        from fixedfontocr.frontend import extract_frontend  # noqa: F401
        from fixedfontocr.scorer import SegmentScorer
        from fixedfontocr.types import default_profile

        scorer = SegmentScorer(model)
        extractor = LocalFeatureExtractor(model, profile=default_profile())
        profile = default_profile()
        for i, rec in enumerate(corpus):
            oracle_boxes = None
            if rec["source"] == "synth" and rec["mode"] != "clean":
                try:
                    gt = render_ground_truth(
                        rec["text"],
                        resolve_font(FONT_PATH),
                        font_size=int(rec["size"]),
                        pad=8,
                    )
                    oracle_boxes = tuple(b for b in gt.boxes)
                except Exception:
                    oracle_boxes = None
            lr = harvest_line(
                i, rec, scorer, model, profile, extractor, oracle_boxes,
            )
            lr.image = None  # do not pickle images
            records.append(lr)
            if i % 100 == 0:
                print(f"[harvest] {i}/{len(corpus)} lines "
                      f"({time.time() - t0:.0f}s)")
        harvest_pkl.write_bytes(pickle.dumps(records))
        print(f"[harvest] done: {len(records)} records "
              f"({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------
    # 3. Grouped split
    # ------------------------------------------------------------------
    sessions = sorted({r.session for r in records})
    rng.shuffle(sessions)
    n = len(sessions)
    n_test = max(1, int(round(n * 0.2)))
    n_val = max(1, int(round(n * 0.15)))
    test_sessions = set(sessions[:n_test])
    val_sessions = set(sessions[n_test : n_test + n_val])
    for r in records:
        if r.session in test_sessions:
            r.split = "test"
        elif r.session in val_sessions:
            r.split = "val"
        else:
            r.split = "train"
    split_manifest = {
        "seed": args.seed,
        "n_sessions": n,
        "sessions": {
            "train": sorted(s for s in sessions if s not in test_sessions and s not in val_sessions),
            "val": sorted(val_sessions),
            "test": sorted(test_sessions),
        },
        "counts": {
            split: sum(1 for r in records if r.split == split)
            for split in ("train", "val", "test")
        },
        "chars_per_split": {
            split: sum(len(r.text) for r in records if r.split == split)
            for split in ("train", "val", "test")
        },
    }
    (out / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    oracle_ok = sum(1 for r in records if r.oracle)
    cov_fails = sum(len(r.coverage_failures) for r in records)
    print(
        f"[split] train/val/test sessions "
        f"{len(split_manifest['sessions']['train'])}/"
        f"{len(split_manifest['sessions']['val'])}/"
        f"{len(split_manifest['sessions']['test'])}; "
        f"oracle path recall {oracle_ok}/{len(records)}; "
        f"coverage failures {cov_fails}"
    )

    # ------------------------------------------------------------------
    # 4. Stage 1: local ranker
    # ------------------------------------------------------------------
    train_recs = [r for r in records if r.split == "train"]
    X, y = _stage1_rows(train_recs, model, None)
    w1, b1, mean_v, std_v, info1 = train_local_ranker(
        X, y, VISUAL_FEATURE_IDX, args.epochs1, seed=args.seed
    )
    # Full-vector mean/std: visual positions use the trained stats; the
    # geometry/boundary positions are standardized with their own stats
    # (computed over the same training rows for consistency).
    mean_full = np.zeros(N_FEATURES)
    std_full = np.ones(N_FEATURES)
    mean_full[list(VISUAL_FEATURE_IDX)] = mean_v
    std_full[list(VISUAL_FEATURE_IDX)] = std_v
    geo_rows = X[:, list(GEOMETRY_FEATURE_IDX)]
    bnd_rows = X[:, list(BOUNDARY_FEATURE_IDX)]
    mean_full[list(GEOMETRY_FEATURE_IDX)] = geo_rows.mean(axis=0)
    std_full[list(GEOMETRY_FEATURE_IDX)] = np.where(
        geo_rows.std(axis=0) > 1e-9, geo_rows.std(axis=0), 1.0
    )
    mean_full[list(BOUNDARY_FEATURE_IDX)] = bnd_rows.mean(axis=0)
    std_full[list(BOUNDARY_FEATURE_IDX)] = np.where(
        bnd_rows.std(axis=0) > 1e-9, bnd_rows.std(axis=0), 1.0
    )
    print(f"[stage1] local top-1 (train) {info1['local_top1']:.3f} "
          f"over {info1['n_pairs']} pairs")

    # ------------------------------------------------------------------
    # 5. Stage 2: structured path training + alpha/beta selection
    # ------------------------------------------------------------------

    def make_config(
        alpha: float, beta: float, wg, wb, T, bg, cal=None
    ) -> dict:
        coef = np.zeros(N_FEATURES)
        coef[list(VISUAL_FEATURE_IDX)] = w1
        return {
            "version": 2,
            "score_type": "log_probability",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "local_ranker": {
                "coef": [float(v) for v in w1],
                "intercept": float(b1),
            },
            "feature_normalization": {
                "mean": [float(v) for v in mean_full],
                "std": [float(v) for v in std_full],
                "clip": 4.0,
            },
            "geometry_weights": {"coef": [float(v) for v in wg]},
            "boundary_weights": {"coef": [float(v) for v in wb]},
            "temperature": float(T),
            "length_alpha": float(alpha),
            "length_beta": float(beta),
            "render_states": {
                "enabled": True,
                "top_m": 3,
                "state_prior": "uniform",
                "clean_fallback": True,
            },
            "confidence_calibration": [list(p) for p in (cal or [])],
            "reject": {"enabled": False, "confidence_threshold": 0.5, "margin_threshold": 0.0},
            "k_best": 8,
            "num_alternatives": 8,
            "unknown_log_prob": -8.0,
            "background_logit": float(bg),
            "max_state_approx": False,
            "domain_prior": {"type": "open_text"},
        }

    grid = [(0.5, 0.0), (0.75, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0)]
    best_ab: tuple[float, float] | None = None
    best_val: tuple[float, float] = (-1e9, 1e9)
    wg = np.full(9, -0.2)
    wb = np.zeros(13)
    T = 1.0
    bg = 0.0
    domain_logp = -math.log(max(len(charset), 1))
    for alpha, beta in grid:
        cfg = make_config(alpha, beta, wg, wb, T, bg)
        replay = HarvestReplay(cfg, len(charset))
        val_em = 0
        val_cer = 0.0
        val_n = 0
        for r in records:
            if r.split != "val" or not r.gt_candidates:
                continue
            best_text = _replay_best_text(replay, r, domain_logp)
            val_em += int(best_text == r.text)
            val_cer += sequence_cer(best_text, r.text)
            val_n += 1
        if val_n:
            em = val_em / val_n
            cer = val_cer / val_n
            print(f"[stage2] alpha={alpha} beta={beta}: val exact "
                  f"{em:.3f} CER {cer:.3f}")
            if (em, -cer) > best_val:
                best_val = (em, -cer)
                best_ab = (alpha, beta)
    assert best_ab is not None
    alpha, beta = best_ab
    print(f"[stage2] selected length normalization "
          f"alpha={alpha} beta={beta}")

    # Structured SGD over (wg, wb, logT, bg): pairwise logistic over the
    # GT path vs the k-best competing paths (deterministic replay).
    logT = 0.0
    lr = 0.1
    for epoch in range(args.epochs2):
        grad_wg = np.zeros(9)
        grad_wb = np.zeros(13)
        grad_logT = 0.0
        grad_bg = 0.0
        n_pairs = 0
        for r in records:
            if r.split != "train" or not r.gt_candidates:
                continue
            cfg = make_config(alpha, beta, wg, wb, math.exp(logT), bg)
            replay = HarvestReplay(cfg, len(charset))
            gt_cands = tuple(r.gt_candidates)
            gt_ids = tuple(int(c) for c in r.gt_char_ids)
            gt_score = replay.path_score_of(r, gt_cands, gt_ids, domain_logp)
            if gt_score is None:
                continue
            g_gt = replay.path_grads(r, gt_cands, gt_ids, domain_logp)
            for e in replay.decode_texts(r, domain_logp):
                if e[1] == gt_cands and e[2] == gt_ids:
                    continue
                neg_score = e[0]
                m = gt_score - neg_score
                m = float(np.clip(m, -30.0, 30.0))
                sig = 1.0 / (1.0 + np.exp(-m))
                g_neg = replay.path_grads(r, e[1], e[2], domain_logp)
                grad_wg += sig * (np.asarray(g_gt[0]) - np.asarray(g_neg[0]))
                grad_wb += sig * (np.asarray(g_gt[1]) - np.asarray(g_neg[1]))
                grad_logT += sig * (g_gt[2] - g_neg[2])
                grad_bg += sig * (g_gt[3] - g_neg[3])
                n_pairs += 1
        if n_pairs == 0:
            break
        grad_wg /= n_pairs
        grad_wb /= n_pairs
        grad_logT /= n_pairs
        grad_bg /= n_pairs
        wg = wg - lr * grad_wg
        wb = wb - lr * grad_wb
        logT = logT - lr * grad_logT
        bg = bg - lr * grad_bg
        lr *= 0.85
        print(f"[stage2] epoch {epoch}: {n_pairs} pairs, "
              f"wg[0]={wg[0]:+.3f} wb[0]={wb[0]:+.3f} "
              f"T={math.exp(logT):.2f} bg={bg:+.3f}")
    T = math.exp(logT)

    # ------------------------------------------------------------------
    # 6. Calibration (validation only) + reject thresholds
    # ------------------------------------------------------------------
    cfg = make_config(alpha, beta, wg, wb, T, bg)
    replay = HarvestReplay(cfg, len(charset))
    val_conf: list[float] = []
    val_correct: list[bool] = []
    for r in records:
        if r.split != "val" or not r.gt_candidates:
            continue
        c, ok = _replay_confidence(replay, r, domain_logp)
        val_conf.append(c)
        val_correct.append(ok)
    cal = []
    if len(val_conf) > 20:
        cal = _bin_calibration(
            np.asarray(val_conf), np.asarray(val_correct), bins=6
        )
    # Reject thresholds: confidence at which validation error <= 8%,
    # margin threshold = 0.5 * median runner-up margin of correct paths.
    margins = []
    for r in records:
        if r.split != "val" or not r.gt_candidates:
            continue
        m = _replay_margin(replay, r, domain_logp)
        if m is not None:
            margins.append(m)
    margin_thr = (
        0.5 * float(np.median(np.asarray(margins))) if margins else 0.0
    )
    conf_thr = 0.5
    if len(val_conf) > 20:
        arr = np.asarray(val_conf)
        ok = np.asarray(val_correct)
        for thr in np.linspace(0.3, 0.95, 27):
            sel = arr >= thr
            if sel.sum() >= 5 and (1.0 - ok[sel].mean()) <= 0.08:
                conf_thr = float(thr)
    print(f"[calibration] {len(cal)} confidence points; reject "
          f"conf<{conf_thr:.2f} margin<{margin_thr:.3f}")

    # Final config (with calibration) and final replay for the report.
    cfg = make_config(alpha, beta, wg, wb, T, bg, cal)
    replay = HarvestReplay(cfg, len(charset))

    # ------------------------------------------------------------------
    # 6b. Replay vs runtime equivalence (a few train lines)
    # ------------------------------------------------------------------
    _check_replay_equivalence(
        records, replay, domain_logp, model, args
    )

    # ------------------------------------------------------------------
    # 7. Export model/game_cn_prob
    # ------------------------------------------------------------------
    export = ROOT / "model" / "game_cn_prob"
    if export.exists():
        shutil.rmtree(export)
    export.mkdir(parents=True)
    for name in ("charset.txt", "weights.bin", "templates.bin", "geometry.json"):
        src = args.model / name
        if src.exists():
            shutil.copy2(src, export / name)
    full_config = json.loads(
        (args.model / "config.json").read_text(encoding="utf-8")
    )
    full_config["decoder"] = make_config(
        alpha, beta, wg, wb, T, bg, cal
    )
    full_config["decoder"]["reject"] = {
        "enabled": False,
        "confidence_threshold": conf_thr,
        "margin_threshold": margin_thr,
    }
    payload = json.dumps(full_config, indent=2, ensure_ascii=False) + "\n"
    (export / "config.json").write_text(payload, encoding="utf-8")
    (export / "model.json").write_text(payload, encoding="utf-8")
    (out / "decoder_block.json").write_text(
        json.dumps(full_config["decoder"], indent=2) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 8. Metrics report (replay on all splits)
    # ------------------------------------------------------------------
    report = _metrics_report(replay, records, charset, model)
    report["split_manifest"] = str(out / "split_manifest.json")
    report["length_normalization"] = {"alpha": alpha, "beta": beta}
    report["stage1"] = info1
    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[done] exported {export} in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# Replay helpers (using HarvestReplay)
# ---------------------------------------------------------------------------


def _replay_best_text(replay: HarvestReplay, rec: LineRecord, domain_logp):
    """Best path text over all states (deterministic)."""

    merged = replay.decode_texts(rec, domain_logp)
    if not merged:
        return ""
    return _entry_text(rec, merged[0][2])


def _replay_kbest_texts(replay: HarvestReplay, rec: LineRecord, domain_logp):
    return [
        (e[0], _entry_text(rec, e[2]))
        for e in replay.decode_texts(rec, domain_logp)
    ]


def _replay_path_score(replay: HarvestReplay, rec: LineRecord, domain_logp):
    gt_cands = tuple(rec.gt_candidates)
    gt_ids = tuple(int(c) for c in rec.gt_char_ids)
    return replay.path_score_of(rec, gt_cands, gt_ids, domain_logp)


def _replay_confidence(replay, rec, domain_logp):
    """(calibrated confidence, correctness) of the best path."""

    kbest = _replay_kbest_texts(replay, rec, domain_logp)
    if not kbest:
        return 0.0, False
    scores = np.asarray([s for s, _t in kbest])
    lse = logsumexp(scores)
    conf = float(math.exp(np.clip(scores[0] - lse, -30.0, 0.0)))
    cal = replay.cfg.get("confidence_calibration", [])
    if cal:
        conf = float(calibrate_monotone(np.asarray([conf]), cal)[0])
    return conf, kbest[0][1] == rec.text


def _replay_margin(replay, rec, domain_logp):
    kbest = _replay_kbest_texts(replay, rec, domain_logp)
    if len(kbest) < 2:
        return None
    return float(kbest[0][0] - kbest[1][0])


def _check_replay_equivalence(
    records, replay, domain_logp, model, args
) -> None:
    """Assert the harvested replay reproduces the runtime decoder."""

    from fixedfontocr.prob_decoder import ProbabilisticDecoder
    from fixedfontocr.scorer import SegmentScorer
    from fixedfontocr.types import default_profile, profile_from_dict
    from fixedfontocr.frontend import extract_frontend
    from fixedfontocr.preprocess import find_lines
    from fixedfontocr.segmentation import segment_line

    sample = [r for r in records if r.split == "train" and r.gt_candidates][:3]
    if not sample:
        return
    model_dir = args.model
    tmp_cfg = dict(model.config)
    tmp_cfg["decoder"] = dict(replay.cfg)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy2(model_dir / "charset.txt", td / "charset.txt")
        for name in ("weights.bin", "templates.bin", "geometry.json"):
            src = model_dir / name
            if src.exists():
                shutil.copy2(src, td / name)
        (td / "config.json").write_text(
            json.dumps(tmp_cfg, indent=2) + "\n", encoding="utf-8"
        )
        (td / "model.json").write_text(
            json.dumps(tmp_cfg, indent=2) + "\n", encoding="utf-8"
        )
        dec, warn = ProbabilisticDecoder.try_build(load_model(td))
        assert dec is not None, warn
        scorer = SegmentScorer(load_model(td))
        from fixedfontocr.domain_prior import OpenTextPrior

        domain = OpenTextPrior(tuple(model.charset))
        for r in sample:
            if r.category.startswith("real-game"):
                # reload the real image + its manifest profile (the same
                # inputs the harvest used).
                import json as _json

                m = _json.loads(
                    (ROOT / "tests" / "game_samples" / "manifest.json")
                    .read_text(encoding="utf-8")
                )
                img_path = None
                prof_cfg = {}
                for s_ in m["samples"]:
                    if s_["expected"] == r.text and s_["category"] == r.category:
                        img_path = ROOT / "tests" / "game_samples" / s_["file"]
                        base = dict(m.get("default_profile", {}))
                        base.update(s_.get("profile", {}))
                        prof_cfg = base
                        break
                if img_path is None or not img_path.is_file():
                    continue
                from PIL import Image as _PILImage

                image = np.asarray(
                    _PILImage.open(img_path).convert("RGB"), dtype=np.uint8
                )
                prof = profile_from_dict(prof_cfg) if prof_cfg else default_profile()
            else:
                size = int(getattr(r, "_size", 16)) or 16
                image = _render_line(r.text, size, "bilinear")
                prof = default_profile()
            frontend = extract_frontend(image, prof)
            lines = find_lines(frontend.binary_mask, prof)
            if len(lines) != 1:
                continue
            path = segment_line(lines[0], prof, scorer, prob_decoder=dec,
                                prob_domain=domain)
            runtime_text = path.text
            replay_text = _replay_best_text(replay, r, domain_logp)
            if runtime_text != replay_text:
                # Diagnostics for the mismatch: compare the option tables.
                merged = replay.decode_texts(r, domain_logp)
                rt = []
                for e in merged[:4]:
                    rt.append((_entry_text(r, e[2]), round(float(e[0]), 3)))
                diag = path.prob_diagnostics or {}
                raise AssertionError(
                    f"replay/runtime mismatch for {r.text!r}: "
                    f"replay={replay_text!r} runtime={runtime_text!r}\n"
                    f"  replay kbest: {rt}\n"
                    f"  runtime states: {diag.get('states')} "
                    f"argmax={diag.get('line_render_state')}\n"
                    f"  runtime per-char: {diag.get('per_char')}"
                )
    print("[check] replay reproduces the runtime decoder on sample lines")


def _bin_calibration(
    conf: np.ndarray, correct: np.ndarray, bins: int
) -> list[list[float]]:
    """Monotone piecewise-linear calibration (validation-only fit)."""

    order = np.argsort(conf, kind="stable")
    cs = conf[order]
    ok = correct[order].astype(np.float64)
    n = len(cs)
    points: list[tuple[float, float]] = [(0.0, 0.0)]
    edges = np.linspace(0, n, bins + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        block = slice(a, b)
        x = float(np.mean(cs[block]))
        p = (float(ok[block].sum()) + 1.0) / ((b - a) + 2.0)
        points.append((x, p))
    points.append((1.0, 1.0))
    mono: list[tuple[float, float]] = []
    for x, y in points:
        if not mono or y >= mono[-1][1]:
            mono.append((x, y))
        else:
            mono.append((x, mono[-1][1]))
    return [[float(x), float(y)] for x, y in mono]


def _metrics_report(replay, records, charset, model) -> dict:
    """Per-split metrics on the harvested lines (decoder-level)."""

    domain_logp = -math.log(max(len(charset), 1))
    splits = ("train", "val", "test")
    out: dict = {}
    for split in splits:
        rs = [r for r in records if r.split == split]
        exact = 0
        cer = 0.0
        confs: list[float] = []
        corrects: list[bool] = []
        nlls: list[float] = []
        local_top1 = 0
        local_n = 0
        coverage = 0
        oracle_ok = 0
        n = 0
        for r in rs:
            if not r.gt_candidates:
                continue
            n += 1
            oracle_ok += int(r.oracle)
            coverage += int(len(r.coverage_failures) == 0)
            kbest = _replay_kbest_texts(replay, r, domain_logp)
            if kbest:
                best_text = kbest[0][1]
            else:
                best_text = ""
            exact += int(best_text == r.text)
            cer += sequence_cer(best_text, r.text)
            c, ok = _replay_confidence(replay, r, domain_logp)
            confs.append(c)
            corrects.append(ok)
            # NLL of the GT path under the k-best distribution.
            scores = np.asarray([s for s, _t in kbest]) if kbest else np.asarray([-1e9])
            lse = logsumexp(scores)
            gt_score = _replay_path_score(replay, r, domain_logp)
            if gt_score is not None:
                nlls.append(float(-(gt_score - lse)))
            else:
                nlls.append(30.0)
            # local top-1 over the GT positions
            for pos, ci in enumerate(r.gt_candidates):
                if pos >= len(r.gt_char_ids) or ci >= len(r.candidates):
                    continue
                gt_cid = r.gt_char_ids[pos]
                ids = r.candidates[ci][2]
                if gt_cid in ids:
                    local_n += 1
                    local_top1 += int(ids[0] == gt_cid)
        if n == 0:
            out[split] = {"n_lines": 0}
            continue
        cm = calibration_metrics(
            np.asarray(confs), np.asarray(corrects),
            nll=float(np.mean(nlls)) if nlls else float("nan"),
        )
        out[split] = {
            "n_lines": n,
            "oracle_path_recall": oracle_ok / n,
            "gt_topk_coverage": coverage / n,
            "local_top1": (local_top1 / local_n) if local_n else float("nan"),
            "exact_match": exact / n,
            "cer": cer / n,
            "nll": cm.nll,
            "brier": cm.brier,
            "ece": cm.ece,
            "reject_rate@0.5": float(
                np.mean(np.asarray(confs) < 0.5)
            ) if confs else float("nan"),
            "wrong_association": None,
        }
    return out


if __name__ == "__main__":
    # Pickle compatibility: classes defined in ``__main__`` are re-exported
    # under the module name so the harvest cache loads in later runs.
    sys.modules.setdefault("train_probabilistic_decoder", sys.modules["__main__"])
    main()
