"""Shared helpers for probabilistic-decoder (v2) tests.

Two test vehicles:

* ``v2_game_model`` -- a *derived* copy of the bundled hybrid model with a
  hand-crafted v2 decoder block (template-driven ranker), so the full
  pipeline (raw evidence attachment -> features -> v2 decode) can be
  exercised end to end;
* manual lattice/component helpers for the unit-level path tests.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from fixedfontocr.model import load_model

N_FEATURES = 48
N_VISUAL = 26
N_GEO = 9
N_BND = 13


def decoder_block(rank="template") -> dict:
    """A valid, hand-crafted v2 decoder block.

    ``rank="template"`` drives the local ranker by the raw template score
    (plus a weak CNN term and geometry/boundary priors), which makes the
    v2 decoder behave like a template-first recognizer -- good enough to
    pin the *behavioral* contracts without a training run.
    """

    coef = [0.0] * N_VISUAL
    if rank == "template":
        # Balanced template-first ranker (the trained ranker learns the
        # same trade-off from data; this hand-crafted version only pins
        # the behavioral contracts). The legacy fused score is
        # deliberately NOT used here: v2 must not be a re-fit of v1's
        # fusion.
        coef[0] = 8.0  # tpl_score
        coef[11] = 1.0  # tpl_state_score_z (line render state evidence)
        coef[12] = 0.5  # tpl_state_delta_z
        coef[14] = 0.3  # cnn_logit_c
        coef[18] = 0.5  # cnn_margin_c
        coef[19] = -0.4  # cnn_rank_c
        gcoef = [-0.4] * N_GEO
        bcoef = [0.0] * N_BND
        bcoef[0] = 0.0  # bnd_single (neutral: no fragmentation reward)
        bcoef[1] = -0.3  # bnd_merge
        bcoef[2] = -0.3  # bnd_split
        bcoef[5] = -0.6  # bnd_seam_ink
        bcoef[6] = -0.6  # bnd_seam_dev
    elif rank == "cnn":
        coef[14] = 1.0
        coef[19] = 0.5
        coef[20] = -0.5
        gcoef = [0.0] * N_GEO
        bcoef = [0.0] * N_BND
    elif rank == "cnn_flat":
        # Only the character's own CNN logit ranks the options: perfectly
        # tied logits give a perfectly tied local distribution.
        coef[14] = 1.0
        gcoef = [0.0] * N_GEO
        bcoef = [0.0] * N_BND
    else:
        raise ValueError(rank)
    return {
        "version": 2,
        "score_type": "log_probability",
        "feature_schema_version": 1,
        "local_ranker": {"coef": coef, "intercept": 0.0},
        "feature_normalization": {
            "mean": [0.0] * N_FEATURES,
            "std": [1.0] * N_FEATURES,
            "clip": 4.0,
        },
        "geometry_weights": {"coef": gcoef},
        "boundary_weights": {"coef": bcoef},
        "temperature": 1.0,
        "length_alpha": 1.0,
        "length_beta": 0.0,
        "render_states": {"enabled": True, "top_m": 3},
        "confidence_calibration": [],
        "reject": {"enabled": False, "confidence_threshold": 0.5, "margin_threshold": 0.0},
        "k_best": 8,
        "num_alternatives": 8,
        "unknown_log_prob": -8.0,
        "background_logit": 0.0,
    }


@pytest.fixture(scope="session")
def v2_game_model(tmp_path_factory) -> Path:
    """Derived copy of ``model/game_cn`` with a v2 decoder block."""

    from fixedfontocr.defaults import MODEL_PATH

    if not (MODEL_PATH / "config.json").exists():
        pytest.skip("model/game_cn not built; run tools/train/build_model.py")
    out = tmp_path_factory.mktemp("v2_game") / "game_cn_v2"
    out.mkdir(parents=True)
    for name in ("config.json", "model.json", "charset.txt", "weights.bin",
                 "templates.bin", "geometry.json"):
        src = MODEL_PATH / name
        if src.exists():
            shutil.copy2(src, out / name)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    config["decoder"] = decoder_block("template")
    payload = json.dumps(config, indent=2) + "\n"
    (out / "config.json").write_text(payload, encoding="utf-8")
    (out / "model.json").write_text(payload, encoding="utf-8")
    return out


def build_v2_cnn_digits_model(tmp_path: Path, rank: str) -> Path:
    """Derived copy of the tiny CNN-digits fixture with a v2 block."""

    src = Path(__file__).parent / "fixtures" / "cnn_digits"
    out = tmp_path / f"cnn_digits_v2_{rank}"
    out.mkdir(parents=True)
    for name in ("config.json", "charset.txt", "weights.bin"):
        shutil.copy2(src / name, out / name)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    config["decoder"] = decoder_block(rank)
    payload = json.dumps(config, indent=2) + "\n"
    (out / "config.json").write_text(payload, encoding="utf-8")
    (out / "model.json").write_text(payload, encoding="utf-8")
    return out


@pytest.fixture(scope="session")
def v2_cnn_digits_model(tmp_path_factory) -> Path:
    """Derived copy of the tiny CNN-digits fixture with a v2 block."""

    return build_v2_cnn_digits_model(
        tmp_path_factory.mktemp("v2_cnn"), "cnn"
    )


def manual_raw_evidence(char_ids, cnn_logits, template_scores=None):
    """A RawEvidence with hand-set per-char raw scores (no glyph)."""

    from fixedfontocr.types import RawEvidence

    cnn_logits = [float(x) for x in cnn_logits]
    return RawEvidence(
        char_ids=tuple(int(c) for c in char_ids),
        template_scores=tuple(
            float(x) for x in (template_scores or [0.0] * len(char_ids))
        ),
        cnn_logits=tuple(cnn_logits),
        cnn_top1_logit=max(cnn_logits),
        cnn_top2_logit=(
            sorted(cnn_logits)[-2] if len(cnn_logits) > 1 else -np.inf
        ),
        template_winner=(-1, -1, -1, -1, -1),
        glyph=None,
    )


def manual_candidate(
    start: int,
    end: int,
    char_ids,
    cnn_logits,
    template_scores=None,
    segment=None,
    components=None,
):
    """One VisualCandidate with VisualScores + RawEvidence (manual lattice)."""

    from fixedfontocr.types import VisualCandidate, VisualScores

    logits = [float(x) for x in cnn_logits]
    scores = VisualScores(
        char_ids=tuple(int(c) for c in char_ids),
        logits=tuple(logits),
        raw_score=float(max(logits)),
        score_type="cnn",
        raw_evidence=manual_raw_evidence(char_ids, logits, template_scores),
    )
    return VisualCandidate(
        start=start,
        end=end,
        components=tuple(components or (start,)),
        atoms=(start, end),
        segment=segment,
        scores=scores,
    )


def manual_mask(w: int, h: int, ink_cols: set[int] | None = None) -> np.ndarray:
    """A simple binary mask (default: full ink)."""

    mask = np.zeros((h, w), dtype=bool)
    if ink_cols is None:
        mask[:, :] = True
    else:
        for c in ink_cols:
            if 0 <= c < w:
                mask[:, c] = True
    return mask
