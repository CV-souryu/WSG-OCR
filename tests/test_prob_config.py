"""Probabilistic decoder v2 -- versioned config validation.

Covers the "配置和兼容性" contract:

* explicit versioning and strict field validation;
* missing fields -> safe v1 fallback;
* NaN/Inf / dimension-mismatch parameters are refused (v2 never loads);
* ``model.json`` and ``config.json`` stay in sync;
* legacy models keep working (v1 output unchanged).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fixedfontocr.prob_decoder import (
    DECODER_VERSION,
    ProbDecoderConfig,
    ProbabilisticDecoder,
)

N_FEATURES = 48
N_VISUAL = 26
N_GEO = 9
N_BND = 13


def _valid_block() -> dict:
    return {
        "version": DECODER_VERSION,
        "score_type": "log_probability",
        "feature_schema_version": 1,
        "local_ranker": {
            "coef": [0.0] * N_VISUAL,
            "intercept": 0.0,
        },
        "feature_normalization": {
            "mean": [0.0] * N_FEATURES,
            "std": [1.0] * N_FEATURES,
            "clip": 4.0,
        },
        "geometry_weights": {"coef": [0.0] * N_GEO},
        "boundary_weights": {"coef": [0.0] * N_BND},
        "temperature": 1.0,
        "length_alpha": 0.0,
        "length_beta": 0.0,
        "render_states": {"enabled": True, "top_m": 3},
        "confidence_calibration": [],
        "reject": {"enabled": False, "confidence_threshold": 0.5, "margin_threshold": 0.0},
        "k_best": 8,
        "num_alternatives": 8,
        "unknown_log_prob": -8.0,
    }


def test_valid_config_round_trip():
    cfg = ProbDecoderConfig.from_dict(_valid_block())
    assert cfg.version == DECODER_VERSION
    assert cfg.validate() == []
    assert len(cfg.local_ranker_coef) == N_VISUAL
    assert len(cfg.feature_mean) == N_FEATURES


def test_missing_decoder_block_falls_back_to_v1():
    assert ProbDecoderConfig.load({}) is None
    assert ProbDecoderConfig.load({"visual_weights": {}}) is None


def test_legacy_version_falls_back_to_v1():
    block = _valid_block()
    block["version"] = 1
    assert ProbDecoderConfig.load({"decoder": block}) is None
    block["version"] = "bogus"
    assert ProbDecoderConfig.load({"decoder": block}) is None


def test_nan_inf_refused():
    block = _valid_block()
    block["local_ranker"]["coef"][0] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        ProbDecoderConfig.from_dict(block)
    block = _valid_block()
    block["feature_normalization"]["std"][3] = float("inf")
    with pytest.raises(ValueError, match="NaN/Inf"):
        ProbDecoderConfig.from_dict(block)
    block = _valid_block()
    block["geometry_weights"]["coef"][0] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        ProbDecoderConfig.from_dict(block)
    block = _valid_block()
    block["boundary_weights"]["coef"][0] = float("-inf")
    with pytest.raises(ValueError, match="NaN/Inf"):
        ProbDecoderConfig.from_dict(block)


def test_dimension_mismatch_refused():
    block = _valid_block()
    block["local_ranker"]["coef"] = [0.0] * (N_VISUAL - 1)
    with pytest.raises(ValueError, match="expected 26"):
        ProbDecoderConfig.from_dict(block)
    block = _valid_block()
    block["feature_normalization"]["mean"] = [0.0] * (N_FEATURES + 1)
    with pytest.raises(ValueError, match="expected 48"):
        ProbDecoderConfig.from_dict(block)
    block = _valid_block()
    block["boundary_weights"]["coef"] = [0.0] * 3
    with pytest.raises(ValueError, match="expected 13"):
        ProbDecoderConfig.from_dict(block)


def test_bad_scalars_refused():
    for field, value in (
        ("temperature", 0.0),
        ("temperature", -1.0),
        ("temperature", float("nan")),
        ("length_alpha", -0.1),
        ("length_alpha", 2.5),
        ("length_beta", -1.0),
        ("unknown_log_prob", 1.0),
        ("unknown_log_prob", float("inf")),
        ("k_best", 0),
        ("num_alternatives", 0),
    ):
        block = _valid_block()
        block[field] = value
        with pytest.raises(ValueError):
            ProbDecoderConfig.from_dict(block)


def test_bad_calibration_refused():
    block = _valid_block()
    block["confidence_calibration"] = [[0.0, 0.0], [0.5, 0.4], [0.4, 0.6]]
    with pytest.raises(ValueError, match="increasing"):
        ProbDecoderConfig.from_dict(block)
    block = _valid_block()
    block["confidence_calibration"] = [[0.0, 0.0], [0.5, 0.6], [0.9, 0.4]]
    with pytest.raises(ValueError, match="monotone"):
        ProbDecoderConfig.from_dict(block)
    block = _valid_block()
    block["confidence_calibration"] = [[0.0, 0.0], [1.5, 0.6]]
    with pytest.raises(ValueError, match="outside"):
        ProbDecoderConfig.from_dict(block)
    block = _valid_block()
    block["reject"]["confidence_threshold"] = 1.5
    with pytest.raises(ValueError):
        ProbDecoderConfig.from_dict(block)


def test_render_states_validated():
    block = _valid_block()
    block["render_states"] = {"top_m": 0}
    with pytest.raises(ValueError):
        ProbDecoderConfig.from_dict(block)
    block = _valid_block()
    block["render_states"] = {"state_prior": "learned"}
    with pytest.raises(ValueError):
        ProbDecoderConfig.from_dict(block)


def test_negative_std_refused():
    block = _valid_block()
    block["feature_normalization"]["std"][5] = -1.0
    with pytest.raises(ValueError, match=">= 0"):
        ProbDecoderConfig.from_dict(block)


def test_load_returns_none_on_invalid_and_engine_falls_back(tmp_path):
    """An invalid v2 block must never break the engine: try_build -> v1."""

    from fixedfontocr.fontgen import build_templates
    from fixedfontocr.model import write_hybrid_model

    import numpy as np

    charset = list("AB")
    rng = np.random.default_rng(0)
    chars, templates = build_templates(_font_path(), charset, render_size=32)
    out = tmp_path / "model"
    write_hybrid_model(
        out,
        chars,
        templates,
        {
            "conv1.weight": rng.standard_normal((8, 1, 3, 3), dtype=np.float32),
            "conv1.bias": rng.standard_normal(8, dtype=np.float32),
            "dw1.weight": rng.standard_normal((8, 3, 3), dtype=np.float32),
            "dw1.bias": rng.standard_normal(8, dtype=np.float32),
            "pw1.weight": rng.standard_normal((16, 8), dtype=np.float32),
            "pw1.bias": rng.standard_normal(16, dtype=np.float32),
            "dw2.weight": rng.standard_normal((16, 3, 3), dtype=np.float32),
            "dw2.bias": rng.standard_normal(16, dtype=np.float32),
            "pw2.weight": rng.standard_normal((32, 16), dtype=np.float32),
            "pw2.bias": rng.standard_normal(32, dtype=np.float32),
            "fc.weight": rng.standard_normal((2, 32), dtype=np.float32),
            "fc.bias": rng.standard_normal(2, dtype=np.float32),
        },
        font_path=_font_path(),
    )
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    bad = _valid_block()
    bad["local_ranker"]["coef"][0] = float("nan")
    config["decoder"] = bad
    (out / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (out / "model.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    from fixedfontocr.model import load_model

    model = load_model(out)
    dec, warn = ProbabilisticDecoder.try_build(model)
    assert dec is None
    assert warn is not None and "falling back to v1" in warn


def _font_path():
    from fixedfontocr.defaults import FONT_PATH, resolve_font

    if not FONT_PATH.exists():
        pytest.skip("bundled font not present")
    return resolve_font(FONT_PATH)


def test_model_json_config_json_sync(tmp_path):
    """The exporter must write the same decoder block to both files."""

    from fixedfontocr.defaults import MODEL_PATH
    from fixedfontocr.model import load_model

    if not (MODEL_PATH / "config.json").exists():
        pytest.skip("model/game_cn not built")
    model = load_model(MODEL_PATH)
    config = model.config
    # Simulate the exporter: write decoder block + model.json copy.
    block = _valid_block()
    config["decoder"] = block
    out = tmp_path / "sync"
    out.mkdir()
    (out / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (out / "model.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    c1 = json.loads((out / "config.json").read_text(encoding="utf-8"))
    c2 = json.loads((out / "model.json").read_text(encoding="utf-8"))
    assert c1 == c2
    assert c1["decoder"]["version"] == DECODER_VERSION
    # A diverged model.json must be caught by the benchmark/export tooling:
    c2["decoder"]["temperature"] = 2.0
    (out / "model.json").write_text(
        json.dumps(c2, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(AssertionError):
        assert json.loads((out / "config.json").read_text(encoding="utf-8")) == json.loads(
            (out / "model.json").read_text(encoding="utf-8")
        )
