"""Probabilistic decoder v2 -- probability primitives.

Covers the Goal "概率基础测试" checklist:

* log-softmax normalization;
* temperature monotonicity;
* calibration monotonicity;
* no NaN/Inf on extreme logits;
* extreme-logit stability (``1e30``/``-1e30``, all-masked rows).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fixedfontocr.prob_math import (
    calibrate_log_monotone,
    calibrate_monotone,
    calibration_metrics,
    log_softmax,
    logsumexp,
    sequence_cer,
    temperature_scale,
)

BIG = 1e30


def test_logsumexp_basic_and_stable():
    assert logsumexp(np.array([0.0, 0.0])) == pytest.approx(math.log(2))
    assert logsumexp(np.array([1000.0, 1000.0])) == pytest.approx(1000 + math.log(2))
    assert logsumexp(np.array([BIG, BIG])) == pytest.approx(BIG + math.log(2))
    assert logsumexp(np.array([-BIG, -BIG])) == pytest.approx(-BIG + math.log(2))
    assert logsumexp(np.array([1.0, -np.inf])) == pytest.approx(1.0)


def test_logsumexp_all_masked_is_minus_inf_not_nan():
    out = logsumexp(np.array([-np.inf, -np.inf]))
    assert out == -np.inf
    out2 = logsumexp(np.array([[1.0, 2.0], [-np.inf, -np.inf]]), axis=1)
    assert out2[0] == pytest.approx(2.0 + math.log(1 + math.exp(-1)))
    assert out2[1] == -np.inf
    assert not np.any(np.isnan(out2))


def test_log_softmax_normalizes():
    logits = np.array([[2.0, 1.0, 0.5], [0.0, 0.0, 0.0], [-5.0, 5.0, 0.0]])
    ls = log_softmax(logits)
    probs = np.exp(ls)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(np.log(probs), ls)
    assert not np.any(np.isnan(ls)) and not np.any(np.isposinf(ls))


def test_log_softmax_extreme_logits_stable():
    logits = np.array([[BIG, BIG, -BIG], [-BIG, -BIG, -BIG], [1e-30, 1e-30, 1e-30]])
    ls = log_softmax(logits)
    probs = np.exp(ls)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[0], [0.5, 0.5, 0.0], atol=1e-12)
    # All-masked rows fall back to the uniform distribution, not NaN.
    assert np.allclose(probs[1], [1 / 3] * 3)
    assert not np.any(np.isnan(ls)) and not np.any(np.isposinf(ls))


def test_temperature_scaling_is_monotone():
    logits = np.array([2.0, 1.0, 0.5])

    def probs(t):
        p = np.exp(log_softmax(logits / t))
        return p

    p_low = probs(0.5)
    p_high = probs(2.0)
    # Lower temperature sharpens: top-1 prob up, tail prob down.
    assert p_low[0] > p_high[0]
    assert p_low[-1] < p_high[-1]
    # Mid temperature sits between.
    p_mid = probs(1.0)
    assert p_low[0] > p_mid[0] > p_high[0]


def test_temperature_rejects_non_positive():
    with pytest.raises(ValueError):
        temperature_scale(np.array([1.0]), 0.0)
    with pytest.raises(ValueError):
        temperature_scale(np.array([1.0]), -1.0)
    with pytest.raises(ValueError):
        temperature_scale(np.array([1.0]), float("nan"))
    with pytest.raises(ValueError):
        temperature_scale(np.array([1.0]), float("inf"))


def test_calibration_is_monotone_and_bounded():
    cal = [(0.0, 0.0), (0.3, 0.2), (0.7, 0.8), (1.0, 1.0)]
    x = np.linspace(-0.5, 1.5, 101)
    y = calibrate_monotone(x, cal)
    assert np.all(np.diff(y) >= -1e-12)
    assert np.all(y >= 0.0) and np.all(y <= 1.0)
    # Clamping at the edges.
    assert y[0] == pytest.approx(0.0)
    assert y[-1] == pytest.approx(1.0)


def test_calibration_rejects_non_monotone_points():
    with pytest.raises(ValueError):
        calibrate_monotone(np.array([0.5]), [(0.0, 0.0), (0.4, 0.5), (0.8, 0.3)])
    with pytest.raises(ValueError):
        calibrate_monotone(np.array([0.5]), [(0.0, 0.0), (0.4, 0.5), (0.4, 0.6)])


def test_log_calibration_monotone():
    cal = [(-10.0, -8.0), (-2.0, -4.0), (0.0, -0.5)]
    x = np.linspace(-12.0, 2.0, 51)
    y = calibrate_log_monotone(x, cal)
    assert np.all(np.diff(y) >= -1e-12)
    assert not np.any(np.isnan(y))


def test_calibration_metrics_sane():
    conf = np.array([0.9, 0.2, 0.7, 0.55, 0.95])
    ok = np.array([True, False, True, True, True])
    m = calibration_metrics(conf, ok, nll=-0.4)
    assert m.n == 5
    assert 0.0 <= m.brier <= 1.0
    assert 0.0 <= m.ece <= 1.0
    assert m.accuracy == pytest.approx(0.8)
    assert m.nll == pytest.approx(-0.4)
    # Perfect prediction -> zero Brier/ECE.
    m2 = calibration_metrics(np.array([1.0, 0.0]), np.array([True, False]))
    assert m2.brier == pytest.approx(0.0)
    assert m2.ece == pytest.approx(0.0)


def test_sequence_cer():
    assert sequence_cer("abc", "abc") == 0.0
    assert sequence_cer("abc", "abd") == pytest.approx(1 / 3)
    assert sequence_cer("", "abc") == 1.0
    assert sequence_cer("abc", "") == 1.0
    assert sequence_cer("", "") == 0.0
    assert sequence_cer("ab", "a") == pytest.approx(0.5)


def test_no_nan_inf_through_full_chain():
    logits = np.array([[BIG, -BIG], [-BIG, BIG], [0.0, 0.0]], dtype=np.float64)
    ls = log_softmax(temperature_scale(logits, 1.0))
    cal = calibrate_monotone(np.exp(ls[:, 0]), [(0.0, 0.0), (0.5, 0.4), (1.0, 1.0)])
    assert not np.any(np.isnan(ls))
    assert not np.any(np.isnan(cal))
    assert np.all(cal >= 0.0) and np.all(cal <= 1.0)
