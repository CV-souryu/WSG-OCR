"""Deterministic NumPy probability primitives for the probabilistic decoder.

All functions here are pure NumPy, finite-safe and deterministic (ties are
always broken by explicit index order, never by object address). They
implement the small set of operations the v2 decoder needs:

* numerically stable ``logsumexp`` / ``log_softmax`` (no NaN/Inf on extreme
  inputs);
* temperature scaling (monotone in the temperature);
* monotone piecewise-linear calibration (used both for confidence
  calibration and for calibrating lexicon support into a log prior);
* the calibration/evaluation metrics the training report requires: NLL,
  Brier score, ECE, CER, exact-match rate.

Nothing in this module may depend on PyTorch/scikit-learn; training
scripts may use heavier tools, but the runtime and its metrics stay NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_LOG_2 = math.log(2.0)


def logsumexp(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Stable ``log(sum(exp(x)))``.

    Shifts by the per-slice max, so ``x`` may contain ``-inf`` (masked
    classes) and arbitrarily large finite values. Fully masked slices
    (all ``-inf``) return ``-inf``, never NaN.
    """

    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isfinite(x_max), x_max, -np.inf)
    # All-masked slices shift against 0 so no inf - inf NaN is ever formed.
    shift = np.where(np.isfinite(x_max), x_max, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        exp_sum = np.sum(np.exp(x - shift), axis=axis)
    out = x_max.reshape(exp_sum.shape if axis is not None else ()) + np.log(
        np.where(exp_sum > 0.0, exp_sum, 1.0)
    )
    # A slice that is entirely -inf keeps -inf; otherwise recompute exactly.
    finite = np.isfinite(x_max)
    if axis is not None:
        any_finite = finite.any(axis=axis)
    else:
        any_finite = bool(np.any(finite))
    out = np.where(any_finite, out, -np.inf)
    if axis is None:
        return float(out)
    return out


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log-softmax; rows of all-``-inf`` map to uniform ``-log(n)``.

    The uniform fallback keeps the decoder coverable when every class is
    masked (e.g. an empty allowed set), instead of producing NaN.

    The shift-cancelled form ``(x - shift) - log(sum(exp(x - shift)))``
    keeps the normalization constant exact even when ``|x|`` is huge
    (e.g. ``1e30``), where the naive ``x - logsumexp(x)`` would lose the
    constant to float cancellation.
    """

    logits = np.asarray(logits, dtype=np.float64)
    n = logits.shape[axis]
    x_max = np.max(logits, axis=axis, keepdims=True)
    shift = np.where(np.isfinite(x_max), x_max, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        s = np.sum(np.exp(logits - shift), axis=axis, keepdims=True)
    log_s = np.log(np.where(s > 0.0, s, 1.0))
    out = (logits - shift) - log_s
    all_inf = ~np.any(np.isfinite(logits), axis=axis, keepdims=True)
    uniform = -math.log(max(n, 1))
    return np.where(all_inf, uniform, out)


def _lse_keep(logits: np.ndarray, axis: int) -> np.ndarray:
    """logsumexp with the axis kept (dedicated helper, no keepdims dance)."""

    x = np.asarray(logits, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isfinite(x_max), x_max, -np.inf)
    shift = np.where(np.isfinite(x_max), x_max, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        s = np.sum(np.exp(x - shift), axis=axis, keepdims=True)
    lse = x_max + np.log(np.where(s > 0.0, s, 1.0))
    any_finite = np.any(np.isfinite(x), axis=axis, keepdims=True)
    return np.where(any_finite, lse, -np.inf)


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by a positive temperature (shape preserved).

    Monotone in the usual sense: for ``T2 > T1`` the softmax over scaled
    logits is flatter (closer to uniform). Raises on non-positive or
    non-finite temperatures.
    """

    t = float(temperature)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError(f"temperature must be finite and > 0, got {t!r}")
    return np.asarray(logits, dtype=np.float64) / t


def finite_logits(logits: np.ndarray) -> bool:
    """True when ``logits`` has no NaN and no +inf (``-inf`` masking is ok)."""

    x = np.asarray(logits)
    return bool(not np.any(np.isnan(x)) and not np.any(np.isposinf(x)))


def is_finite_array(values) -> bool:
    """True when every element of a nested float structure is finite."""

    arr = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.isfinite(arr)))


def calibrate_monotone(
    values: np.ndarray,
    points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> np.ndarray:
    """Monotone piecewise-linear calibration of ``values`` in ``[0, 1]``.

    ``points`` must be strictly increasing in x and non-decreasing in y,
    with x within ``[0, 1]``. Values outside the fitted range clamp to the
    end points. The empty point list is the identity.
    """

    pts = sorted((float(a), float(b)) for a, b in (points or ()))
    if not pts:
        return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    xs = np.asarray([p[0] for p in pts], dtype=np.float64)
    ys = np.asarray([p[1] for p in pts], dtype=np.float64)
    if np.any(np.diff(xs) <= 0.0):
        raise ValueError("calibration x points must be strictly increasing")
    if np.any(np.diff(ys) < 0.0):
        raise ValueError("calibration y points must be non-decreasing")
    v = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    out = np.interp(v, xs, ys)
    return np.clip(out, 0.0, 1.0)


def calibrate_log_monotone(
    log_values: np.ndarray,
    points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> np.ndarray:
    """Monotone calibration in log space (for log priors / log-evidence).

    ``points`` are ``(raw_value, calibrated_value)`` pairs with
    ``raw_value`` on the log scale; the mapping is monotone
    piecewise-linear in the raw value. This is what turns a heuristic
    support score into a calibrated log prior without taking ``log`` of a
    0..1 heuristic directly.
    """

    pts = sorted((float(a), float(b)) for a, b in (points or ()))
    if not pts:
        return np.asarray(log_values, dtype=np.float64)
    xs = np.asarray([p[0] for p in pts], dtype=np.float64)
    ys = np.asarray([p[1] for p in pts], dtype=np.float64)
    if np.any(np.diff(xs) <= 0.0):
        raise ValueError("log-calibration x points must be strictly increasing")
    if np.any(np.diff(ys) < 0.0):
        raise ValueError("log-calibration y points must be non-decreasing")
    v = np.asarray(log_values, dtype=np.float64)
    return np.interp(v, xs, ys)


@dataclass(frozen=True)
class CalibrationMetrics:
    """Calibration/evaluation metrics of one prediction set."""

    nll: float
    brier: float
    ece: float
    accuracy: float
    n: int


def calibration_metrics(
    confidences: np.ndarray,
    correct: np.ndarray,
    nll: float | None = None,
    bins: int = 10,
) -> CalibrationMetrics:
    """Brier / ECE / accuracy from binary correctness + predicted confidence.

    ``confidences`` in ``[0, 1]``, ``correct`` a bool array. ECE uses
    equal-mass bins; empty bins are skipped. ``nll`` may be passed in from
    a log-probability prediction (otherwise reported as NaN).
    """

    conf = np.asarray(confidences, dtype=np.float64)
    ok = np.asarray(correct, dtype=np.float64)
    if conf.ndim != 1 or ok.ndim != 1 or conf.shape != ok.shape:
        raise ValueError("confidences and correct must be equal-length 1D arrays")
    if conf.size == 0:
        return CalibrationMetrics(nll=float("nan"), brier=float("nan"), ece=float("nan"), accuracy=0.0, n=0)
    brier = float(np.mean((conf - ok) ** 2))
    acc = float(np.mean(ok))
    # Equal-mass binning; ties in confidence may make bins slightly uneven.
    order = np.argsort(conf, kind="stable")
    cs, os_ = conf[order], ok[order]
    n = cs.size
    edges = np.linspace(0, n, min(bins, n) + 1).astype(int)
    ece = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        block = slice(a, b)
        ece += float(np.abs(os_[block].mean() - cs[block].mean()) * (b - a) / n)
    return CalibrationMetrics(
        nll=float("nan") if nll is None else float(nll),
        brier=brier,
        ece=ece,
        accuracy=acc,
        n=n,
    )


def sequence_cer(hyp: str, ref: str) -> float:
    """Character error rate (Levenshtein distance / max(len, 1))."""

    h, r = hyp, ref
    m, n = len(h), len(r)
    if m == 0 or n == 0:
        return float(max(m, n)) / max(max(m, n), 1)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if h[i - 1] == r[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return float(prev[n]) / max(m, n)


def median_p95(values: list[float]) -> tuple[float, float]:
    """Deterministic median/p95 of a float list (sorted, interpolated)."""

    if not values:
        return float("nan"), float("nan")
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = arr.size
    med = float(np.median(arr))
    p95 = float(arr[min(n - 1, int(math.ceil(0.95 * n)) - 1)])
    return med, p95


def clamp_log_prob(x: float, lo: float = -30.0, hi: float = 0.0) -> float:
    """Clamp a log probability/log-evidence into a finite safe range."""

    return float(min(hi, max(lo, x)))
