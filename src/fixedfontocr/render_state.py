"""Line-level shared render state ``z`` (probabilistic decoder, v2).

Template V2's render size, sub-pixel phase and downsample mode should not
be chosen independently per character. This module implements the discrete
line-level render state:

* the finite state set is enumerated from the Goal 9 prototype metadata
  (``(render_size, downsample_mode)`` pairs plus the ``clean`` high-res
  prototype);
* every candidate's winning prototype votes for a state; the line keeps
  the top-``M`` states (default 3) in deterministic order;
* each state ``z`` conditions the local template evidence (the
  per-character score at that state), the whole path is scored per state,
  and the final path score marginalizes ``z`` with ``logsumexp`` (or,
  when a caller explicitly records the approximation, takes the max
  state);
* the ``clean`` state is always retained as the fallback for
  high-resolution prototypes that cannot be assigned to any grid state;
* the computation is deterministic and NaN/Inf-safe
  (:func:`fixedfontocr.prob_math.logsumexp`).
"""

from __future__ import annotations

import math

import numpy as np

from .prob_features import CLEAN_STATE, STATE_KEYS, LineContext
from .prob_math import logsumexp


class RenderStateModel:
    """Selects and marginalizes the line-level render state ``z``."""

    def __init__(
        self,
        top_m: int = 3,
        enabled: bool = True,
        state_prior: str = "uniform",
        clean_fallback: bool = True,
    ):
        if top_m < 1:
            raise ValueError("render_states.top_m must be >= 1")
        self.top_m = int(top_m)
        self.enabled = bool(enabled)
        if state_prior not in ("uniform",):
            raise ValueError(
                f"render_states.state_prior {state_prior!r} is not supported; "
                "use 'uniform'"
            )
        self.state_prior = state_prior
        self.clean_fallback = bool(clean_fallback)

    @classmethod
    def from_config(cls, cfg: dict | None) -> "RenderStateModel":
        cfg = cfg or {}
        return cls(
            top_m=int(cfg.get("top_m", 3)),
            enabled=bool(cfg.get("enabled", True)),
            state_prior=str(cfg.get("state_prior", "uniform")),
            clean_fallback=bool(cfg.get("clean_fallback", True)),
        )

    def select_states(self, ctx: LineContext) -> list[str]:
        """Top-M line states in deterministic order (clean always included).

        Returns at most ``top_m`` grid states plus the ``clean`` fallback
        (``top_m + 1`` states at most), so a high-resolution line always
        keeps a fallback state even when every candidate votes for grid
        states. When the machinery is disabled (``enabled=False``) only
        the ``clean`` fallback state remains, i.e. the decoder degenerates
        to a single unconditioned state.
        """

        if not self.enabled:
            return [CLEAN_STATE] if self.clean_fallback else []
        ranked = list(ctx.state_candidates)
        keys = [k for k, _s in ranked if k in STATE_KEYS]
        out: list[str] = []
        for k in keys:
            if k not in out:
                out.append(k)
            if len(out) >= self.top_m:
                break
        if self.clean_fallback and CLEAN_STATE not in out:
            out.append(CLEAN_STATE)
        return out

    def log_prior(self, states: list[str], z: str) -> float:
        """Uniform prior over the selected states (``-log M``)."""

        return -math.log(max(len(states), 1))

    def marginalize(
        self,
        per_state_scores: dict[str, float] | list[tuple[str, float]],
        use_max: bool = False,
    ) -> tuple[float, str]:
        """Marginalize ``z`` (logsumexp) or take the max state.

        Returns ``(marginal_score, argmax_state)``. ``use_max=True`` is the
        explicitly recorded max-state approximation; the default
        ``logsumexp`` is the exact marginalization. NaN entries are
        sanitized to ``-inf`` (config validation already refuses NaN
        parameters; this is a defensive guard) and fully-masked inputs
        return ``(-inf, "")``, never NaN.
        """

        if isinstance(per_state_scores, dict):
            items = [(k, float(v)) for k, v in per_state_scores.items()]
        else:
            items = [(k, float(v)) for k, v in per_state_scores]
        if not items:
            return -np.inf, ""
        keys = [k for k, _ in items]
        vals = np.asarray([v for _, v in items], dtype=np.float64)
        vals = np.where(np.isnan(vals), -np.inf, vals)
        if not np.any(np.isfinite(vals)):
            return -np.inf, keys[0]
        argmax = keys[int(np.argmax(vals))]
        if use_max:
            return float(np.max(vals)), argmax
        return float(logsumexp(vals)), argmax
