"""Unified domain prior interface for the probabilistic decoder (v2).

The path score includes ``log P(c_{1:n} | domain)``. This module defines
the ``DomainPrior`` interface and four concrete priors:

* :class:`OpenTextPrior` -- uniform over the whole charset (or the
  ``allowed_chars`` subset), i.e. no domain information;
* :class:`AllowedCharsPrior` -- uniform over a caller-supplied allowed
  character set (the UI-field domain). Visual-equivalent characters such
  as Latin ``T`` / Cyrillic ``Т`` are disambiguated *only* inside the
  allowed set: the local softmax normalizer simply excludes the forbidden
  classes, exactly like v1's ``allowed_chars`` restriction;
* :class:`UnigramPrior` -- a smoothed per-character unigram (e.g. from the
  game UI corpus), returned as a calibrated log prior;
* :class:`LexiconPrior` -- the Goal 11 lexicon's *term support* mapped to
  a calibrated log prior. The 0..1 heuristic match score of the legacy
  matcher is never fed to ``log`` directly: only the lexicon's
  per-character support counts (a frequency, not a heuristic) enter, and
  the map from support to log-prior is a monotone calibration fitted on
  the validation split and stored in the model config.

Goal 15 principles are enforced structurally: every prior is a bounded
additive term, so a clear visual log-likelihood always dominates; priors
that are non-unique (near-uniform support) stay near-flat, keeping the
result ambiguous instead of inventing a character.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .prob_math import calibrate_log_monotone

_EPS = 1e-9


class DomainPrior(ABC):
    """Maps a character id to a calibrated log prior."""

    name: str = "domain"

    @abstractmethod
    def log_prior(self, char_id: int) -> float:
        ...

    @abstractmethod
    def to_config(self) -> dict:
        ...


@dataclass(frozen=True)
class OpenTextPrior(DomainPrior):
    """Uniform prior over a charset (optionally restricted)."""

    charset: tuple[str, ...] = ()
    allowed_ids: frozenset[int] = frozenset()
    name: str = "open_text"

    def __post_init__(self) -> None:
        n = len(self.allowed_ids) if self.allowed_ids else len(self.charset)
        object.__setattr__(self, "_log_uniform", -math.log(max(n, 1)))

    def log_prior(self, char_id: int) -> float:
        if self.allowed_ids and int(char_id) not in self.allowed_ids:
            return -np.inf
        return float(self._log_uniform)

    def to_config(self) -> dict:
        return {"type": "open_text"}


@dataclass(frozen=True)
class AllowedCharsPrior(DomainPrior):
    """Uniform prior over the caller's ``allowed_chars`` subset."""

    allowed_ids: frozenset[int]
    charset: tuple[str, ...] = ()
    name: str = "allowed_chars"

    def __post_init__(self) -> None:
        n = len(self.allowed_ids)
        object.__setattr__(self, "_log_uniform", -math.log(max(n, 1)))
        object.__setattr__(self, "_chars", sorted(int(i) for i in self.allowed_ids))

    def log_prior(self, char_id: int) -> float:
        if int(char_id) not in self.allowed_ids:
            return -np.inf
        return float(self._log_uniform)

    def to_config(self) -> dict:
        return {
            "type": "allowed_chars",
            "n_allowed": len(self.allowed_ids),
            "allowed": "".join(
                self.charset[i] for i in self._chars if 0 <= i < len(self.charset)
            ),
        }


@dataclass(frozen=True)
class UnigramPrior(DomainPrior):
    """Smoothed per-character unigram, calibrated to a log prior."""

    # count per char id (raw frequency), calibrated by a monotone map.
    counts: dict[int, float] = field(default_factory=dict)
    total: float = 0.0
    # monotone log-calibration points [(raw_log_support, calibrated_log_p)]
    calibration: tuple[tuple[float, float], ...] = ()
    floor_log_p: float = -8.0
    name: str = "unigram"

    def __post_init__(self) -> None:
        if self.total <= 0.0:
            tot = sum(self.counts.values())
            object.__setattr__(self, "total", tot if tot > 0.0 else 1.0)

    def _raw_support(self, char_id: int) -> float:
        return float(self.counts.get(int(char_id), 0.0))

    def log_prior(self, char_id: int) -> float:
        raw = self._raw_support(int(char_id))
        if raw <= 0.0:
            return float(self.floor_log_p)
        raw_log = math.log(raw / max(self.total, 1.0))
        if self.calibration:
            val = float(
                calibrate_log_monotone(
                    np.asarray([raw_log], dtype=np.float64), self.calibration
                )[0]
            )
        else:
            val = raw_log
        return float(max(val, self.floor_log_p))

    def to_config(self) -> dict:
        return {
            "type": "unigram",
            "total": self.total,
            "n_chars": len(self.counts),
            "calibration": [list(p) for p in self.calibration],
            "floor_log_p": self.floor_log_p,
        }


@dataclass(frozen=True)
class LexiconPrior(DomainPrior):
    """Lexicon term support -> calibrated log prior (never log(heuristic))."""

    # per-char support counts from lexicon terms (frequency, not match score)
    counts: dict[int, float] = field(default_factory=dict)
    max_count: float = 1.0
    # monotone log-calibration points over log(support / max_count)
    calibration: tuple[tuple[float, float], ...] = ()
    # characters absent from the lexicon keep a floor prior (unknown text is
    # still emitted, Goal 15).
    floor_log_p: float = -6.0
    lexicon_name: str = ""
    name: str = "lexicon"

    def __post_init__(self) -> None:
        if self.max_count <= 0.0:
            mc = max(self.counts.values()) if self.counts else 1.0
            object.__setattr__(self, "max_count", float(mc) if mc > 0 else 1.0)

    def log_prior(self, char_id: int) -> float:
        count = float(self.counts.get(int(char_id), 0.0))
        if count <= 0.0:
            return float(self.floor_log_p)
        raw_log = math.log(count / max(self.max_count, 1.0))
        if self.calibration:
            val = float(
                calibrate_log_monotone(
                    np.asarray([raw_log], dtype=np.float64), self.calibration
                )[0]
            )
        else:
            val = raw_log
        return float(max(val, self.floor_log_p))

    def to_config(self) -> dict:
        return {
            "type": "lexicon",
            "lexicon_name": self.lexicon_name,
            "max_count": self.max_count,
            "n_chars": len(self.counts),
            "calibration": [list(p) for p in self.calibration],
            "floor_log_p": self.floor_log_p,
        }


@dataclass(frozen=True)
class LexiconPriorBuilder:
    """Builds a :class:`LexiconPrior` from a Goal 11 lexicon."""

    lexicon_name: str
    terms: tuple[str, ...] = ()

    def build(self, charset: list[str]) -> LexiconPrior:
        cid_map = {ch: i for i, ch in enumerate(charset)}
        counts: dict[int, float] = {}
        for term in self.terms:
            for ch in term:
                cid = cid_map.get(ch)
                if cid is not None:
                    counts[cid] = counts.get(cid, 0.0) + 1.0
        return LexiconPrior(
            counts=counts,
            max_count=max(counts.values()) if counts else 1.0,
            lexicon_name=self.lexicon_name,
        )

    @classmethod
    def from_lexicon(cls, lexicon, charset: list[str]) -> "LexiconPriorBuilder":
        return cls(lexicon_name=str(getattr(lexicon, "name", "lexicon")),
                   terms=tuple(getattr(lexicon, "terms", ())))


def domain_prior_from_config(
    cfg: dict | None,
    charset: list[str],
    allowed_ids: set[int] | None = None,
    lexicon=None,
) -> DomainPrior:
    """Build the configured domain prior (with runtime overrides).

    ``allowed_ids``/``lexicon`` supplied by the caller override the
    configured prior type (the UI-field restriction and the requested
    dictionary are per-call evidence). Without overrides the config's
    stored prior (e.g. the trained lexicon/unigram calibration) is used.
    """

    charset_t = tuple(charset)
    if allowed_ids is not None:
        ids = frozenset(int(i) for i in allowed_ids)
        if not ids:
            # Empty allowed set: uniform over nothing -> every char is
            # forbidden; the decoder's "?" fallback handles coverage.
            return AllowedCharsPrior(ids, charset_t)
        if lexicon is not None:
            builder = LexiconPriorBuilder.from_lexicon(lexicon, charset)
            prior = builder.build(charset)
            cal = (cfg or {}).get("calibration")
            if cal:
                prior = LexiconPrior(
                    counts=prior.counts,
                    max_count=prior.max_count,
                    calibration=tuple((float(a), float(b)) for a, b in cal),
                    floor_log_p=float((cfg or {}).get("floor_log_p", -6.0)),
                    lexicon_name=prior.lexicon_name,
                )
            return prior
        return AllowedCharsPrior(ids, charset_t)
    if lexicon is not None:
        builder = LexiconPriorBuilder.from_lexicon(lexicon, charset)
        prior = builder.build(charset)
        cal = (cfg or {}).get("calibration")
        if cal:
            prior = LexiconPrior(
                counts=prior.counts,
                max_count=prior.max_count,
                calibration=tuple((float(a), float(b)) for a, b in cal),
                floor_log_p=float((cfg or {}).get("floor_log_p", -6.0)),
                lexicon_name=prior.lexicon_name,
            )
        return prior
    return OpenTextPrior(charset_t)
