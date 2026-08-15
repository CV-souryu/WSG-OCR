"""Goal 13: the joint decoder (DP and beam search).

The decoder is the new architecture's core. It consumes the *visual
lattice* (all merge/split candidates with their Top-K visual scores), the
*font geometry* database, and an optional *lexicon*, then picks the best
path and its alternatives with one unified score:

    total =
        visual
      + geometry
      + lexicon
      + word_prior
      - segmentation_penalty

* ``visual`` is the classifier-only fused score of the chosen character
  (template + CNN before the geometry term);
* ``geometry`` is the Goal 8 font-geometry agreement of the candidate
  (bbox / aspect / ink ratio against the chosen character's database
  entry, plus the pipeline's merge/split penalties);
* ``lexicon`` is the Goal 11/12 dictionary match of the complete visible
  string (exact, term-in-text, prefix/suffix/inner crop or gap crop),
  scaled by the path's visual uncertainty so a visually confident path is
  never rewritten by a dictionary substring (Goal 15);
* ``word_prior`` is a gentle per-character prior estimated from the
  lexicon's own terms (unigram support), gated by that character's visual
  uncertainty;
* ``segmentation_penalty`` is a small per-extra-hypothesis cost. It is a
  tie-breaker that prefers fewer character hypotheses when the visual
  evidence is close (so a glyph is not fragmented into many candidate
  characters, ``小`` -> three "characters"), while the substantive
  merge/split costs stay in the Goal 8 geometry score. It must stay small:
  a genuine multi-character line must never be penalized just because it
  has more characters.

Two search algorithms are provided:

* :func:`decode_dp` -- the first-version exact dynamic program. It keeps
  the current lexicon prefix (a trie node) in its DP state, so exact term
  completions and lexicon prefixes influence the path while candidate
  terms are being explored;
* :func:`decode_beam` -- the beam-search upgrade (``beam_width`` 8..32,
  default 16) that explores complete-path hypotheses and returns the
  runner-up texts as ``alternatives``. Every complete path retained by the
  beam (plus the exact-DP path as a safety net) is re-ranked with the full
  crop-aware :func:`score_path` formula, so the complete path score -- not
  the DP's local prefix/term bonuses -- is the final authority for ``best``.

Both return a :class:`DecodePath` whose ``text`` is the visible string,
``char_ids`` records the chosen charset id per candidate, and
``alternatives`` holds the runner-up strings. The public pipeline calls
:func:`decode_lattice` (beam search by default).
"""

from __future__ import annotations

import functools
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

from .geometry import FontGeometryDatabase, char_geometry
from .lexicon import Lexicon
from .types import DecodePath, VisualCandidate, VisualLattice

UNKNOWN_CHAR = "?"
BASE_PRIOR = 0.35


@dataclass(frozen=True)
class DecoderConfig:
    """Search width and scoring weights for the Goal 13 decoder.

    The score of a complete path is

    ``visual_weight * mean(visual) + geometry_weight * mean(geometry)
    + lexicon_weight * lexicon_match * (1 - visual)
    + word_prior_weight * mean(gated word_prior)
    - segmentation_penalty * max(0, n_candidates - 1)``.

    ``segmentation_penalty`` defaults to a small value (0.005) so it only
    breaks near-ties toward fewer hypotheses; real text with many
    characters is not penalized for its length.

    ``prefix_bonus``/``term_bonus`` are used by the DP's trie-guided search
    as a small look-ahead for lexicon prefixes and exact term completions;
    the public :func:`score_path` reports the exact formula above.
    """

    beam_width: int = 16
    visual_weight: float = 1.0
    geometry_weight: float = 1.0
    lexicon_weight: float = 0.2
    word_prior_weight: float = 0.15
    segmentation_penalty: float = 0.005
    prefix_bonus: float = 0.02
    term_bonus: float = 0.06
    num_alternatives: int = 8

    def __post_init__(self) -> None:
        if self.beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.num_alternatives < 1:
            raise ValueError("num_alternatives must be >= 1")
        if any(
            w < 0.0
            for w in (
                self.visual_weight,
                self.geometry_weight,
                self.lexicon_weight,
                self.word_prior_weight,
                self.segmentation_penalty,
            )
        ):
            raise ValueError("decoder weights/penalties must be >= 0")


@dataclass(frozen=True)
class PathScore:
    """Decomposed Goal 13 score of one complete decode path.

    ``visual``/``geometry`` are means over the path's candidates;
    ``word_prior`` is the mean per-character dictionary support gated by
    each character's visual uncertainty; ``lexicon`` is the raw Goal 11/12
    match score of the complete visible text (0 without a lexicon);
    ``segmentation_penalty`` is the raw ``max(0, n_candidates - 1)`` count
    (the config weight scales it in ``total``). In ``total`` the lexicon
    term is additionally scaled by ``1 - clip(visual, 0, 1)``, which is how
    Goal 15 keeps confident visual evidence dominant.
    """

    visual: float
    geometry: float
    lexicon: float
    word_prior: float
    segmentation_penalty: float
    total: float


@dataclass
class _TrieNode:
    children: dict[str, int] = field(default_factory=dict)
    term: bool = False


class _LexiconTrie:
    """Prefix trie over lexicon terms, used by the DP's lexicon state."""

    def __init__(self, terms: Iterable[str]):
        self._nodes: list[_TrieNode] = [_TrieNode()]
        for term in terms:
            node = 0
            for ch in term:
                nxt = self._nodes[node].children.get(ch)
                if nxt is None:
                    nxt = len(self._nodes)
                    self._nodes[node].children[ch] = nxt
                    self._nodes.append(_TrieNode())
                node = nxt
            self._nodes[node].term = True

    def step(self, node: int, ch: str) -> tuple[int, bool, bool]:
        """Advance the longest matching lexicon prefix by one character.

        Returns ``(next_node, on_prefix, at_term_end)``. When the character
        cannot continue the current prefix it restarts from the root, so a
        new lexicon prefix may begin at any position in the text.
        """

        nxt = self._nodes[node].children.get(ch)
        if nxt is None:
            root = self._nodes[0].children.get(ch)
            if root is None:
                return 0, False, False
            return root, True, self._nodes[root].term
        return nxt, True, self._nodes[nxt].term

@functools.lru_cache(maxsize=16)
def _char_priors(lexicon: Lexicon | None) -> dict[str, float]:
    """Per-character unigram support estimated from lexicon terms.

    Characters that appear in many dictionary terms get a prior close to
    1.0; characters absent from the dictionary keep ``BASE_PRIOR`` so
    unknown-but-clearly-visible text is still allowed (Goal 15).
    """

    if lexicon is None:
        return {}
    counts: Counter[str] = Counter()
    for term in lexicon.terms:
        counts.update(term)
    if not counts:
        return {}
    max_count = max(counts.values())
    return {
        ch: BASE_PRIOR + 0.65 * count / max_count
        for ch, count in counts.items()
    }


def _prepare(
    lattice: VisualLattice | list[VisualCandidate],
    n_components: int | None,
) -> tuple[VisualLattice | None, list[VisualCandidate], int]:
    """Normalize decoder input into (lattice, candidates, n_atoms)."""

    if isinstance(lattice, VisualLattice):
        n_components = len(lattice.components)
        candidates = list(lattice.candidates)
    else:
        candidates = list(lattice)
    if n_components is None:
        raise ValueError(
            "decoder needs n_components when given a candidate list"
        )
    if not candidates:
        return lattice if isinstance(lattice, VisualLattice) else None, [], 0
    n_atoms = max(c.atom_span[1] for c in candidates)
    return (
        lattice if isinstance(lattice, VisualLattice) else None,
        candidates,
        n_atoms,
    )


def _char_options(
    candidate: VisualCandidate,
    charset: list[str],
) -> list[tuple[int, str, float]]:
    """All ranked ``(char_id, char, visual_score)`` choices of a candidate.

    Uses the candidate's Goal 1 Top-K ``VisualScores`` when present
    (never only Top-1); otherwise falls back to the legacy
    ``candidate.score`` top-1 pick. A candidate with no valid character
    (e.g. every class is filtered by ``allowed_chars``) yields the public
    unknown ``"?"`` with visual score 0 so the lattice is always coverable.
    """

    out: list[tuple[int, str, float]] = []
    scores = candidate.scores
    if scores is not None and scores.char_ids:
        logits = scores.logits or ()
        for i, cid in enumerate(scores.char_ids):
            cid = int(cid)
            if 0 <= cid < len(charset):
                visual = (
                    float(logits[i])
                    if i < len(logits)
                    else float(scores.visual_score)
                )
                out.append((cid, charset[cid], visual))
    if not out and candidate.score is not None:
        cid = int(getattr(candidate.score, "char_id", -1))
        if 0 <= cid < len(charset):
            visual = float(getattr(candidate.score, "visual_score", 0.0))
            out.append((cid, charset[cid], visual))
    if not out:
        out.append((-1, UNKNOWN_CHAR, 0.0))
    return out


def _candidate_geometry(
    candidate: VisualCandidate,
    char_id: int,
    geometry: FontGeometryDatabase | None,
) -> float:
    """Geometry evidence for one candidate/character choice.

    Goal 20 split:

    * ``candidate.candidate_geometry`` is the identity-independent half
      computed by the segmentation pipeline (alignment, gaps, segmentation
      width). It is computed once per candidate.
    * ``char_geometry(candidate, char_id, ...)`` is the character-specific
      half and is evaluated here for *every* character alternative, so a
      lexicon-selected Top-2/Top-3 character never inherits the Top-1
      character's bbox/aspect/ink/component geometry.

    Legacy/manually-built lattices that only provide ``geometry_score`` keep
    the old scalar full-geometry behaviour.
    """

    base = candidate.candidate_geometry
    if base is not None:
        # Pipeline candidates have the identity-independent half precomputed.
        # Preserve the legacy exact-template override for the top-1 char: an
        # exact visual match proves the candidate is a real glyph, so the
        # whole geometry term is zero for that char.
        if _exact_template_geometry_override(candidate, int(char_id)):
            return 0.0
        return float(
            max(
                base
                + char_geometry(
                    candidate,
                    int(char_id),
                    geometry,
                    normalize_geometry=candidate.normalize_geometry,
                ),
                -0.12,
            )
        )

    if candidate.geometry_score != 0.0:
        return float(candidate.geometry_score)
    if geometry is None or char_id < 0:
        return 0.0
    seg = candidate.segment
    if seg is None or seg.h < 2 or seg.w < 1:
        return 0.0
    # Manual lattices without pipeline geometry still get the character-
    # specific database agreement as a real decoder input.
    return char_geometry(
        candidate,
        int(char_id),
        geometry,
        normalize_geometry=candidate.normalize_geometry,
    )


def _exact_template_geometry_override(
    candidate: VisualCandidate,
    char_id: int,
) -> bool:
    """True when the pipeline zeroed geometry for an exact template Top-1."""
    score = candidate.score
    return bool(
        score is not None
        and getattr(score, "char_id", -1) == char_id
        and getattr(score, "score_type", "") == "template"
        and getattr(score, "template_raw_score", 0.0) >= 1.0 - 1e-9
        and candidate.geometry_score == 0.0
    )


def _chosen_terms(
    candidate: VisualCandidate,
    char_id: int,
    charset: list[str],
    priors: dict[str, float],
    geometry: FontGeometryDatabase | None,
) -> tuple[str, float, float, float]:
    """``(char, visual, geometry, word_prior)`` for one chosen character.

    ``word_prior`` is gated by the chosen character's visual uncertainty
    (``prior * (1 - visual)``) so dictionary support can only break ties or
    help genuinely ambiguous glyphs -- never override a confident visual
    pick (Goal 15).
    """

    for cid, ch, visual in _char_options(candidate, charset):
        if cid == int(char_id):
            geom = _candidate_geometry(candidate, cid, geometry)
            prior = priors.get(ch, BASE_PRIOR) if priors else 0.0
            prior *= 1.0 - float(np.clip(visual, 0.0, 1.0))
            return ch, visual, geom, prior
    return (
        UNKNOWN_CHAR,
        0.0,
        _candidate_geometry(candidate, -1, geometry),
        priors.get(UNKNOWN_CHAR, BASE_PRIOR) if priors else 0.0,
    )


def _empty_path(lattice: VisualLattice | None) -> DecodePath:
    return DecodePath(candidates=(), mean_score=0.0, lattice=lattice)


def score_path(
    path: DecodePath,
    charset: list[str],
    lexicon: Lexicon | None = None,
    config: DecoderConfig | None = None,
    geometry: FontGeometryDatabase | None = None,
) -> PathScore:
    """Goal 13 score of one complete path (the canonical public formula)."""

    cfg = config or DecoderConfig()
    n = len(path.candidates)
    if n == 0:
        return PathScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    priors = _char_priors(lexicon)
    visuals: list[float] = []
    geometries: list[float] = []
    priors_seen: list[float] = []
    char_ids = path.char_ids or ()
    for i, cand in enumerate(path.candidates):
        cid = int(char_ids[i]) if i < len(char_ids) else _top_char_id(cand)
        _ch, visual, geom, prior = _chosen_terms(
            cand, cid, charset, priors, geometry
        )
        visuals.append(visual)
        geometries.append(geom)
        priors_seen.append(prior)
    visual = float(np.mean(visuals)) if visuals else 0.0
    geometry = float(np.mean(geometries)) if geometries else 0.0
    word_prior = float(np.mean(priors_seen)) if priors_seen else 0.0
    lex_score = 0.0
    if lexicon is not None and path.text:
        best = lexicon.best_match(path.text)
        lex_score = float(best.score) if best is not None else 0.0
    seg_penalty = float(max(0, n - 1))
    # Goal 15 gate: the lexicon may only help when the visual evidence is
    # not already confident, otherwise a clear screen glyph would be
    # rewritten by a dictionary substring (e.g. "Z3" inside "ABCXYZ999").
    uncertainty = 1.0 - float(np.clip(visual, 0.0, 1.0))
    total = (
        cfg.visual_weight * visual
        + cfg.geometry_weight * geometry
        + cfg.lexicon_weight * lex_score * uncertainty
        + cfg.word_prior_weight * word_prior
        - cfg.segmentation_penalty * seg_penalty
    )
    return PathScore(
        visual=visual,
        geometry=geometry,
        lexicon=lex_score,
        word_prior=word_prior,
        segmentation_penalty=seg_penalty,
        total=float(total),
    )


def _top_char_id(candidate: VisualCandidate) -> int:
    scores = candidate.scores
    if scores is not None and scores.char_ids:
        return int(scores.char_ids[0])
    if candidate.score is not None:
        return int(getattr(candidate.score, "char_id", -1))
    return -1


def _decode_path(
    candidates: tuple[VisualCandidate, ...],
    char_ids: tuple[int, ...],
    charset: list[str],
    lattice: VisualLattice | None,
) -> DecodePath:
    text = "".join(
        (
            charset[int(cid)]
            if 0 <= int(cid) < len(charset)
            else UNKNOWN_CHAR
        )
        for cid in char_ids
    )
    return DecodePath(
        candidates=candidates,
        text=text,
        char_ids=char_ids,
        lattice=lattice,
    )


def decode_dp(
    lattice: VisualLattice | list[VisualCandidate],
    charset: list[str],
    lexicon: Lexicon | None = None,
    config: DecoderConfig | None = None,
    n_components: int | None = None,
    geometry: FontGeometryDatabase | None = None,
) -> DecodePath:
    """Exact DP over the visual lattice (first-version Goal 13 decoder).

    The DP state is ``(atom_end, lexicon_trie_node)``: the trie node keeps
    the longest lexicon prefix of the path text, so exact term completions
    and lexicon prefixes receive a small look-ahead bonus while the path is
    being built. The returned path is then re-scored with the complete
    Goal 13 formula (:func:`score_path`), including crop-aware lexicon
    matches.
    """

    lat, candidates, n_atoms = _prepare(lattice, n_components)
    if not candidates or n_atoms == 0:
        return _empty_path(lat)
    cfg = config or DecoderConfig()
    priors = _char_priors(lexicon)
    trie = _LexiconTrie(lexicon.terms) if lexicon is not None else None

    by_end: dict[int, list[VisualCandidate]] = {}
    for cand in candidates:
        by_end.setdefault(cand.atom_span[1], []).append(cand)

    NEG = -1e18
    # (end, node) -> (local_sum, count, prev_end, prev_node, candidate, char_id)
    dp: dict[tuple[int, int], tuple[float, int, object]] = {
        (0, 0): (0.0, 0, None)
    }
    for end in range(1, n_atoms + 1):
        for cand in by_end.get(end, ()):
            start = cand.atom_span[0]
            for (s, node), (local_sum, count, _prev) in list(dp.items()):
                if s != start:
                    continue
                for cid, ch, visual in _char_options(cand, charset):
                    geom = _candidate_geometry(cand, cid, geometry)
                    prior = priors.get(ch, BASE_PRIOR) if priors else 0.0
                    uncertainty = 1.0 - float(np.clip(visual, 0.0, 1.0))
                    prior *= uncertainty
                    if trie is not None:
                        node2, on_prefix, at_term = trie.step(node, ch)
                    else:
                        node2, on_prefix, at_term = 0, False, False
                    local = (
                        cfg.visual_weight * visual
                        + cfg.geometry_weight * geom
                        + cfg.word_prior_weight * prior
                        + (cfg.prefix_bonus if on_prefix else 0.0) * uncertainty
                        + (cfg.term_bonus if at_term else 0.0) * uncertainty
                    )
                    new_sum = local_sum + local
                    new_count = count + 1
                    score = (
                        new_sum / new_count
                        - cfg.segmentation_penalty * (new_count - 1)
                    )
                    key = (end, node2)
                    prev = dp.get(key)
                    if prev is None or score > prev[0] / prev[1] - cfg.segmentation_penalty * (
                        prev[1] - 1
                    ) + 1e-12:
                        dp[key] = (
                            new_sum,
                            new_count,
                            (dp[(start, node)], cand, cid),
                        )

    best: tuple[float, int, object] | None = None
    best_key: tuple[int, int] | None = None
    best_score = NEG
    for key, state in dp.items():
        if key[0] != n_atoms:
            continue
        score = state[0] / state[1] - cfg.segmentation_penalty * (state[1] - 1)
        if best is None or score > best_score + 1e-12:
            best = state
            best_key = key
            best_score = score
    if best is None or best_key is None:
        return _empty_path(lat)

    chosen: list[VisualCandidate] = []
    char_ids: list[int] = []
    pos, node = best_key
    state = dp[(pos, node)]
    while state[2] is not None:
        prev_state, cand, cid = state[2]  # type: ignore[misc]
        chosen.append(cand)
        char_ids.append(int(cid))
        state = prev_state
    chosen.reverse()
    char_ids.reverse()
    path = _decode_path(
        tuple(chosen), tuple(char_ids), charset, lat
    )
    ps = score_path(path, charset, lexicon, cfg, geometry)
    confidence = float(np.clip(ps.visual, 0.0, 1.0))
    return replace(
        path,
        mean_score=ps.total,
        confidence=confidence,
        alternatives=(),
    )


@dataclass
class _Hypothesis:
    end: int
    candidates: tuple[VisualCandidate, ...] = ()
    char_ids: tuple[int, ...] = ()
    text: str = ""
    visuals: tuple[float, ...] = ()
    geometries: tuple[float, ...] = ()
    priors: tuple[float, ...] = ()


def _partial_total(
    hyp: _Hypothesis,
    config: DecoderConfig,
    lexicon: Lexicon | None,
) -> float:
    """Beam-search look-ahead score for a partial path.

    Uses the same Goal 13 terms as :func:`score_path`, with the crop-aware
    lexicon match evaluated on the current visible prefix so lexicon
    evidence guides exploration before the path is complete.
    """

    n = len(hyp.visuals)
    if n == 0:
        return 0.0
    visual = sum(hyp.visuals) / n
    geometry = sum(hyp.geometries) / n
    word_prior = sum(hyp.priors) / n
    lex = 0.0
    if lexicon is not None and hyp.text:
        best = lexicon.best_match(hyp.text)
        lex = float(best.score) if best is not None else 0.0
    uncertainty = 1.0 - float(np.clip(visual, 0.0, 1.0))
    return (
        config.visual_weight * visual
        + config.geometry_weight * geometry
        + config.word_prior_weight * word_prior
        + config.lexicon_weight * lex * uncertainty
        - config.segmentation_penalty * (n - 1)
    )


def _hypothesis_path(
    hyp: _Hypothesis,
    charset: list[str],
    lattice: VisualLattice | None,
) -> DecodePath:
    return _decode_path(hyp.candidates, hyp.char_ids, charset, lattice)


def _beam_complete_paths(
    lattice: VisualLattice | list[VisualCandidate],
    charset: list[str],
    lexicon: Lexicon | None,
    config: DecoderConfig,
    n_atoms: int,
    geometry: FontGeometryDatabase | None,
) -> list[DecodePath]:
    """All complete paths retained by the beam search.

    The beam itself still uses the partial Goal 13 score as a look-ahead,
    but this helper returns complete ``DecodePath`` objects (not just text)
    so the caller can rank every retained path with the canonical
    :func:`score_path` formula.
    """

    lat = lattice if isinstance(lattice, VisualLattice) else None
    candidates = list(lattice.candidates if lat is not None else lattice)
    priors = _char_priors(lexicon)
    by_start: dict[int, list[VisualCandidate]] = {}
    for cand in candidates:
        by_start.setdefault(cand.atom_span[0], []).append(cand)

    beam: list[_Hypothesis] = [_Hypothesis(end=0)]
    for _pos in range(n_atoms):
        new_beam: list[_Hypothesis] = [
            h for h in beam if h.end >= n_atoms
        ]
        for hyp in beam:
            if hyp.end >= n_atoms:
                continue
            for cand in by_start.get(hyp.end, ()):
                end = cand.atom_span[1]
                if end <= hyp.end:
                    continue
                for cid, ch, visual in _char_options(cand, charset):
                    geom = _candidate_geometry(cand, cid, geometry)
                    prior = priors.get(ch, BASE_PRIOR) if priors else 0.0
                    prior *= 1.0 - float(np.clip(visual, 0.0, 1.0))
                    new_beam.append(
                        _Hypothesis(
                            end=end,
                            candidates=hyp.candidates + (cand,),
                            char_ids=hyp.char_ids + (int(cid),),
                            text=hyp.text + ch,
                            visuals=hyp.visuals + (visual,),
                            geometries=hyp.geometries + (geom,),
                            priors=hyp.priors + (prior,),
                        )
                    )
        if not new_beam:
            break
        new_beam.sort(
            key=lambda h: _partial_total(h, config, lexicon),
            reverse=True,
        )
        beam = new_beam[: config.beam_width]

    complete = [h for h in beam if h.end == n_atoms]
    paths: list[DecodePath] = []
    seen: set[tuple[str, tuple[int, ...], tuple[int, ...]]] = set()
    for hyp in complete:
        path = _hypothesis_path(hyp, charset, lat)
        key = (
            path.text,
            tuple(id(c) for c in path.candidates),
            tuple(int(cid) for cid in path.char_ids),
        )
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def decode_beam(
    lattice: VisualLattice | list[VisualCandidate],
    charset: list[str],
    lexicon: Lexicon | None = None,
    config: DecoderConfig | None = None,
    n_components: int | None = None,
    geometry: FontGeometryDatabase | None = None,
) -> DecodePath:
    """Beam-search decoder over the visual lattice (Goal 13 upgrade).

    The beam explores complete-path hypotheses and returns both the best
    path and runner-up texts. Every retained complete path -- including the
    exact-DP path as a safety net -- is ranked with the canonical complete
    :func:`score_path` formula, so the complete crop-aware lexicon score
    decides ``best``. The previous behaviour (the DP result always winning
    over the beam alternatives) is intentionally gone: local DP prefix/term
    bonuses are only a look-ahead, never the final authority.
    """

    lat, candidates, n_atoms = _prepare(lattice, n_components)
    if not candidates or n_atoms == 0:
        return _empty_path(lat)
    cfg = config or DecoderConfig()

    dp_path = decode_dp(
        lattice,
        charset,
        lexicon,
        cfg,
        n_components,
        geometry,
    )
    beam_paths = _beam_complete_paths(
        lattice,
        charset,
        lexicon,
        cfg,
        n_atoms,
        geometry,
    )

    scored: list[tuple[float, PathScore, DecodePath]] = []
    seen: set[tuple[str, tuple[int, ...], tuple[int, ...]]] = set()
    for path in [dp_path, *beam_paths]:
        if not path.candidates:
            continue
        key = (
            path.text,
            tuple(id(c) for c in path.candidates),
            tuple(int(cid) for cid in path.char_ids),
        )
        if key in seen:
            continue
        seen.add(key)
        ps = score_path(path, charset, lexicon, cfg, geometry)
        scored.append((ps.total, ps, path))
    if not scored:
        return dp_path

    # Full-path score is the only final authority. Tie-break deterministically
    # toward fewer candidates and then lexicographically by visible text.
    scored.sort(
        key=lambda row: (
            -row[0],
            len(row[2].candidates),
            row[2].text,
        )
    )
    best_total, best_ps, best_path = scored[0]
    alternatives: list[str] = []
    for _total, _ps, path in scored:
        if (
            path.text
            and path.text != best_path.text
            and path.text not in alternatives
        ):
            alternatives.append(path.text)
        if len(alternatives) >= cfg.num_alternatives:
            break
    return replace(
        best_path,
        mean_score=float(best_total),
        confidence=float(np.clip(best_ps.visual, 0.0, 1.0)),
        alternatives=tuple(alternatives),
    )


def decode_lattice(
    lattice: VisualLattice,
    charset: list[str],
    lexicon: Lexicon | None = None,
    config: DecoderConfig | None = None,
    geometry: FontGeometryDatabase | None = None,
    beam: bool = True,
) -> DecodePath:
    """Public Goal 13 decoder entry point.

    ``beam=True`` (default) runs the beam-search upgrade and returns the
    best path with ``alternatives``; ``beam=False`` runs the first-version
    exact DP.
    """

    if beam:
        return decode_beam(lattice, charset, lexicon, config, geometry=geometry)
    return decode_dp(lattice, charset, lexicon, config, geometry=geometry)
