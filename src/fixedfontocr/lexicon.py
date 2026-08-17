"""Goal 11/12: Lexicon Layer with partial-word support.

The runtime lexicon is a pure-Python layer over ``charsets/words/``. It
loads the domain word lists (ships / equipment / ui), normalizes away the
whitespace the OCR pipeline cannot emit, and exposes three API modes:

* ``none``    -- no dictionary evidence is used;
* ``prefer``  -- the dictionary annotates the result and may resolve a
  visually ambiguous character when that character is already a plausible
  Top-K alternative and the visual evidence is not confident;
* ``topk``    -- "Top-3 取词表": when the visible text is not itself a
  dictionary term, promote the first decoder alternative that is an exact
  term, or rewrite from the per-position visual Top-3 when a term is fully
  supported by those alternatives (unique matches only);
* ``strict``  -- only a full dictionary term is accepted; otherwise the
  result is rejected (empty text, confidence 0).

The layer never invents visible text: ``prefer`` changes a character only
when the lexicon target is present in that candidate's Top-K visual
alternatives and the original character was visually uncertain.

Goal 12 partial-word matching is built into the same matcher. A screen
crop of a dictionary term (``"C2C3C4C5"`` from ``"C1C2C3C4C5C6"``) is
allowed as ``prefix_crop`` / ``suffix_crop`` / ``inner_crop``, while
internally missing characters are ranked as ``gap_crop`` with a higher
penalty. The visible text is never expanded into the full term; the
inferred entity and its half-open span are returned in
``matched_term`` / ``matched_span`` (and ``LexiconMatch``).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .defaults import PROJECT_ROOT
from .types import CharResult, LexiconMatch, OCRResult
from . import realglyphs as _realglyphs

WORDS_DIR = PROJECT_ROOT / "charsets" / "words"

# Public domain names and aliases accepted by ``load_lexicon`` /
# ``recognize(..., lexicon=...)``.
LEXICON_FILES: dict[str, str] = {
    # ``ships`` is a domain that combines both ship-name word lists; the
    # mapping below names the primary list for introspection.
    "ships": "ship_names.txt",
    "ship": "ship_names.txt",
    "ship_names": "ship_names.txt",
    "harmonized": "ship_names_harmonized.txt",
    "ships_harmonized": "ship_names_harmonized.txt",
    "harmonized_ships": "ship_names_harmonized.txt",
    "ship_names_harmonized": "ship_names_harmonized.txt",
    "equipment": "equipment_names.txt",
    "equipment_names": "equipment_names.txt",
    "ui": "ui_texts.txt",
    "ui_texts": "ui_texts.txt",
}
ALL_LEXICON_FILES: tuple[str, ...] = (
    "ship_names.txt",
    "ship_names_harmonized.txt",
    "equipment_names.txt",
    "ui_texts.txt",
)
LEXICON_DOMAINS: tuple[str, ...] = ("ships", "equipment", "ui")
SHIP_LEXICON_FILES: tuple[str, ...] = (
    "ship_names.txt",
    "ship_names_harmonized.txt",
)

LEXICON_MODES = ("none", "prefer", "topk", "strict", "dict")
TOP_K_GUESS = 3
TOP_K_GUESS_MIN_SCORE = 0.6
TOP_K_GUESS_UNIQUE_MARGIN = 0.02

# ``dict`` mode (associative dictionary inference): the visible text is
# scanned position by position and every dictionary term that can explain
# it -- exact, cropped, confusable or partially damaged -- competes as a
# word hypothesis. The winner becomes ``matched_term``; without any
# surviving hypothesis the whole result is rejected (empty output).
DICT_GAP_PEN = 0.18          # dictionary char with no visible evidence
DICT_SUB_PEN = 0.06          # visible char replaced by a Top-K confusable
DICT_DROP_PEN = 0.18         # per dropped residue char, scaled by the
                             # explained term span (mprime)
DICT_LEN_PEN = 0.15          # per-char penalty when term is shorter than text
DICT_MIN_SCORE = 0.50        # acceptance floor for multi-char terms
DICT_MIN_SCORE_SINGLE = 0.80  # stricter floor for single-char terms
DICT_UNIQUE_MARGIN = 0.02    # best hypothesis must beat the runner-up term
DICT_FORCE_FLOOR = 0.32      # evidence floor for forced (unverified) chars
DICT_FORCE_CONF_CAP = 2      # max conflicting visible positions when forcing
DICT_FORCE_NEED_MATCH = 0.75  # matched-position mean required for forcing
DICT_MAX_DROPS = 3           # max visible chars dropped anywhere
DICT_MATCH_MEAN_MIN = 0.70   # matched-char mean evidence floor (regular)
DICT_CONTEXT_BONUS = 0.04    # tie-break bonus for caller-supplied context terms
DICT_VERIFY_TEMPLATE_MIN = 0.60  # forced target must be template-plausible
DICT_TEMPLATE_STRONG_MIN = 0.88  # full-window template recovery floor
DICT_TEMPLATE_RERANK_MAX = 0.80  # weak fallback score eligible for reranking
DICT_TOP_K = 5               # per-position visual Top-K consulted


def normalize_text(text: str) -> str:
    """Remove every whitespace character from a dictionary/OCR string.

    Spaces are layout, not ink: the pipeline emits ``"使 用"`` as
    ``"使用"``, so matching is always performed on whitespace-free strings.
    """

    return "".join(ch for ch in text if not ch.isspace())


def is_lexicon_ref(value: object) -> bool:
    """True when ``value`` can be passed as the ``lexicon`` argument."""

    if isinstance(value, Lexicon):
        return True
    if isinstance(value, Path):
        return value.is_file() or (PROJECT_ROOT / value).is_file()
    if isinstance(value, str):
        key = value.strip().lower()
        if key == "all" or key in LEXICON_FILES or key in LEXICON_DOMAINS:
            return True
        p = Path(value)
        return p.is_file() or (PROJECT_ROOT / p).is_file()
    return False


def _read_terms(path: Path) -> tuple[str, ...]:
    """Read one word per line, normalize whitespace and deduplicate."""

    terms: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        term = normalize_text(line)
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
    return tuple(terms)


def _match(
    term: str,
    span: tuple[int, int],
    score: float,
    kind: str,
    text: str,
    text_span: tuple[int, int] | None = None,
) -> LexiconMatch:
    return LexiconMatch(
        term=term,
        span=span,
        confidence=float(score),
        score=float(score),
        text=text,
        text_span=text_span,
        kind=kind,
    )


def _subsequence_span(text: str, term: str) -> tuple[tuple[int, int], int] | None:
    """Greedy earliest alignment of ``text`` inside ``term``.

    Returns the covered half-open term span and how many term characters
    are skipped inside that span. Used only for the Goal 12 partial-word
    case where visible characters are contiguous in the image but the
    dictionary term has internal characters missing from the screen crop.
    """

    if not text or not term or len(text) > len(term):
        return None
    idx = -1
    first: int | None = None
    last = -1
    for ch in text:
        idx = term.find(ch, idx + 1)
        if idx < 0:
            return None
        if first is None:
            first = idx
        last = idx
    if first is None:
        return None
    span = (first, last + 1)
    missing_inside = (span[1] - span[0]) - len(text)
    return span, max(0, missing_inside)


_KIND_RANK = {
    "exact": 0,
    "term_in_text": 1,
    "prefix_crop": 2,
    "suffix_crop": 3,
    "inner_crop": 4,
    "gap_crop": 5,
}


def match_term(text: str, term: str) -> LexiconMatch | None:
    """Match one normalized visible string against one dictionary term."""

    text = normalize_text(text)
    term = normalize_text(term)
    if not text or not term:
        return None

    if text == term:
        return _match(
            term,
            (0, len(term)),
            1.0,
            "exact",
            text,
            (0, len(text)),
        )

    pos = term.find(text)
    if pos >= 0:
        end = pos + len(text)
        if pos == 0:
            kind = "prefix_crop"
            score = 0.92
        elif end == len(term):
            kind = "suffix_crop"
            score = 0.92
        else:
            kind = "inner_crop"
            score = 0.80
        return _match(term, (pos, end), score, kind, text, (0, len(text)))

    pos = text.find(term)
    if pos >= 0:
        return _match(
            term,
            (0, len(term)),
            0.97,
            "term_in_text",
            text,
            (pos, pos + len(term)),
        )

    aligned = _subsequence_span(text, term)
    if aligned is None:
        return None
    span, missing_inside = aligned
    score = max(0.20, 0.70 - 0.12 * missing_inside)
    if span[0] > 0 and span[1] < len(term):
        score = max(0.15, score - 0.05)
    return _match(term, span, score, "gap_crop", text, (0, len(text)))


@dataclass(frozen=True)
class Lexicon:
    """An in-memory domain dictionary.

    ``terms`` are whitespace-normalized and deduplicated. Matching returns
    ranked :class:`LexiconMatch` objects; the highest-ranked entry is the
    best dictionary hypothesis and the visible text is never rewritten by
    the matcher itself.
    """

    name: str
    terms: tuple[str, ...]
    path: Path | None = None

    @classmethod
    def from_file(cls, path: Path, name: str | None = None) -> "Lexicon":
        path = Path(path)
        return cls(
            name=name or path.stem,
            terms=_read_terms(path),
            path=path,
        )

    @classmethod
    def load(cls, path: Path, name: str | None = None) -> "Lexicon":
        """Alias for :meth:`from_file`."""
        return cls.from_file(path, name=name)

    @classmethod
    def from_terms(
        cls,
        name: str,
        terms: Iterable[str],
        path: Path | None = None,
    ) -> "Lexicon":
        seen: set[str] = set()
        out: list[str] = []
        for raw in terms:
            term = normalize_text(raw)
            if term and term not in seen:
                seen.add(term)
                out.append(term)
        return cls(name=name, terms=tuple(out), path=path)

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self) -> Iterator[str]:
        return iter(self.terms)

    def __contains__(self, term: object) -> bool:
        return isinstance(term, str) and normalize_text(term) in self.terms

    @property
    def charset(self) -> str:
        """Union of every term character, sorted like ``charsets/sets``.

        ``tools/charset/extract_charset.py`` writes each domain charset as
        codepoint-sorted unique characters, so strict mode uses the same
        ordering as the checked-in set files.
        """

        seen: set[str] = set()
        for term in self.terms:
            for ch in term:
                seen.add(ch)
        return "".join(sorted(seen, key=ord))

    def match(self, text: str, limit: int | None = None) -> list[LexiconMatch]:
        """Return all dictionary matches, best first."""

        text = normalize_text(text)
        if not text:
            return []
        matches = [m for m in (match_term(text, t) for t in self.terms) if m]
        def _rank_key(m: LexiconMatch) -> tuple:
            # For a full term visible inside longer text, the longer term is
            # the more specific dictionary hypothesis; for a cropped view,
            # the shorter term is the safer inference.
            length_key = (
                -len(m.term) if m.kind == "term_in_text" else len(m.term)
            )
            return (
                -m.score,
                _KIND_RANK.get(m.kind, 99),
                length_key,
                m.span[0],
                m.term,
            )

        matches.sort(key=_rank_key)
        if limit is not None:
            if limit <= 0:
                return []
            return matches[:limit]
        return matches

    def best_match(self, text: str, unique: bool = False) -> LexiconMatch | None:
        """Return the top match, optionally only when it is unambiguous."""

        matches = self.match(text)
        if not matches:
            return None
        best = matches[0]
        if unique and len(matches) > 1:
            second = matches[1]
            if (
                second.term != best.term
                and abs(second.score - best.score) < 1e-9
            ):
                return None
        return best


# Compatibility name for the Goal 11 "Lexicon Layer" module concept.
LexiconLayer = Lexicon


@functools.lru_cache(maxsize=32)
def _load_cached(path: str, name: str) -> Lexicon:
    return Lexicon.from_file(Path(path), name=name)


@functools.lru_cache(maxsize=8)
def _load_all_cached() -> Lexicon:
    terms: list[str] = []
    seen: set[str] = set()
    for filename in ALL_LEXICON_FILES:
        for term in _read_terms(WORDS_DIR / filename):
            if term not in seen:
                seen.add(term)
                terms.append(term)
    # The generated ``all_charset.txt`` also includes the complete ASCII
    # set, some of which never appears in the game word lists. Keep the
    # "all" domain aligned with that checked-in charset by adding every
    # ASCII character as a single-character term.
    ascii_path = PROJECT_ROOT / "charsets" / "sets" / "ascii.txt"
    if ascii_path.is_file():
        for ch in ascii_path.read_text(encoding="utf-8"):
            if not ch.isspace() and ch not in seen:
                seen.add(ch)
                terms.append(ch)
    return Lexicon(name="all", terms=tuple(terms), path=WORDS_DIR)


@functools.lru_cache(maxsize=4)
def _load_ships_cached() -> Lexicon:
    terms: list[str] = []
    seen: set[str] = set()
    for filename in SHIP_LEXICON_FILES:
        for term in _read_terms(WORDS_DIR / filename):
            if term not in seen:
                seen.add(term)
                terms.append(term)
    return Lexicon(
        name="ships",
        terms=tuple(terms),
        path=WORDS_DIR / SHIP_LEXICON_FILES[0],
    )


def load_lexicon(lexicon: str | Path | Lexicon) -> Lexicon:
    """Load a built-in or explicit word-list lexicon.

    Built-in names: ``ships`` / ``ship_names`` / ``ship``,
    ``harmonized_ships`` / ``ship_names_harmonized``, ``equipment`` /
    ``equipment_names``, ``ui`` / ``ui_texts``, and ``all``. A path to any
    one-word-per-line file is also accepted.
    """

    if isinstance(lexicon, Lexicon):
        return lexicon

    if isinstance(lexicon, Path) or (
        isinstance(lexicon, str)
        and (
            lexicon.endswith(".txt")
            or "/" in lexicon
            or "\\" in lexicon
            or (PROJECT_ROOT / lexicon).is_file()
        )
    ):
        path = Path(lexicon)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"lexicon file not found: {path}")
        return _load_cached(str(path), path.stem)

    if not isinstance(lexicon, str):
        raise TypeError(
            "lexicon must be a Lexicon, a path, or a built-in name, "
            f"got {type(lexicon).__name__}"
        )
    key = lexicon.strip().lower()
    if key == "all":
        return _load_all_cached()
    if key == "ships":
        return _load_ships_cached()
    if key not in LEXICON_FILES:
        known = ", ".join(["all", *LEXICON_DOMAINS])
        raise ValueError(
            f"unknown lexicon {lexicon!r}; expected one of: {known}"
        )
    filename = LEXICON_FILES[key]
    path = WORDS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"lexicon file {path} is missing; run tools/charset/export_names.py"
        )
    return _load_cached(str(path), key)


def _top_sets(result: OCRResult, charset: list[str] | None) -> list[set[str]]:
    """Per-position visual Top-K characters for the decoded path."""

    n = len(result.text)
    sets: list[set[str]] = [set() for _ in range(n)]
    if not n:
        return sets
    if (
        charset
        and result.path is not None
        and len(result.path.candidates) == n
    ):
        for i, cand in enumerate(result.path.candidates):
            scores = cand.scores
            if scores is None:
                continue
            for cid in scores.char_ids:
                if 0 <= int(cid) < len(charset):
                    sets[i].add(charset[int(cid)])
    for alt in result.alternatives:
        if len(alt) != n:
            continue
        for i, ch in enumerate(alt):
            if ch:
                sets[i].add(ch)
    return sets


def _position_topk(
    result: OCRResult,
    charset: list[str] | None,
    top_k: int = TOP_K_GUESS,
) -> list[list[str]]:
    """Ordered per-position visual Top-K characters (decoded char first).

    Uses each candidate's Goal 1 ``VisualScores.char_ids`` (already ranked)
    plus same-length decoder alternatives. The decoded character is kept at
    the front so an exact visible term always outranks a rewrite.
    """

    n = len(result.text)
    lists: list[list[str]] = [[] for _ in range(n)]
    if not n:
        return lists
    if (
        charset
        and result.path is not None
        and len(result.path.candidates) == n
    ):
        for i, cand in enumerate(result.path.candidates):
            scores = cand.scores
            if scores is None:
                continue
            for cid in scores.char_ids[:top_k]:
                if 0 <= int(cid) < len(charset):
                    ch = charset[int(cid)]
                    if ch not in lists[i]:
                        lists[i].append(ch)
    for alt in result.alternatives:
        if len(alt) != n:
            continue
        for i, ch in enumerate(alt):
            if ch and ch not in lists[i]:
                lists[i].append(ch)
    for i, ch in enumerate(result.text):
        if ch and ch not in lists[i]:
            lists[i].insert(0, ch)
    return lists


def _topk_rank_weight(
    position: list[str],
    decoded: str,
    target: str,
    top_k: int,
) -> float:
    """Visual support for replacing one decoded char by ``target``.

    Rank 0 (the decoded char) scores 1.0; the remaining visual Top-K ranks
    score 0.85 / 0.7; characters only seen in decoder alternatives score
    0.35. Characters outside the Top-3 set score 0 (never guessed).
    """

    if target == decoded:
        return 1.0
    try:
        idx = position.index(target)
    except ValueError:
        return 0.0
    if idx == 0:
        return 0.85
    if idx == 1:
        return 0.7
    if idx < top_k:
        return 0.6
    return 0.35


# ---------------------------------------------------------------------------
# ``dict`` mode: associative dictionary inference (triggered whole-word match)
# ---------------------------------------------------------------------------
#
# Scanning C2 (or a confusable of C2) must put the whole term C1C2C3 on the
# table: every term that can explain the visible string -- exact, cropped at
# either edge, confusable (substitution) or partially damaged (forced
# completion of a unique prefix/suffix) -- competes as one word hypothesis.
# The visible ``text`` is never rewritten; the winner is annotated as
# ``matched_term``/``matched_span`` and, when no hypothesis survives, the
# whole result is rejected (dictionary mode emits nothing unless a word
# matches).


def _dict_evidence(
    result: OCRResult,
    charset: list[str] | None,
    *,
    template=None,
    template_chars: Iterable[str] = (),
) -> list[dict[str, float]]:
    """Per-position visual support for the decoded path (Goal 1 Top-K).

    Each visible position maps candidate characters to a rank/confidence
    weight: rank 0 -> 1.0, rank 1 -> 0.85, rank 2 -> 0.70, rank 3 -> 0.55,
    rank 4 -> 0.40, scaled by ``0.5 + 0.5 * confidence``.
    """

    n = len(result.text)
    positions: list[dict[str, float]] = [{} for _ in range(n)]
    if not n:
        return positions
    if (
        charset
        and result.path is not None
        and len(result.path.candidates) == n
    ):
        charset_ids = {ch: i for i, ch in enumerate(charset)}
        for i, cand in enumerate(result.path.candidates):
            scores = cand.scores
            if scores is None or not scores.char_ids:
                continue
            conf = (
                float(result.chars[i].confidence)
                if i < len(result.chars)
                else 0.5
            )
            for rank, cid in enumerate(scores.char_ids[:DICT_TOP_K]):
                cid = int(cid)
                if 0 <= cid < len(charset):
                    positions[i][charset[cid]] = _dict_rank_weight(
                        rank, conf
                    )
            if template is not None and cand.segment is not None:
                # The ordinary Top-K evidence is deliberately cheap, but a
                # failed dictionary association can still lose the correct
                # character before the real-glyph fallback gets a chance.
                # Fixed-font template scores are independent evidence and
                # are added only for the characters in the candidate term
                # set supplied by the caller.
                for ch in template_chars:
                    cid = charset_ids.get(ch)
                    if cid is None:
                        continue
                    _out, score = template.match(
                        cand.segment.mask, None, {cid}
                    )
                    if score > positions[i].get(ch, 0.0):
                        positions[i][ch] = float(score)
    return positions


def _dict_rank_weight(rank: int, confidence: float) -> float:
    """Visual support of one Top-K rank, softened by the char confidence."""

    table = (1.0, 0.85, 0.70, 0.55, 0.40)
    return table[min(rank, len(table) - 1)] * (0.5 + 0.5 * confidence)


@functools.lru_cache(maxsize=32)
def _dict_char_index(lex: Lexicon) -> dict[str, set[str]]:
    """char -> set of terms containing it (candidate-term prefilter)."""

    index: dict[str, set[str]] = {}
    for term in lex.terms:
        for ch in set(term):
            index.setdefault(ch, set()).add(term)
    return index


def _dict_align(
    inner: str,
    inner_pos: list[dict[str, float]],
    term: str,
    k: int,
) -> tuple[float, int, int, float, int] | None:
    """Align ``inner`` into ``term[k:]`` with confusable/drop/gap tolerance.

    Returns ``(score, mprime, drops, matched_mean, subs)`` of the best
    alignment or ``None``: a visible char matches its term char, is
    substituted by a Top-K confusable (``DICT_SUB_PEN``, at most two), is
    dropped as a residue *anywhere* in the string (``DICT_DROP_PEN``, at
    most ``DICT_MAX_DROPS`` -- a glyph fragmented into two positions drops
    its sliver), or skips a dictionary char as a gap (``DICT_GAP_PEN``, at
    most two gaps). ``matched_mean`` is the mean evidence of the visible
    chars that actually matched (drops do not dilute it), and ``subs``
    counts the confusable substitutions.
    """

    mi = len(inner)
    best: tuple[float, int, int, float, int] | None = None
    # mprime term chars are covered: every non-dropped visible char matches
    # (or substitutes) one term position, plus ``g`` dictionary gaps.
    for mprime in range(
        max(1, mi - DICT_MAX_DROPS),
        min(len(term) - k, mi + 2) + 1,
    ):
        net = mprime - mi  # gaps - drops
        for g in range(max(0, net), min(2, net + DICT_MAX_DROPS) + 1):
            v = g - net
            if v < 0 or v > DICT_MAX_DROPS:
                continue
            inf = -1e9
            dp = np.full((mi + 1, g + 1, v + 1, 3), inf)
            dp[0, 0, 0, 0] = 0.0
            for i in range(mi):
                for gg in range(g + 1):
                    for vv in range(v + 1):
                        for ss in range(3):
                            cur = dp[i, gg, vv, ss]
                            if cur <= inf / 2:
                                continue
                            # Term position after i visible chars: drops do
                            # not consume term chars, gaps do.
                            pos = i - vv + gg
                            if pos < mprime:
                                target = term[k + pos]
                                ev = inner_pos[i].get(target)
                                if ev is not None:
                                    if target == inner[i]:
                                        dp[i + 1, gg, vv, ss] = max(
                                            dp[i + 1, gg, vv, ss], cur + ev
                                        )
                                    elif ss < 2:
                                        dp[i + 1, gg, vv, ss + 1] = max(
                                            dp[i + 1, gg, vv, ss + 1],
                                            cur + ev - DICT_SUB_PEN,
                                        )
                            if vv < v:
                                dp[i + 1, gg, vv + 1, ss] = max(
                                    dp[i + 1, gg, vv + 1, ss],
                                    cur - DICT_DROP_PEN,
                                )
                            if gg < g:
                                dp[i, gg + 1, vv, ss] = max(
                                    dp[i, gg + 1, vv, ss],
                                    cur - DICT_GAP_PEN,
                                )
            for ss in range(3):
                if dp[mi, g, v, ss] > inf / 2:
                    score = dp[mi, g, v, ss] / mprime
                    n_matched = mi - v
                    matched_mean = (
                        dp[mi, g, v, ss]
                        + DICT_DROP_PEN * v
                        + DICT_GAP_PEN * g
                        + DICT_SUB_PEN * ss
                    ) / max(1, n_matched)
                    if best is None or score > best[0]:
                        best = (score, mprime, v, matched_mean, ss)
    return best


def _dict_kind(text: str, term: str, k: int, mprime: int) -> str:
    """Goal 11/12 alignment kind of an associative hypothesis.

    In ``dict`` mode ``"exact"`` covers the whole term (possibly through a
    Top-K confusable), while the crop kinds name which term side is cut off
    by the visible crop.
    """

    if text == term:
        return "exact"
    if k == 0 and k + mprime == len(term):
        # Whole term covered by a substitution-corrected visible string.
        return "exact"
    if k == 0:
        return "prefix_crop"
    if k + mprime == len(term):
        return "suffix_crop"
    return "inner_crop"


def assoc_match(
    result: OCRResult,
    lex: Lexicon,
    charset: list[str] | None = None,
    context_terms: Iterable[str] = (),
) -> LexiconMatch | None:
    """Best associative dictionary hypothesis for one decoded result.

    Combines the regular alignment (edge residues dropped, confusables
    substituted, small gaps) with forced completion: when the matched part
    is strong and the term is the unique continuation of a scanned
    prefix/suffix, up to ``DICT_FORCE_CONF_CAP`` conflicting visible
    positions and a few unverified term characters are accepted at the
    ``DICT_FORCE_FLOOR`` evidence level. Returns ``None`` when no unique
    hypothesis clears the floors. ``context_terms`` (e.g. the neighboring
    list rows' already-resolved terms) receive a small ranking bonus that
    only breaks near-ties.
    """

    matched, _forced, _conflicts = _assoc_match(
        result, lex, charset, context_terms
    )
    return matched


def _assoc_match(
    result: OCRResult,
    lex: Lexicon,
    charset: list[str] | None = None,
    context_terms: Iterable[str] = (),
    *,
    template=None,
    all_terms: bool = False,
) -> tuple[LexiconMatch | None, bool, tuple[tuple[int, str], ...]]:
    """``assoc_match`` plus the winner's source.

    Returns ``(match, forced, conflicts)``: ``forced`` marks a winner from
    forced completion, and ``conflicts`` lists the ``(position, target
    char)`` pairs that had no visual Top-K evidence (used by the caller's
    image-verification pass).
    """

    text = normalize_text(result.text)
    if not text:
        return None, False, ()
    m = len(text)
    if all_terms:
        candidates: set[str] = set(lex.terms)
    else:
        index = _dict_char_index(lex)
        candidates = set()
        for ch in set(text):
            candidates |= index.get(ch, set())
    template_chars: set[str] = set()
    if template is not None:
        for term in candidates:
            template_chars.update(term)
    positions = _dict_evidence(
        result,
        charset,
        template=template,
        template_chars=template_chars,
    )
    if m != len(positions):
        return None, False, ()
    context: frozenset[str] = frozenset(context_terms)

    # term -> list of (score, k, mprime, drops, forced, subs) hypotheses.
    hypotheses: dict[str, list[tuple[float, int, int, int, int, int]]] = {}

    for term in candidates:
        n = len(term)
        rows: list[tuple[float, int, int, int, int, int]] = []
        # --- regular alignment: confusables substituted, residues dropped
        # anywhere (a glyph fragmented into two positions drops its
        # sliver), small dictionary gaps ---
        for k in range(0, n):
            aligned = _dict_align(text, positions, term, k)
            if aligned is None:
                continue
            score, mprime, drops, matched_mean, subs = aligned
            # A single aligned char must not free-ride the middle of a
            # longer term (e.g. 灶 alone can never explain 女灶神), but a
            # prefix/suffix crop of one char stays valid.
            if m - drops < 2 and k > 0 and k + mprime < n:
                continue
            # Most of the visible string must be explained: residues may be
            # dropped, but the matched part must stay the majority, so a
            # lucky 2-char overlap cannot claim a 4-char garbage string.
            # Exactly half is accepted only when every matched char is
            # exact (a pure split-garbage crop like 灶7 -> Z17).
            frac = (m - drops) / m
            if frac < 0.5 or (frac == 0.5 and subs > 0):
                continue
            # Drops may explain residues, never weak matches: the chars
            # that DID match must be visually strong on their own.
            if matched_mean < DICT_MATCH_MEAN_MIN:
                continue
            # Term shorter than the *effective* visible length (drops are
            # already penalized per character, so they do not count twice).
            if n < m - drops:
                score -= DICT_LEN_PEN * (m - drops - n)
            if template is not None and drops:
                # In this recovery pass a fully covered crop is more
                # trustworthy than a longer term that explains only a
                # substring and discards one visible glyph (e.g. the
                # ``纳尔坡`` vs ``纳尔逊`` ambiguity).
                score -= 0.10 * drops
            if template is not None and not (
                k == 0 and mprime == n and drops == 0
            ) and (k == 0 or k + mprime == n):
                # A fixed-font UI crop is much more often clipped at the
                # left/right edge than cut out of the middle of a word.
                # Use that geometry only in the template recovery pass and
                # only as a near-tie prior; the normal Top-K association is
                # byte-for-byte unchanged.
                score += 0.03
            rows.append((score, k, mprime, drops, 0, subs, ()))
        # --- forced completion: prefix/suffix window, conflict-aware ---
        for k in (0, max(0, n - m)):
            matched = 0.0
            n_matched = 0
            conflicts = 0
            for i in range(min(m, n - k)):
                ev = positions[i].get(term[k + i])
                if ev is not None:
                    matched += ev - (
                        DICT_SUB_PEN if term[k + i] != text[i] else 0.0
                    )
                    n_matched += 1
                else:
                    conflicts += 1
            if n_matched == 0:
                continue
            unseen = n - (k + min(m, n - k))
            mean_ok = matched / n_matched >= DICT_FORCE_NEED_MATCH
            conf_ok = conflicts <= DICT_FORCE_CONF_CAP
            # A forced completion must not free-ride: one matched char may
            # only close at most one unseen char, two matched chars at most
            # two, and four matched chars up to four (long prefix crops).
            unseen_ok = (
                (unseen <= 1)
                or (unseen <= 2 and n_matched >= 2)
                or (n_matched >= 4 and unseen <= 4)
            )
            if mean_ok and conf_ok and unseen_ok:
                score = (
                    matched + DICT_FORCE_FLOOR * (conflicts + unseen)
                ) / n
                score -= 0.02 * conflicts
                if n < m:
                    score -= DICT_LEN_PEN * (m - n)
                confs = tuple(
                    (i, term[k + i])
                    for i in range(min(m, n - k))
                    if term[k + i] not in positions[i]
                )
                rows.append((score, k, min(m, n - k), 0, 1, -1, confs))
        if rows:
            hypotheses[term] = rows

    if not hypotheses:
        return None, False, ()

    def _exact_whole(hyp_rows, term: str) -> bool:
        """A hypothesis where the visible text IS the whole term."""
        n = len(term)
        if text != term:
            return False
        return any(
            h[1] == 0 and h[2] == n and h[3] == 0 for h in hyp_rows
        )

    def _drop_only(hyp_rows, term: str) -> float:
        """Best score of a pure-residue hypothesis: the visible string is
        the term minus dropped residues (no substitution, no gap)."""
        n = len(term)
        best_d = -1.0
        for h in hyp_rows:
            s_h, kk, mp, dd, ff, ss, _ = h
            gaps = mp - (m - dd)
            if not ff and ss == 0 and gaps == 0 and kk == 0 and mp == n:
                best_d = max(best_d, s_h)
        return best_d

    # A pure-residue explanation (visible ⊆ term, no rewrite) is the safest
    # inference: when one exists near the top, prefer the shortest such
    # term, exactly like the matcher's shorter-term rule for crops.
    # 波*特 explains 波特 even when a longer term scores the same.
    top_overall = max(
        max(h[0] for h in hypotheses[t])
        + (DICT_CONTEXT_BONUS if t in context else 0.0)
        for t in hypotheses
    )
    drop_only_candidates = [
        (t, s) for t in hypotheses
        if (s := _drop_only(hypotheses[t], t)) >= top_overall - 0.06
    ]
    if drop_only_candidates:
        best_term = min(
            drop_only_candidates, key=lambda ts: (len(ts[0]), -ts[1])
        )[0]
    else:
        # Term ranking uses each term's strongest hypothesis; the
        # uniqueness margin compares term maxima, so a tie cannot hide
        # behind one term's preferred output span. On a score tie the
        # exact whole-term match wins: two different terms can never both
        # be exact. Caller-supplied context terms (neighboring resolved
        # rows) get a small bonus that only breaks near-ties.
        best_term = max(
            hypotheses,
            key=lambda t: (
                max(h[0] for h in hypotheses[t]) + (
                    DICT_CONTEXT_BONUS if t in context else 0.0
                ),
                _exact_whole(hypotheses[t], t),
            ),
        )
    rows = hypotheses[best_term]
    best_score = max(h[0] for h in rows)
    # Same-term preference: within a generous window prefer the hypothesis
    # that is not forced, drops fewer visible chars and covers more of the
    # term, so a whole-term confusable explanation wins over discarding
    # the residue.
    rows = [h for h in rows if h[0] >= best_score - 0.15]
    score, k, mprime, drops, forced, subs, conflicts = min(
        rows, key=lambda h: (h[3], -h[2], -h[0])
    )
    second = max(
        (max(h[0] for h in hypotheses[t]))
        for t in hypotheses
        if t != best_term
    ) if len(hypotheses) > 1 else -1.0
    floor = DICT_MIN_SCORE_SINGLE if len(best_term) == 1 else DICT_MIN_SCORE
    if score < floor:
        return None, False, ()
    # An exact whole-term winner needs no uniqueness margin (no rewrite is
    # involved; e.g. 约克 exact vs the 约克城 prefix crop at equal score).
    # The same holds when the visible string is the term minus pure
    # residues (subs == 0, no gaps): 波*特 explains 波特, and 波特兰's
    # prefix crop at the same score must not veto it.
    gaps = mprime - (m - drops)
    drop_only = (
        not forced and subs == 0 and gaps == 0
        and k == 0 and mprime == len(best_term)
    )
    exact_win = (
        k == 0 and mprime == len(best_term) and drops == 0 and text == best_term
    ) or drop_only
    strong_template_win = False
    if (
        template is not None
        and not exact_win
        and k == 0
        and mprime == len(best_term)
        and drops == 0
        and mprime == m
        and result.path is not None
        and len(result.path.candidates) == m
    ):
        cid_map = {ch: i for i, ch in enumerate(charset or ())}
        template_scores: list[float] = []
        for cand, ch in zip(result.path.candidates, best_term):
            cid = cid_map.get(ch)
            if cid is None or cand.segment is None:
                template_scores = []
                break
            _out, raw = template.match(cand.segment.mask, None, {cid})
            template_scores.append(float(raw))
        strong_template_win = bool(template_scores) and min(template_scores) >= (
            DICT_TEMPLATE_STRONG_MIN
        )
    if (
        not exact_win
        and not strong_template_win
        and second > best_score - DICT_UNIQUE_MARGIN
    ):
        return None, False, ()
    return (
        _match(
            best_term,
            (k, k + mprime),
            float(score),
            _dict_kind(text, best_term, k, mprime),
            text,
            (0, m),
        ),
        bool(forced),
        conflicts,
    )


def _ncc_assoc_match(
    result: OCRResult,
    lex: Lexicon,
    bank: _realglyphs.RealGlyphBank,
    soft_glyphs: list[np.ndarray],
    *,
    all_terms: bool = False,
) -> LexiconMatch | None:
    """NCC arbitration over real-glyph prototypes (same gates as assoc).

    Runs the associative alignment a second time with per-position evidence
    from real-game prototype NCC instead of classifier Top-K, so game
    renders are compared with game renders. Used only when the Top-K based
    match produced nothing, so existing hits are never re-arbitrated.

    ``all_terms`` is a failure-only fallback for damaged classifier text.
    The normal path prefilters terms by characters in the decoded text; when
    the decoder has confused every character (for example ``塞甫`` for the
    visible ``鹞鹰`` crop), that prefilter removes the correct term before
    the real-glyph evidence gets a chance to help.  Scanning the complete
    ship lexicon is intentionally opt-in because it is substantially more
    expensive than the normal indexed path.
    """

    text = normalize_text(result.text)
    if not text:
        return None
    m = len(text)
    if m != len(soft_glyphs):
        return None

    if all_terms:
        candidates = set(lex.terms)
    else:
        index = _dict_char_index(lex)
        candidates = set()
        for ch in set(text):
            candidates |= index.get(ch, set())

    # NCC evidence is a pure function of (glyph position, character), and
    # dozens of candidate terms share the same visible characters (common
    # chars like 尔/维 put ~100 terms on the table for one crop). Without
    # the cache every term re-scans the same prototype stacks, which turned
    # the 乌戈里尼 crop's arbitration into a ~116 ms scan of ~3900 repeated
    # NCC evaluations instead of ~200 unique ones.
    evidence: dict[tuple[int, str], float | None] = {}

    def glyph_evidence(i: int, ch: str) -> float | None:
        key = (i, ch)
        if key not in evidence:
            evidence[key] = bank.best_evidence(soft_glyphs[i], ch)
        return evidence[key]

    best: tuple[float, str, int, int, int] | None = None
    second = -1.0
    for term in candidates:
        n = len(term)
        term_chars = set(term)
        positions: list[dict[str, float]] = []
        for i in range(m):
            d: dict[str, float] = {}
            for ch in term_chars:
                ev = glyph_evidence(i, ch)
                if ev is not None:
                    d[ch] = ev
            positions.append(d)
        for k in range(0, n):
            aligned = _dict_align(text, positions, term, k)
            if aligned is None:
                continue
            score, mprime, drops, matched_mean, subs = aligned
            if m - drops < 2 and k > 0 and k + mprime < n:
                continue
            frac = (m - drops) / m
            if frac < 0.5 or (frac == 0.5 and subs > 0):
                continue
            if matched_mean < _realglyphs.NCC_MEAN_FLOOR:
                continue
            if n < m - drops:
                score -= DICT_LEN_PEN * (m - drops - n)
            if best is None or score > best[0]:
                if best is not None and best[1] != term:
                    second = max(second, best[0])
                best = (score, term, k, mprime, drops)
            elif term != best[1]:
                second = max(second, score)

    if best is None:
        return None
    score, term, k, mprime, drops = best
    floor = DICT_MIN_SCORE_SINGLE if len(term) == 1 else DICT_MIN_SCORE
    if score < floor:
        return None
    if second > score - DICT_UNIQUE_MARGIN:
        return None
    return _match(
        term,
        (k, k + mprime),
        float(score),
        _dict_kind(text, term, k, mprime),
        text,
        (0, m),
    )


def _promote_term_alternative(
    result: OCRResult,
    lex: Lexicon,
) -> OCRResult | None:
    """Promote the first decoder alternative that is an exact lexicon term.

    Only used when the visible text is *not* already a dictionary term, so a
    clear visible term (e.g. ``狮``) is never replaced by a visually
    confused alternative (e.g. ``蜩``). The promoted alternative may have a
    different segmentation than the decoded text (``灶7`` -> ``Z17``), which
    the per-position rewrite below cannot handle.
    """

    text = normalize_text(result.text)
    if not text or text in lex:
        return None
    for rank, alt in enumerate(result.alternatives):
        term = normalize_text(alt)
        if not term or term == text or term not in lex:
            continue
        n = len(term)
        confidence = max(0.5, 0.95 - 0.05 * rank)
        chars = tuple(
            CharResult(
                char=ch,
                x=i,
                y=0,
                w=1,
                h=1,
                confidence=confidence,
            )
            for i, ch in enumerate(term)
        )
        match = LexiconMatch(
            term=term,
            span=(0, n),
            confidence=confidence,
            score=confidence,
            mode="topk",
            text=term,
            text_span=(0, n),
            kind="exact",
        )
        return replace(
            result,
            text=term,
            confidence=confidence,
            chars=chars,
            matched_term=term,
            matched_span=(0, n),
            lexicon_match=match,
        )
    return None


def _topk_lexicon_guess(
    result: OCRResult,
    lex: Lexicon,
    charset: list[str] | None,
) -> OCRResult | None:
    """Rewrite from the per-position visual Top-3 when a term fits uniquely.

    Looks for dictionary terms with the same length as the visible text
    whose every character is supported by that position's Top-3 visual
    alternatives. The best-supported term wins; two differently-worded
    terms within ``TOP_K_GUESS_UNIQUE_MARGIN`` keep the visible text so the
    guess never invents an arbitrary name (Goal 15's non-unique rule).
    """

    text = normalize_text(result.text)
    if not text or len(result.chars) != len(text) or text in lex:
        return None
    topk = _position_topk(result, charset)
    candidates: list[tuple[float, int, str, list[tuple[int, str]]]] = []
    for term in lex.terms:
        if len(term) != len(text):
            continue
        changes: list[tuple[int, str]] = []
        weights: list[float] = []
        ok = True
        for i, (target, original) in enumerate(zip(term, text)):
            weight = _topk_rank_weight(topk[i], original, target, TOP_K_GUESS)
            if weight <= 0.0:
                ok = False
                break
            weights.append(weight)
            if target != original:
                changes.append((i, target))
        if not ok or not changes:
            continue
        score = float(np.mean(weights))
        if score >= TOP_K_GUESS_MIN_SCORE:
            candidates.append((score, len(changes), term, changes))

    if not candidates:
        return None
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    best_score, _best_changes_n, best_term, best_changes = candidates[0]
    if len(candidates) > 1:
        second_score, _second_n, second_term, _ = candidates[1]
        if (
            second_term != best_term
            and (best_score - second_score) < TOP_K_GUESS_UNIQUE_MARGIN
        ):
            return None

    new_chars = list(result.chars)
    for i, target in best_changes:
        old = new_chars[i]
        new_chars[i] = CharResult(
            char=target,
            x=old.x,
            y=old.y,
            w=old.w,
            h=old.h,
            confidence=_topk_rank_weight(
                topk[i], result.text[i], target, TOP_K_GUESS
            ),
        )
    confidence = (
        float(np.mean([c.confidence for c in new_chars]))
        if new_chars
        else 0.0
    )
    n = len(best_term)
    match = LexiconMatch(
        term=best_term,
        span=(0, n),
        confidence=best_score,
        score=best_score,
        mode="topk",
        text=best_term,
        text_span=(0, n),
        kind="exact",
    )
    return replace(
        result,
        text=best_term,
        confidence=confidence,
        chars=tuple(new_chars),
        matched_term=best_term,
        matched_span=(0, n),
        lexicon_match=match,
    )


def _lexicon_correction(
    result: OCRResult,
    lexicon: Lexicon,
    mode: str,
    charset: list[str] | None,
    prefer_threshold: float,
    correction_min_score: float,
    correction_unique_margin: float,
) -> OCRResult | None:
    """Build a corrected exact-term result when visual evidence permits.

    A correction is only emitted when the visual evidence for the changed
    characters is genuinely uncertain (``confidence < prefer_threshold``)
    and the lexicon target is already in that character's visual Top-K.
    When two different dictionary terms are both plausible corrections and
    their scores are close (within ``correction_unique_margin``) the match
    is not unique, so the visible OCR text is kept (Goal 15).
    """

    text = normalize_text(result.text)
    if not text or len(result.chars) != len(text):
        return None
    top_sets = _top_sets(result, charset)
    candidates: list[tuple[float, str, list[tuple[int, str]]]] = []

    for term in lexicon.terms:
        if len(term) != len(text):
            continue
        changes: list[tuple[int, str]] = []
        ok = True
        penalty = 0.0
        for i, (target, original) in enumerate(zip(term, text)):
            if target == original:
                continue
            conf = float(result.chars[i].confidence)
            if conf >= prefer_threshold or target not in top_sets[i]:
                ok = False
                break
            changes.append((i, target))
            penalty += max(0.0, (prefer_threshold - conf) / prefer_threshold)
        if not ok or not changes:
            continue
        score = 1.0 - min(0.6, 0.5 * penalty)
        if score >= correction_min_score:
            candidates.append((score, term, changes))

    if not candidates:
        return None
    candidates.sort(key=lambda row: (-row[0], row[1]))
    best_score, best_term, best_changes = candidates[0]
    if len(candidates) > 1:
        second_score, second_term, _ = candidates[1]
        if (
            second_term != best_term
            and (best_score - second_score) < correction_unique_margin
        ):
            return None

    new_chars = list(result.chars)
    for i, target in best_changes:
        old = new_chars[i]
        new_chars[i] = CharResult(
            char=target,
            x=old.x,
            y=old.y,
            w=old.w,
            h=old.h,
            confidence=old.confidence,
        )
    confidence = (
        float(np.mean([c.confidence for c in new_chars]))
        if new_chars
        else 0.0
    )
    match = LexiconMatch(
        term=best_term,
        span=(0, len(best_term)),
        confidence=best_score,
        score=best_score,
        mode=mode,
        text=best_term,
        text_span=(0, len(best_term)),
        kind="exact",
    )
    return replace(
        result,
        text=best_term,
        confidence=confidence,
        chars=tuple(new_chars),
        matched_term=best_term,
        matched_span=(0, len(best_term)),
        lexicon_match=match,
    )


def apply_lexicon(
    result: OCRResult,
    lexicon: str | Path | Lexicon | None,
    mode: str | None = None,
    *,
    charset: list[str] | None = None,
    context_terms: Iterable[str] = (),
    template=None,
    soft_glyphs: list[np.ndarray] | None = None,
    real_bank: _realglyphs.RealGlyphBank | None = None,
    prefer_threshold: float = 0.9,
    correction_min_score: float = 0.0,
    correction_unique_margin: float = 0.02,
) -> OCRResult:
    """Apply the Goal 11 lexicon modes to an OCR result.

    ``mode`` may be ``"none"``, ``"prefer"``, ``"topk"``, ``"strict"`` or
    ``"dict"``. With ``"prefer"`` the best dictionary match is attached to
    the result and a visually uncertain character is corrected only when the
    dictionary target is already among that character's Top-K alternatives
    and the correction is unique (a second, differently-worded correction
    within ``correction_unique_margin`` keeps the visible OCR text, Goal
    15). With ``"topk"`` a non-term visible text is rewritten from the
    lexicon when an exact term is the first supported decoder alternative or
    is fully covered by each position's visual Top-3 (unique match only).
    With ``"strict"`` only an exact dictionary term is accepted; anything
    else is rejected as empty output. With ``"dict"`` the decoded result is
    matched associatively against the dictionary (exact / cropped /
    confusable / uniquely forced word hypotheses, see :func:`assoc_match`);
    the visible text is never rewritten, the winning word is annotated as
    ``matched_term`` and a result without any word hypothesis is rejected
    as empty output ("return nothing unless a word matches").
    """

    if mode is None:
        mode = "prefer" if lexicon is not None else "none"
    normalized_mode = mode.strip().lower()
    if normalized_mode not in LEXICON_MODES:
        raise ValueError(
            f"lexicon_mode {mode!r} is not supported; "
            "use 'none', 'prefer', 'topk', 'strict' or 'dict'"
        )
    if normalized_mode == "none" or lexicon is None:
        return result
    lex = load_lexicon(lexicon) if not isinstance(lexicon, Lexicon) else lexicon

    # ``dict`` mode performs its own associative alignment below. It does
    # not consume the ordinary ``Lexicon.match`` result, so avoid scanning
    # the entire word list once before entering that path.
    if normalized_mode == "dict":
        matched, forced, conflicts = _assoc_match(
            result, lex, charset, context_terms
        )
        assoc_matched = matched
        # 命中后回图验证: a forced-completion hit must be template-
        # plausible on the crop itself -- every conflict position is
        # re-matched with the registered font's templates, and a target
        # whose template score is too low rejects the whole hit.
        if (
            matched is not None
            and forced
            and conflicts
            and template is not None
            and charset
        ):
            cid_map = {c: i for i, c in enumerate(charset)}
            for i, ch in conflicts:
                if i >= len(result.path.candidates):
                    continue
                mask = result.path.candidates[i].segment.mask
                target = cid_map.get(ch)
                if target is None:
                    continue
                _out, tpl_score = template.match(mask, None, {target})
                if tpl_score < DICT_VERIFY_TEMPLATE_MIN:
                    matched = None
                    break
        if matched is None:
            # A forced hypothesis rejected by image verification is still a
            # failure for the purpose of the independent fallback passes.
            assoc_matched = None
        # Top-K 无解时, 用真实字形库做 NCC 仲裁 (游戏渲染 vs 游戏渲染)。
        if matched is None and real_bank is not None and soft_glyphs:
            matched = _ncc_assoc_match(result, lex, real_bank, soft_glyphs)
            if matched is None:
                # The character-index prefilter is deliberately cheap, but
                # a fully confused OCR string can hide the correct term
                # from it.  Retry only after the indexed NCC path fails.
                matched = _ncc_assoc_match(
                    result,
                    lex,
                    real_bank,
                    soft_glyphs,
                    all_terms=True,
                )
        if template is not None and (
            assoc_matched is None
            or (
                matched is not None
                and matched.kind != "exact"
                and matched.score < DICT_TEMPLATE_RERANK_MAX
            )
        ):
            # Failure-only template recovery.  The regular association is
            # intentionally Top-K driven; when every target character is
            # just outside that list, the registered fixed-font templates
            # can still provide an independent per-position signal.  Keep
            # the indexed pass bounded by terms sharing at least one visible
            # character, then use the full lexicon only for fully confused
            # crops. A strong real-glyph result remains authoritative.
            template_indexed, _, _ = _assoc_match(
                result,
                lex,
                charset,
                context_terms,
                template=template,
            )
            template_match = template_indexed
            if template_match is None or template_match.kind != "exact":
                template_all, _, _ = _assoc_match(
                    result,
                    lex,
                    charset,
                    context_terms,
                    template=template,
                    all_terms=True,
                )
                if (
                    template_all is not None
                    and (
                        template_match is None
                        or template_all.score > template_match.score
                    )
                ):
                    template_match = template_all
            if template_match is not None:
                current_score = matched.score if matched is not None else -1.0
                template_score = template_match.score
                if (
                    matched is None
                    or template_score >= current_score + 0.05
                ):
                    matched = template_match
        if matched is None:
            return replace(
                result,
                text="",
                confidence=0.0,
                chars=(),
                matched_term=None,
                matched_span=None,
                lexicon_match=None,
                alternatives=(),
            )
        matched = replace(matched, mode="dict")
        return replace(
            result,
            matched_term=matched.term,
            matched_span=matched.span,
            lexicon_match=matched,
        )

    text = normalize_text(result.text)
    matches = lex.match(text)
    top = matches[0] if matches else None

    if normalized_mode == "prefer":
        if top is not None:
            top = replace(top, mode="prefer")
            out = replace(
                result,
                matched_term=top.term,
                matched_span=top.span,
                lexicon_match=top,
            )
        else:
            out = replace(
                result,
                matched_term=None,
                matched_span=None,
                lexicon_match=None,
            )
        if top is None or top.kind != "exact":
            corrected = _lexicon_correction(
                result,
                lex,
                "prefer",
                charset,
                prefer_threshold,
                correction_min_score,
                correction_unique_margin,
            )
            if corrected is not None:
                return corrected
        return out

    if normalized_mode == "topk":
        matches = lex.match(text)
        top = matches[0] if matches else None
        if top is not None:
            top = replace(top, mode="topk")
            out = replace(
                result,
                matched_term=top.term,
                matched_span=top.span,
                lexicon_match=top,
            )
        else:
            out = replace(
                result,
                matched_term=None,
                matched_span=None,
                lexicon_match=None,
            )
        promoted = _promote_term_alternative(result, lex)
        if promoted is not None:
            return promoted
        guessed = _topk_lexicon_guess(result, lex, charset)
        if guessed is not None:
            return guessed
        return out

    # strict
    if top is not None and top.kind == "exact":
        top = replace(top, mode="strict")
        return replace(
            result,
            matched_term=top.term,
            matched_span=top.span,
            lexicon_match=top,
        )
    corrected = _lexicon_correction(
        result,
        lex,
        "strict",
        charset,
        prefer_threshold,
        correction_min_score,
        correction_unique_margin,
    )
    if corrected is not None:
        return corrected
    return replace(
        result,
        text="",
        confidence=0.0,
        chars=(),
        matched_term=None,
        matched_span=None,
        lexicon_match=None,
        alternatives=(),
    )
