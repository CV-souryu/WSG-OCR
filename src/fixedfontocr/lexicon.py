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

LEXICON_MODES = ("none", "prefer", "topk", "strict")
TOP_K_GUESS = 3
TOP_K_GUESS_MIN_SCORE = 0.6
TOP_K_GUESS_UNIQUE_MARGIN = 0.02


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
    prefer_threshold: float = 0.9,
    correction_min_score: float = 0.0,
    correction_unique_margin: float = 0.02,
) -> OCRResult:
    """Apply the Goal 11 lexicon modes to an OCR result.

    ``mode`` may be ``"none"``, ``"prefer"``, ``"topk"`` or ``"strict"``. With
    ``"prefer"`` the best dictionary match is attached to the result and a
    visually uncertain character is corrected only when the dictionary
    target is already among that character's Top-K alternatives and the
    correction is unique (a second, differently-worded correction within
    ``correction_unique_margin`` keeps the visible OCR text, Goal 15).
    With ``"topk"`` a non-term visible text is rewritten from the lexicon
    when an exact term is the first supported decoder alternative or is
    fully covered by each position's visual Top-3 (unique match only).
    With ``"strict"`` only an exact dictionary term is accepted; anything
    else is rejected as empty output.
    """

    if mode is None:
        mode = "prefer" if lexicon is not None else "none"
    normalized_mode = mode.strip().lower()
    if normalized_mode not in LEXICON_MODES:
        raise ValueError(
            f"lexicon_mode {mode!r} is not supported; use 'none', 'prefer' or 'strict'"
        )
    if normalized_mode == "none" or lexicon is None:
        return result
    lex = load_lexicon(lexicon) if not isinstance(lexicon, Lexicon) else lexicon

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
