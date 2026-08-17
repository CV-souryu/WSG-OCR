"""Oracle lattice recall: is the ground-truth segmentation in the lattice?

The oracle ignores all scores: it builds the same candidate lattice the
production pipeline uses and checks whether a path covering the per-char
ground-truth ink boxes exists.  If it does not, no classifier can recover
the correct text -- the failure is in segmentation (candidate generation /
pruning).  If it does but the decoder picked another path, the failure is
in classification / decoding.

This file tests the metric itself with deterministic synthetic lines:

* normal rendered strings -> oracle path exists;
* fully-glued glyph pairs (no vertical valley at the boundary) -> oracle
  path missing with a straddled-boundary diagnostic naming the missing
  cut position (the ``LV`` / ``4+3`` / ``U-`` failure mode);
* thin-valley glued pairs -> oracle path exists (the current valley
  heuristic handles these);
* a merge pruned by the gap rule -> ``missing_candidate``.
"""

from __future__ import annotations

import numpy as np
import pytest

from fixedfontocr.oracle import (
    COVERAGE_THRESHOLD,
    GroundTruthBox,
    GroundTruthLine,
    evaluate_oracle,
    format_report,
    glue_ground_truth,
    oracle_lattice_recall,
    render_ground_truth,
)
from fixedfontocr.segmentation import connected_components
from fixedfontocr.types import default_profile

NORMAL_STRINGS = [
    "潜甲",
    "鲃鱼。",
    "小",
    "巴尔的摩",
    "Z17",
    "LV.40",
    "T-23",
    "U-",
    "4+3",
    "1000",
    "岛风",
    "甲申",
]


def _pair(font_path, a: str, b: str, bridge: str) -> GroundTruthLine:
    la = render_ground_truth(a, font_path, font_size=32)
    lb = render_ground_truth(b, font_path, font_size=32)
    return glue_ground_truth(la, lb, bridge=bridge)


# ---------------------------------------------------------------------------
# Ground-truth renderer
# ---------------------------------------------------------------------------


def test_render_boxes_match_standalone_glyph_masks(font_path):
    """Per-char GT boxes are pixel-exact against standalone tight renders.

    Each GT box's region of the line mask must be exactly the standalone
    tight mask of that character (same font size, threshold and integer
    origin), and the union of all boxes must cover every ink pixel.
    """

    for text in NORMAL_STRINGS:
        gt = render_ground_truth(text, font_path, font_size=32)
        covered = np.zeros_like(gt.mask)
        for i, box in enumerate(gt.boxes):
            sub = gt.mask[box.y0 : box.y1, box.x0 : box.x1]
            ref = render_glyph_mask_ref(gt.text[i], font_path)
            assert sub.shape == ref.shape, (
                f"{text!r} char {i} {gt.text[i]!r}: box {sub.shape} != "
                f"standalone {ref.shape}"
            )
            assert np.array_equal(sub, ref), (
                f"{text!r} char {i} {gt.text[i]!r}: raster mismatch"
            )
            covered[box.y0 : box.y1, box.x0 : box.x1] = True
        assert not np.any(gt.mask & ~covered), f"{text!r}: ink outside boxes"


def render_glyph_mask_ref(char: str, font_path) -> np.ndarray:
    from fixedfontocr.oracle import render_glyph_mask

    return render_glyph_mask(char, font_path, font_size=32)


def test_render_boxes_reject_bad_layout(font_path):
    """A GT box with no ink is rejected (would poison the metric)."""

    gt = render_ground_truth("甲", font_path, font_size=32)
    with pytest.raises(ValueError, match="no ink"):
        GroundTruthLine(
            text=gt.text,
            mask=gt.mask,
            boxes=(
                GroundTruthBox("甲", 0, 0, 3, 3),  # empty corner
            ),
        )


# ---------------------------------------------------------------------------
# Oracle recall on normal renders
# ---------------------------------------------------------------------------


def test_oracle_recall_on_normal_renders(font_path):
    profile = default_profile()
    for text in NORMAL_STRINGS:
        gt = render_ground_truth(text, font_path, font_size=32)
        res = oracle_lattice_recall(gt.line, profile, gt.boxes)
        assert res.recall, (
            f"{text!r}: oracle path missing (reason={res.reason!r})"
        )
        assert res.reason == ""
        # The oracle path has one candidate per GT character.
        assert len(res.path) == len(text)
        # Atom order matches text order (sanity on the walk).
        spans = [c.atom_span for c in res.path]
        assert spans[0][0] == 0
        assert all(
            spans[k][1] == spans[k + 1][0] for k in range(len(spans) - 1)
        )
        assert spans[-1][1] == res.n_atoms


def test_oracle_recall_ignores_scores(font_path):
    """The metric must not require a scorer: no score is ever consulted."""

    gt = render_ground_truth("潜甲", font_path, font_size=32)
    res = oracle_lattice_recall(gt.line, default_profile(), gt.boxes)
    assert res.recall
    # Every candidate on the oracle path is unscored.
    assert all(c.score is None and c.scores is None for c in res.path)


# ---------------------------------------------------------------------------
# Fully-glued pairs: the LV / 4+3 / U- failure mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left,right,bridge",
    [
        ("甲", "申", "full"),
        ("4", "3", "full"),
        ("L", "V", "full"),
        ("U", "-", "full"),
    ],
    ids=["jia-shen", "4-3", "L-V", "U-dash"],
)
def test_fully_glued_pairs_have_no_oracle_path(font_path, left, right, bridge):
    """A full-height connector has no valley: no split, oracle path missing.

    This is the diagnostic the metric exists for: the current valley-based
    splitter cannot cut the boundary, so the GT segmentation never enters
    the lattice and no classifier can recover it.  The result must name the
    missing cut position (the straddled boundary).
    """

    gt = _pair(font_path, left, right, bridge)
    # The construction really is one connected component.
    comps = connected_components(gt.line)
    assert len(comps) == 1, "full bridge must produce one component"

    res = oracle_lattice_recall(gt.line, default_profile(), gt.boxes)
    assert not res.recall
    assert res.reason == "unassigned_atom"
    assert len(res.straddled_boundaries) == 1
    l_idx, r_idx, x = res.straddled_boundaries[0]
    assert (l_idx, r_idx) == (0, 1)
    assert res.missing_chars == ()


def test_thin_valley_connector_keeps_oracle_path(font_path):
    """A 1 px connector is a real valley: the split exists, oracle passes."""

    for left, right in [("甲", "申"), ("4", "3")]:
        gt = _pair(font_path, left, right, "row")
        comps = connected_components(gt.line)
        assert len(comps) == 1, "row bridge must produce one component"
        res = oracle_lattice_recall(gt.line, default_profile(), gt.boxes)
        assert res.recall, (
            f"{left}+{right} row-bridge: oracle missing (reason={res.reason!r})"
        )
        assert len(res.path) == 2


def test_separated_pair_oracle_path(font_path):
    """Without a bridge the pair is two components: trivially covered."""

    gt = _pair(font_path, "甲", "申", "none")
    assert len(connected_components(gt.line)) == 2
    res = oracle_lattice_recall(gt.line, default_profile(), gt.boxes)
    assert res.recall
    assert len(res.path) == 2


# ---------------------------------------------------------------------------
# missing_candidate: the needed merge was pruned
# ---------------------------------------------------------------------------


def test_pruned_merge_reports_missing_candidate(font_path):
    """Two components 15 px apart cannot be one glyph (gap rule): the GT
    single-character candidate was pruned -> missing_candidate."""

    bar1 = render_ground_truth("1", font_path, font_size=32)
    bar2 = render_ground_truth("1", font_path, font_size=32)
    w1 = bar1.mask.shape[1]
    h = max(bar1.mask.shape[0], bar2.mask.shape[0])
    gap = 15
    mask = np.zeros((h, w1 + gap + bar2.mask.shape[1]), dtype=bool)
    mask[: bar1.mask.shape[0], :w1] = bar1.mask
    mask[: bar2.mask.shape[0], w1 + gap :] = bar2.mask
    box = GroundTruthBox("X", 0, 0, mask.shape[1], h)
    gt = GroundTruthLine(text="X", mask=mask, boxes=(box,))

    res = oracle_lattice_recall(gt.line, default_profile(), gt.boxes)
    assert not res.recall
    assert res.reason == "missing_candidate"
    assert res.missing_chars == (0,)
    # Best any candidate can do is cover one of the two bars (~half).
    assert res.best_coverage[0] < COVERAGE_THRESHOLD
    assert res.best_coverage[0] >= 0.4
    assert res.atom_assignments == (0, 0)


def test_atom_interleaving_reports_no_path(font_path):
    """Pathological x-interleaved boxes: every atom is assignable and each
    char has a valid candidate, but they cannot tile in text order."""

    mask = np.zeros((13, 10), dtype=bool)
    mask[0:3, 5:8] = True  # dot (char A)
    mask[10:13, 4:9] = True  # bar (char B) sorts BEFORE the dot
    gt = GroundTruthLine(
        text="AB",
        mask=mask,
        boxes=(
            GroundTruthBox("A", 5, 0, 8, 3),
            GroundTruthBox("B", 4, 10, 9, 13),
        ),
    )
    res = oracle_lattice_recall(gt.line, default_profile(), gt.boxes)
    assert not res.recall
    assert res.reason == "no_path"


# ---------------------------------------------------------------------------
# Attribution: segmentation vs decoding errors
# ---------------------------------------------------------------------------


def test_evaluate_attributes_segmentation_vs_decoding(font_path):
    profile = default_profile()
    ok_line = render_ground_truth("潜甲", font_path, font_size=32)
    seg_line = _pair(font_path, "甲", "申", "full")  # oracle missing

    # Correct decoder: only the segmentation failure remains.
    report = evaluate_oracle(
        [ok_line, seg_line],
        profile,
        decode_fn=lambda gt: gt.text,
    )
    assert report.total == 2
    assert report.oracle_recall == 0.5
    assert report.status_counts == {"ok": 1, "segmentation": 1}

    # Wrong decoder on the oracle-ok line: attributed to decoding.
    report = evaluate_oracle(
        [ok_line, seg_line],
        profile,
        decode_fn=lambda gt: "!!",
    )
    assert report.status_counts == {"decoding": 1, "segmentation": 1}

    # Without a decoder the oracle-ok lines are simply "ok".
    report = evaluate_oracle([ok_line], profile)
    assert report.status_counts == {"ok": 1}


def test_format_report_smoke(font_path):
    profile = default_profile()
    report = evaluate_oracle(
        [render_ground_truth("潜甲", font_path, font_size=32)],
        profile,
    )
    text = format_report(report)
    assert "oracle lattice recall: 100.0%" in text
    assert "ok" in text


def test_report_names_missing_cut_position(font_path):
    """The report must say *where* the missing cut is (guides the fix)."""

    gt = _pair(font_path, "甲", "申", "full")
    report = evaluate_oracle([gt], default_profile())
    text = format_report(report)
    assert "甲" in text and "申" in text
    assert "missing cut @x" in text
