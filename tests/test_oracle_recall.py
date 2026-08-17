"""Oracle lattice recall: is the ground-truth segmentation in the lattice?

The oracle ignores all scores: it builds the same candidate lattice the
production pipeline uses and checks whether a path covering the per-char
ground-truth ink boxes exists.  If it does not, no classifier can recover
the correct text -- the failure is in segmentation (candidate generation /
pruning).  If it does but the decoder picked another path, the failure is
in classification / decoding.

Goal 21 candidate-generation upgrades under test:

* forced seam cuts at advance multiples / equal parts rescue valley-free
  fully-touching blobs (甲申 zero-gap);
* the edge-valley rule separates a short trailing/leading glyph that is
  itself a valley run (``U-`` style);
* small punctuation bypasses the Goal 8 bbox envelope, so ``-`` / ``.`` /
  ``·`` candidates reach the scorer (T-23 / U- at 32 px);
* the expected-width estimate caps wide multi-glyph components so single-
  component lines still get split.
"""

from __future__ import annotations

import numpy as np
import pytest

from fixedfontocr.oracle import (
    COVERAGE_THRESHOLD,
    GroundTruthBox,
    GroundTruthLine,
    OracleReport,
    OracleSample,
    evaluate_oracle,
    format_report,
    glue_ground_truth,
    oracle_lattice_recall,
    render_ground_truth,
)
from fixedfontocr.segmentation import (
    _estimate_expected_width,
    _forced_seam_cuts,
    _geometry_bbox_ok,
    _split_along_paths,
    _valley_cuts,
    connected_components,
    expand_atoms,
)
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
    "获得金币1000",
]

MODEL_GEOMETRY = None


def _model_geometry():
    """Model font geometry database, or None when the model is absent."""
    global MODEL_GEOMETRY
    from pathlib import Path

    from fixedfontocr.geometry import FontGeometryDatabase

    if MODEL_GEOMETRY is None:
        path = Path("model/game_cn/geometry.json")
        if path.exists():
            MODEL_GEOMETRY = FontGeometryDatabase.load(path)
        else:
            MODEL_GEOMETRY = False
    return MODEL_GEOMETRY or None


def _pair(font_path, a: str, b: str, bridge: str) -> GroundTruthLine:
    la = render_ground_truth(a, font_path, font_size=32)
    lb = render_ground_truth(b, font_path, font_size=32)
    return glue_ground_truth(la, lb, bridge=bridge)


# ---------------------------------------------------------------------------
# Ground-truth renderer
# ---------------------------------------------------------------------------


def test_render_boxes_match_standalone_glyph_masks(font_path):
    """Per-char GT boxes are pixel-exact against standalone tight renders."""

    from fixedfontocr.oracle import render_glyph_mask

    for text in NORMAL_STRINGS:
        gt = render_ground_truth(text, font_path, font_size=32)
        covered = np.zeros_like(gt.mask)
        for i, box in enumerate(gt.boxes):
            sub = gt.mask[box.y0 : box.y1, box.x0 : box.x1]
            ref = render_glyph_mask(gt.text[i], font_path, font_size=32)
            assert sub.shape == ref.shape, (
                f"{text!r} char {i} {gt.text[i]!r}: box {sub.shape} != "
                f"standalone {ref.shape}"
            )
            assert np.array_equal(sub, ref), (
                f"{text!r} char {i} {gt.text[i]!r}: raster mismatch"
            )
            covered[box.y0 : box.y1, box.x0 : box.x1] = True
        assert not np.any(gt.mask & ~covered), f"{text!r}: ink outside boxes"


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
        assert len(res.path) == len(text)
        spans = [c.atom_span for c in res.path]
        assert spans[0][0] == 0
        assert all(
            spans[k][1] == spans[k + 1][0] for k in range(len(spans) - 1)
        )
        assert spans[-1][1] == res.n_atoms


def test_oracle_recall_with_model_geometry_on_normal_renders(font_path):
    """With the production geometry, every normal render keeps its path.

    This is the no-regression pin for the Goal 21 pruning changes
    (punctuation bypass + wide-component cap + forced seams).
    """

    geometry = _model_geometry()
    profile = default_profile()
    for text in NORMAL_STRINGS:
        gt = render_ground_truth(text, font_path, font_size=32)
        res = oracle_lattice_recall(
            gt.line, profile, gt.boxes, geometry=geometry
        )
        assert res.recall, (
            f"{text!r} (geometry): oracle path missing "
            f"(reason={res.reason!r})"
        )


def test_oracle_recall_ignores_scores(font_path):
    """The metric must not require a scorer: no score is ever consulted."""

    gt = render_ground_truth("潜甲", font_path, font_size=32)
    res = oracle_lattice_recall(gt.line, default_profile(), gt.boxes)
    assert res.recall
    assert all(c.score is None and c.scores is None for c in res.path)


# ---------------------------------------------------------------------------
# Glued pairs: the forced seam cut (valley-free touching)
# ---------------------------------------------------------------------------


def test_zero_gap_jia_shen_is_one_component_with_no_valley(font_path):
    """甲+申 zero-gap: one component whose boundary has no valley column.

    This is the fully-touching case the valley heuristic cannot cut -- the
    regression fixture for the forced advance/seam cuts.
    """

    gt = _pair(font_path, "甲", "申", "full")
    comps = connected_components(gt.line)
    assert len(comps) == 1
    comp = comps[0]
    profile = default_profile()
    expected = _estimate_expected_width(comps, profile)
    assert _valley_cuts(comp.mask, expected, profile, 3) == []
    # The forced seam mechanism does find a cut near the glyph boundary.
    seams = _forced_seam_cuts(comp.mask, expected, profile, 3)
    assert len(seams) == 1
    mean_x = float(np.mean(seams[0][0]))
    # GT boundary sits at 甲's box end (25): seam must land within ~2 px.
    assert abs(mean_x - 25) <= 3, f"seam at x={mean_x:.1f}, boundary at 25"


def test_forced_seam_rescues_valley_free_blob(font_path):
    """甲+申 zero-gap has an oracle path *because* of the split machinery.

    With splitting disabled the single atom straddles both GT boxes and
    the oracle path is missing; with the forced seam it exists.  This is
    the exact 'no valley -> correct path never enters the lattice' failure
    the metric was built to detect, now fixed at the lattice level.
    """

    gt = _pair(font_path, "甲", "申", "full")
    profile = default_profile()
    res_off = oracle_lattice_recall(
        gt.line, profile, gt.boxes, split_wide=False
    )
    assert not res_off.recall
    assert res_off.reason == "unassigned_atom"
    assert res_off.straddled_boundaries == ((0, 1, 25),)

    res_on = oracle_lattice_recall(gt.line, profile, gt.boxes)
    assert res_on.recall
    assert len(res_on.path) == 2
    # The original component stays reachable as the whole merge.
    spans = {c.atom_span for c in res_on.path}
    assert (0, res_on.n_atoms) not in spans  # path uses the split pieces


def test_glued_pairs_with_connector_have_oracle_path(font_path):
    """Thin-connector glued pairs keep their oracle path (valley rule)."""

    for left, right in [("甲", "申"), ("4", "3")]:
        gt = _pair(font_path, left, right, "row")
        comps = connected_components(gt.line)
        assert len(comps) == 1, f"{left}+{right} row bridge must connect"
        res = oracle_lattice_recall(gt.line, default_profile(), gt.boxes)
        assert res.recall, (
            f"{left}+{right} row-bridge: oracle missing "
            f"(reason={res.reason!r})"
        )
        assert len(res.path) == 2


def test_separated_pair_oracle_path(font_path):
    """Without a bridge the pair is two components: trivially covered."""

    gt = _pair(font_path, "甲", "申", "none")
    assert len(connected_components(gt.line)) == 2
    res = oracle_lattice_recall(gt.line, default_profile(), gt.boxes)
    assert res.recall
    assert len(res.path) == 2


def test_lv_and_u_dash_do_not_glue_in_clean_rendering(font_path):
    """L/V and U/- never form one component in this font's clean raster.

    L's right-edge ink is its bottom bar and V's left-edge ink is its top
    arm (rows never overlap at any size); U's right stroke ends above the
    '-' bar.  Their real-game gluing is an anti-aliasing artifact that
    clean rendering cannot reproduce, so the glued regression set pins
    甲申/43 and these two stay normal-render oracle samples.
    """

    for a, b in [("L", "V"), ("U", "-")]:
        gt = _pair(font_path, a, b, "full")
        assert len(connected_components(gt.line)) == 2, f"{a}+{b} glued?!"
        res = oracle_lattice_recall(gt.line, default_profile(), gt.boxes)
        assert res.recall


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
    assert res.best_coverage[0] < COVERAGE_THRESHOLD
    assert res.best_coverage[0] >= 0.4
    assert res.atom_assignments == (0, 0)


def test_atom_interleaving_reports_no_path(font_path):
    """Pathological x-interleaved boxes: atoms are individually assignable
    but the candidates cannot tile in text order."""

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
# Small punctuation: the Goal 8 bbox-envelope bypass
# ---------------------------------------------------------------------------


def test_punctuation_bypasses_bbox_envelope(font_path):
    """Short punctuation (8x3 '-') is not rejected by the height-derived
    bbox envelope -- it must reach the scorer."""

    from fixedfontocr.oracle import render_glyph_mask

    geometry = _model_geometry()
    dash = render_glyph_mask("-", font_path, font_size=32)
    profile = default_profile()
    assert dash.shape[0] < max(2, profile.char_height_min)
    # The envelope alone rejects it...
    from fixedfontocr.types import Component

    comp = Component(mask=dash, x=0, y=0, w=dash.shape[1], h=dash.shape[0])
    assert not _geometry_bbox_ok(comp, geometry)
    # ...but the candidate filter lets it through (height below minimum).
    res = oracle_lattice_recall(
        render_ground_truth("T-23", font_path, font_size=32).line,
        profile,
        render_ground_truth("T-23", font_path, font_size=32).boxes,
        geometry=geometry,
    )
    assert res.recall, f"T-23 (geometry): {res.reason!r}"

    res = oracle_lattice_recall(
        render_ground_truth("U-", font_path, font_size=32).line,
        profile,
        render_ground_truth("U-", font_path, font_size=32).boxes,
        geometry=geometry,
    )
    assert res.recall, f"U- (geometry): {res.reason!r}"


# ---------------------------------------------------------------------------
# Expected-width estimate cap
# ---------------------------------------------------------------------------


def test_expected_width_excludes_wide_components(font_path):
    """A single wide component must not set expected width to itself.

    With the model geometry the height/width-derived em of a glued blob
    would estimate 'expected = the blob' and the split trigger would never
    fire; conversely, a *capped* contribution would drag the median down
    and make real glyph parts (命's 人+一) look splittable.  Wide
    components (aspect > 1.1) are excluded from the estimate, falling back
    to the height heuristic when the whole line is one wide blob.
    """

    geometry = _model_geometry()
    gt = _pair(font_path, "甲", "申", "full")  # one 52px-wide component
    comps = connected_components(gt.line)
    assert len(comps) == 1
    expected = _estimate_expected_width(comps, default_profile(), geometry)
    max_h = max(c.h for c in comps)
    assert expected <= max_h * 0.8 + 1e-6, (
        f"expected width {expected:.1f} not capped at 0.8*height"
    )
    assert comps[0].w > expected * 1.5  # the split trigger fires

    # A normal line with one wide glyph part (命's 人+一, 30x13) must not
    # have it split: no valley exists there, and the seam gate (w >= 2 *
    # expected) keeps the part whole, so its glyph's merge stays within
    # max_merge_components.
    gt2 = render_ground_truth("生命值1000", font_path, font_size=32)
    comps2 = connected_components(gt2.line)
    expected2 = _estimate_expected_width(comps2, default_profile(), geometry)
    atoms2, _ = expand_atoms(
        comps2, default_profile(), 4, True, geometry
    )
    assert len(atoms2) == len(comps2), (
        f"wide glyph part split at expected {expected2:.1f}: "
        f"{len(comps2)} comps -> {len(atoms2)} atoms"
    )


# ---------------------------------------------------------------------------
# Split-along-paths unit
# ---------------------------------------------------------------------------


def test_split_along_paths_partitions_mask():
    rng = np.random.default_rng(7)
    mask = rng.random((9, 20)) < 0.4
    straight = np.full(9, 7, dtype=np.int32)
    pieces = _split_along_paths(mask, [straight])
    assert len(pieces) == 2
    assert np.array_equal(pieces[0] | pieces[1], mask)
    assert not np.any(pieces[0] & pieces[1])
    # Straight cut at x=7: left piece keeps cols [0, 7).
    assert not np.any(pieces[0][:, 7:])

    snake = np.array([6, 7, 7, 8, 8, 7, 6, 6, 7], dtype=np.int32)
    pieces = _split_along_paths(mask, [snake])
    assert len(pieces) == 2
    assert np.array_equal(pieces[0] | pieces[1], mask)
    assert not np.any(pieces[0] & pieces[1])


# ---------------------------------------------------------------------------
# Attribution: segmentation vs decoding errors
# ---------------------------------------------------------------------------


def _pruned_merge_line(font_path) -> GroundTruthLine:
    bar1 = render_ground_truth("1", font_path, font_size=32)
    bar2 = render_ground_truth("1", font_path, font_size=32)
    w1 = bar1.mask.shape[1]
    h = max(bar1.mask.shape[0], bar2.mask.shape[0])
    gap = 15
    mask = np.zeros((h, w1 + gap + bar2.mask.shape[1]), dtype=bool)
    mask[: bar1.mask.shape[0], :w1] = bar1.mask
    mask[: bar2.mask.shape[0], w1 + gap :] = bar2.mask
    return GroundTruthLine(
        text="X",
        mask=mask,
        boxes=(GroundTruthBox("X", 0, 0, mask.shape[1], h),),
    )


def test_evaluate_attributes_segmentation_vs_decoding(font_path):
    profile = default_profile()
    ok_line = render_ground_truth("潜甲", font_path, font_size=32)
    seg_line = _pruned_merge_line(font_path)  # oracle missing

    report = evaluate_oracle(
        [ok_line, seg_line],
        profile,
        decode_fn=lambda gt: gt.text,
    )
    assert report.total == 2
    assert report.oracle_recall == 0.5
    assert report.status_counts == {"ok": 1, "segmentation": 1}

    report = evaluate_oracle(
        [ok_line, seg_line],
        profile,
        decode_fn=lambda gt: "!!",
    )
    assert report.status_counts == {"decoding": 1, "segmentation": 1}

    report = evaluate_oracle([ok_line], profile)
    assert report.status_counts == {"ok": 1}


def test_evaluate_records_candidate_counts(font_path):
    report = evaluate_oracle(
        [render_ground_truth("潜甲", font_path, font_size=32)],
        default_profile(),
    )
    sample = report.samples[0]
    assert sample.n_atoms > 0
    assert sample.n_candidates > 0
    assert sample.n_candidates >= sample.n_atoms


def test_format_report_smoke(font_path):
    report = evaluate_oracle(
        [render_ground_truth("潜甲", font_path, font_size=32)],
        default_profile(),
    )
    text = format_report(report)
    assert "oracle lattice recall: 100.0%" in text
    assert "ok" in text


def test_format_report_names_missing_cut_position():
    """The report must say *where* the missing cut is (guides the fix)."""

    report = OracleReport(
        total=1,
        oracle_recall=0.0,
        status_counts={"segmentation": 1},
        samples=(
            OracleSample(
                text="甲申",
                oracle=False,
                status="segmentation",
                reason="unassigned_atom",
                straddled_boundaries=((0, 1, 25),),
                n_atoms=1,
                n_candidates=1,
            ),
        ),
    )
    text = format_report(report)
    assert "missing cut @x25" in text
    assert "甲|申" in text
