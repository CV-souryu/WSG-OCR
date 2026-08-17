# Oracle lattice recall

When recognition fails, the first question is not "which CNN weights do I
tune?" but **"was the correct segmentation even in the lattice?"** The
lattice never decides cut/no-cut irreversibly — it offers every plausible
candidate and lets the decoder choose. An oracle path (the ground-truth
segmentation realized as lattice candidates) missing means no classifier,
however strong, can recover the text:

| oracle path | decoder output | verdict |
|---|---|---|
| exists | correct | ok |
| exists | wrong | **classification / decoding problem** (lattice fine) |
| missing | wrong | **segmentation problem** (candidate generation/pruning) |

The oracle ignores all scores. It rebuilds the production candidate lattice
and checks geometrically whether a path covers the ground-truth per-char
ink boxes.

## Metric definition

Ground truth is a per-character *raster ink box* (`GroundTruthBox`), i.e.
the tight bbox of the character's thresholded pixels — directly comparable
with candidate segment bboxes. A candidate is a valid representation of a
GT character when:

* it covers ≥ 90% of the character's ink, with ≥ 90% of its own ink inside
  the character's box (`coverage_threshold`, default 0.9);
* its bbox is within 2 px of the character's box on all four sides
  (`tolerance`).

An oracle path exists when valid candidates tile the atom sequence exactly,
in text order (`oracle_lattice_recall`). Failure reasons:

* `unassigned_atom` — an atom's ink spans two character boxes (or the
  connector column between them): a connected blob was not split near the
  GT boundary. `straddled_boundaries` names the missing cut positions.
* `missing_candidate` — atoms are individually inside one box, but no
  candidate covers a whole character: the needed merge/split was pruned
  (e.g. the cut landed inside the left glyph) or never generated.
* `no_path` — pathological: atoms assignable and per-char candidates
  exist, but no ordered tiling exists.

`evaluate_oracle` runs the check over a corpus and attributes each sample
to `ok` / `segmentation` / `decoding` (optionally running the real
recognizer via `decode_fn`); `format_report` renders the table.

## Ground truth

`render_ground_truth(text, font_path, ...)` draws the line with a single
`draw.text` call (the layout the pipeline sees) and computes each
character's GT box from a standalone raster at the same integer origin
shifted by the cumulative advance — pixel-exact at integer advances,
±1 px otherwise (absorbed by the tolerance). A box with no ink, or ink
outside all boxes, is a hard error.

`glue_ground_truth(left, right, bridge=...)` joins two single-char lines
to reproduce the real-game touching failure modes:

* `"full"` — zero gap: the glyphs' own ink columns become adjacent (real
  touching, no extra ink).  Whether the pair forms one connected component
  with a valley at the boundary depends on the glyph shapes (甲+申:
  connected with *no* valley — the forced seam cut case);
* `"row"` — a 1 px blank column filled by a thin connector (column
  projection ≤ 18% of the height): one connected component with a real
  valley (the classic low-res connector);
* `"none"` — the 1 px blank column stays blank: two components.

## Usage

```bash
python tools/oracle_recall.py            # rendered corpus, model geometry + decode
python tools/oracle_recall.py --no-model # metric only
python tools/oracle_recall.py --font-size 14
```

```python
from fixedfontocr.oracle import evaluate_oracle, render_ground_truth, format_report
from fixedfontocr.types import default_profile

lines = [render_ground_truth(t, font_path, font_size=32) for t in ["潜甲", "LV.40"]]
report = evaluate_oracle(lines, default_profile(), decode_fn=ocr_recognize)
print(format_report(report))
```

## Findings

Goal 21 candidate-generation upgrades (frozen CNN/decoder), measured by the
oracle on the 32 px rendered corpus with model geometry:

* **Oracle lattice recall: 63.2% (12/19) -> 100% (16/16)**; the glued-pair
  subset (single connected component) is 100% (3/3) at 6 candidates total
  (no enumeration blowup; ~6.8 candidates/line across the corpus).
* **Forced advance/seam cuts** (`_forced_seam_cuts` in segmentation.py):
  a valley-free fully-touching blob (甲+申 zero-gap, one component, no
  valley column) is now split by a min-cost seam near the font-advance
  multiple / equal-part position; the original whole component stays
  reachable as the merge of its atoms.  Seams run only when the component
  has *no* valley cut and is at least 2x the expected width wide -- both
  gates keep real glyphs' wide components (命's 人+一, 尔's top) from being
  sliced, which would push the glyph's whole merge past
  `max_merge_components`.
* **Small punctuation gets a wider bbox envelope**: an 8x3 `-` (ratio
  2.5) exceeds the normal 1.35x envelope; small candidates (below the
  minimum char height) use a 3.0x envelope instead, so `-`/`=`/`~` reach
  the scorer (T-23 / U- at 32 px had `missing_candidate` before; both now
  have oracle paths), while near-empty 1-2 px bars (ratio ~3.3+, glyph
  fragments) stay rejected.  This also fixed 8 real-game known failures
  (`い156` -> `U-156`, `い96` -> `U-96`) and reads the hyphen in
  `石勒苏益格-荷尔施泰因` crops.
* **Expected-width estimate excludes wide components** (aspect > 1.1) --
  they are either glued blobs or a real glyph's wide part (命's 人+一, 尔's
  top), so their width is not a glyph width.  Excluding them keeps the
  median on the line's normal glyphs; when the whole line is one wide
  blob the height heuristic (0.8 x height) is the fallback.  (A *capped*
  contribution instead drags the median down and makes the wide part
  itself look splittable.)
* L+V and U- never form one connected component in this font's clean
  raster at any size (their outer ink rows do not overlap) -- their
  real-game gluing is an anti-aliasing artifact; they stay normal-render
  oracle samples.
* T-23 at 32 px is now attributed to **decoding**: the oracle path exists
  but the model reads Cyrillic `Т`.  The separability check
  (`tests/test_tt_separability.py`) pins that Latin T and Cyrillic Т are
  *pixel-identical* in the registered font at 14/16/32 px (normalized
  Hamming 0/576), so no CNN can separate them: the fix is charset /
  allowed_chars / lexicon disambiguation, not hard-negative training.

These numbers depend on `font_size` and on whether the model geometry is
passed (it changes `expected_width` and thus the cut positions and the
bbox prior) -- that sensitivity is the point: the metric measures the
lattice the production pipeline actually builds.
