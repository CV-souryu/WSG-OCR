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
with a 1 px blank column between them and optionally fills it, to
reproduce the real-game fully-touching failure mode:

* `"full"` — blank column filled entirely: one connected component with
  *no* valley at the boundary (valley heuristic cannot cut it);
* `"row"` — a thin connector (column projection ≤ 18% of the height): one
  connected component with a real valley (the classic low-res connector);
* `"none"` — blank column stays blank: two components.

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

## Findings on the current corpus (32 px, model geometry)

* Normal renders (`潜甲`, `鲃鱼。`, `LV.40`, `4+3`, `Z17`, `1000`, ...) have
  oracle paths and decode correctly.
* Fully-glued pairs (`甲申`, `43`, `LV`, `U-` with a full-height connector)
  have **no oracle path**: `unassigned_atom` with the missing cut position
  (e.g. `missing cut @x17 (4|3)`). This is the user-visible `LV`/`4+3`/`U-`
  failure mode — the correct fix is not a stronger CNN but lattice
  candidates for those boundaries (forced cuts at advance multiples /
  min-cost seam cuts).
* `T-23` / `U-` at 32 px: the standalone `-` component (8×3) is pruned by
  the Goal 8 bbox prior (`_geometry_bbox_ok`, em estimated from height
  only) → `missing_candidate`, and the model indeed reads `r23` / `山`.
  At 14 px the oracle path exists and the model still reads `Т-23`
  (Cyrillic lookalike) — a *classification* problem, correctly attributed
  as `decoding` rather than `segmentation`.
* A valley cut may land inside the left glyph (e.g. `43` full: cut at x=15,
  two px inside `4`, boundary at x=17): atoms stay assignable, but no
  candidate reaches the coverage threshold → `missing_candidate` with
  per-char best coverage in the report.

These numbers depend on `font_size` and on whether the model geometry is
passed (it changes `expected_width` and thus the valley cut positions and
the bbox prior) — that sensitivity is the point: the metric measures the
lattice the production pipeline actually builds.
