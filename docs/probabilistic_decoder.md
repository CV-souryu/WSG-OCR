# Probabilistic joint decoding (v2)

This document describes the ``probabilistic/v2`` decoder: the trainable,
calibrated, explainable log-probability replacement for the legacy
``legacy/v1`` Goal 13 decoder (``decoder.py``). The v2 runtime is pure
NumPy, deterministic, and selected purely by the model config; models
without a ``decoder`` block keep the v1 output byte-identical.

## 1. Where v1's hand weights lived and what v2 replaces

The v1 decoder scored a complete path as

```
total = visual + geometry + lexicon + word_prior - segmentation_penalty
```

with hard-coded weights (`DecoderConfig`: visual 1.0, geometry 1.0,
lexicon 0.2, word prior 0.15, `segmentation_penalty = 0.005`,
`prefix/term_bonus`, `BASE_PRIOR = 0.35`), hand-tuned geometry penalties
(`char_geometry`), a heuristic lexicon support prior, and a fixed
``0.005 * (n_candidates - 1)`` length control. None of these were
trainable, and ``tune_visual.py`` fitted weights + calibration on the
same sample set.

v2 replaces the whole path scoring with the additive log-probability
model

```
S(pi) = log sum_z exp[
    log P(z)
    + sum_i ( log P(I_i | c_i, z) + log P(G_i | c_i) + log P(B_i) )
    + log P(c_{1:n} | domain) ]
```

* ``log P(I_i | c_i, z)`` — the local ranker's logit over the visual
  feature block (template + CNN + legacy-fused-as-one-feature),
  temperature-scaled, normalized by a per-candidate log-softmax over the
  candidate's Top-K options **plus an implicit background option**
  ("not a known character"). The background option is what makes a
  candidate whose every Top-K option is weak genuinely unlikely — a
  garbage merge/split candidate can no longer saturate its local
  softmax.
* ``log P(G_i | c_i)`` — a learned linear log-evidence over the raw
  geometry-residual block (bbox/aspect/ink/component/baseline vs the
  Goal 8 database entry), clipped to `[-6, 0]` so a clear visual match
  always dominates.
* ``log P(B_i)`` — a learned linear log-evidence over the boundary block
  (single/merge/split type, atom/component counts, approximate seam
  evidence, width ratio, piece balance, internal gaps, alignment,
  small-punctuation flag, the legacy segmentation-width penalty).
* ``log P(c_{1:n} | domain)`` — a calibrated `DomainPrior` (open text /
  allowed chars / lexicon unigram support / explicit unigram), see §5.
* ``z`` — the line-level render state, see §4.

The path is found by a deterministic k-best Viterbi over the atom
sequence per retained render state, the state is marginalized with
``logsumexp``, and the final score is length-normalized by
``(n + beta) ** alpha`` with ``alpha``/``beta`` chosen on the validation
split (§6). Alternatives come from the k-best list, never from post-hoc
string surgery. Ties are broken by (1) fewer anomalous split/merge
candidates, (2) smaller candidate spans, (3) lexicographic text — never
by Python object identity.

## 2. Features (schema v1, 48 dimensions)

For every ``(candidate, char_id)`` the extractor
(`prob_features.py`) builds un-fused features from the frozen raw
evidence the scorer attached (`VisualScores.raw_evidence`,
`RawEvidence`):

| block | features |
| --- | --- |
| template (0..13) | raw per-char template score (best prototype of *that character*), template top-1 flag, template margin, exact-match flag, winner prototype render size / sub-pixel phase / downsample mode, coarse-filter hit, state-table max + argmax, and the three z-conditional features (state score at `z`, delta vs best, clean flag) |
| CNN (14..23) | raw logit of the character, presence flag, top-1/top-2 logits, margin, fused rank, Top-5 entropy approximation, Top-1 softmax mass, missing-evidence and input-mode flags |
| geometry (24..32) | residuals of candidate bbox width/height/aspect/advance, ink ratio, component count and baseline against the chosen character's `geometry.json` entry, plus narrow/full type match and multi-component match |
| boundary (33..45) | candidate type one-hot (single/merge/split), atom count, component count, approximate seam ink and seam deviation from the ideal advance (derived geometrically; the exact seam cost is *not* recorded by the frozen segmentation output — documented limitation), width ratio, piece balance, max internal gap, vertical alignment offset, small-punctuation flag, legacy segmentation-width penalty |
| legacy (46..47) | the v1 fused ``visual_score`` and fused rank — **one feature among 47**, so v2 is not a re-fit of the old rule; the trained weight on it is modest (≈0.56 in the shipped model) |

Everything is deterministic NumPy; the only classifier work is the
per-character prototype re-scan (49 XOR+popcount per char, batched over
the candidate's Top-K in one vectorized call) which is a pure function of
the frozen template data and identical on CPU and WGPU.

## 3. Two-stage training

`tools/train/train_probabilistic_decoder.py`:

1. **Corpus**: the frozen real game samples (`tests/game_samples`,
   grouped by source screenshot: `vpNN` sessions, level slots, synthetic
   renders — every render is its own session, so no adjacent-crop
   leakage) plus freshly rendered synthetic lines (sizes 11..18 px and
   clean 32 px, bilinear/area downsampling with sub-pixel phase; the
   ground-truth render state is recorded).
2. **Harvest**: the frozen pipeline runs once per line; the scored
   lattice, the ground-truth path (oracle ink-boxes for synthetic lines,
   GT-text DP for real crops) and the per-`(candidate, char)` feature
   vectors are stored. The harvest attaches the *exact* lossy
   RawEvidence snapshot the runtime sees (template Top-K scores + CNN
   Top-K logits, `-inf` elsewhere), so training features are identical
   to runtime features. Oracle-missing lines and classifier Top-K
   coverage failures are recorded per split — never silently skipped.
3. **Grouped split**: train/validation/test by session with a fixed
   seed, saved as `split_manifest.json` (sessions, counts, chars per
   split).
4. **Stage 1 — local ranker**: pairwise logistic over the visual block
   (correct char vs the candidate's Top-K wrong chars, hard negatives),
   trained by deterministic full-batch gradient descent in float64. The
   training z-condition is the line's ground-truth render state for
   synthetic lines and the evidence-inferred dominant state for real
   crops.
5. **Stage 2 — structured**: with the local ranker frozen, learn the
   geometry/boundary weights, the local temperature and the background
   logit by pairwise logistic over the k-best paths (GT path positive,
   competing paths negative, exact analytic gradients), with
   ``(alpha, beta)`` length normalization selected on validation.
6. **Calibration**: monotone piecewise confidence calibration and the
   reject thresholds are fitted on validation only.
7. **Export**: `model/game_cn_prob` — a self-contained model copy whose
   `config.json`/`model.json` carry the versioned `decoder` block. The
   training script asserts the harvested replay reproduces the runtime
   decoder on sample lines before exporting.

## 4. Line render state z

Template V2's render size / sub-pixel phase / downsample mode are not
chosen independently per character:

* the finite state set is `{(size, mode)}` over the Goal 9 grid
  (11..16 px × bilinear/area) plus the `clean` high-res prototype;
* every candidate's fused Top-1 character votes for its winner
  prototype's state; the line keeps the top-`M` states (default 3) in
  deterministic order, and `clean` is always retained as the fallback
  for high-resolution prototypes;
* each state conditions only the three z-template features; the whole
  path is scored per state and the final score marginalizes `z` with
  `logsumexp` (`max_state_approx=True` is the explicitly recorded
  max-state approximation);
* a wrong state cannot override strong visual evidence because the
  z-features are a small additive contribution to the same local logit
  that the unconditional template/CNN evidence dominates (pinned by
  `tests/test_render_state.py`).

## 5. DomainPrior

`domain_prior.py` defines the unified interface and four priors: open
text (uniform), allowed chars (uniform over the UI-field set — visual
equivalents such as Latin `T` / Cyrillic `Т` are disambiguated *inside*
the equivalence class by the local normalizer), lexicon (per-character
term support counts — a frequency, never the 0..1 heuristic match score —
mapped to a log prior through a monotone calibration), and an explicit
unigram. The prior is bounded and additive, so clear visual evidence
dominates (Goal 15), non-unique support stays flat (ambiguous results
keep alternatives), and unknown text keeps a floor prior.

## 6. Length normalization

The final score is `S / (n + beta) ** alpha` with `alpha`/`beta` chosen
on validation over a grid. The k-best DP prunes partial paths by the
same normalized look-ahead key, so a long path of individually strong
characters is never pruned by a short path whose one candidate saturates
the local softmax. Why this does not systematically bias path length:
with `alpha = 1, beta = 0` the score is the per-character mean
(length-neutral); the validation-chosen `beta > 0` softens the
short-path regime without imposing the old linear per-candidate penalty,
and the boundary/geometry terms — not the length term — carry the
merge/split decision. The shipped model selected `alpha = 1.0,
beta = 1.0`.

## 7. Data splits and calibration hygiene

* splits are grouped by original screenshot/session (never by crop);
* train fits the ranker, validation selects features, temperature,
  `(alpha, beta)`, calibration points and reject thresholds, test is
  used only for the final report;
* the split manifest (`out/prob_full/split_manifest.json`) is
  reproducible (fixed seed) and reports session/crop/char counts per
  split;
* the test set never fits any parameter.

## 8. Known limitations (measured, not hidden)

The full held-out numbers live in
`benchmarks/probabilistic_decoder/benchmark.json` and the training
report in `out/prob_full/report.json`; the summary (trained model
`model/game_cn_prob`, grouped test split of 165 lines):

| metric | v1 | v2 |
| --- | --- | --- |
| exact match (grouped test) | 0.748 | 0.610 |
| CER (grouped test) | 0.092 | 0.207 |
| Brier / ECE (grouped test) | 0.209 / 0.185 | 0.285 / 0.293 |
| wrong associations (full corpus, prefer mode) | 7 | **1** |
| full-corpus correct lines (of 725) | 537 | 469 |
| regressed passing / improved failures | — | 91 / 23 |

Limitations, in order of impact:

1. **Low-res digit/Latin lines trail v1** (`获得金币1000`/`Z17`/`U-156` at
   14 px): v1's tuned CNN-margin fusion and `_drop_weak_merges` gates
   beat the trained ranker on these; the training corpus under-represents
   them at exactly the game's sizes/degrades. More synthetic coverage and
   a stronger CNN block would close most of the gap.
2. **Latency**: v2 end-to-end median/p95 is 1.39x/1.61x of v1 (the
   acceptance target was 1.15x). The overhead is the per-character
   prototype re-scan needed for the render-state tables (~8 ms on the
   largest lattices) plus the 4-state k-best; both are cached and
   vectorized but not yet eliminated (a scorer API exposing
   per-prototype distances would remove the re-scan entirely).
3. **Boundary weights did not move** in stage 2 (the gradient signal is
   weak with the current corpus); the boundary block is therefore
   close-to-neutral and the merge/split decisions ride on the local
   probabilities and geometry.
4. **Oracle path recall** on the real corpus is only ~0.4-0.55 (real
   crops whose correct segmentation/text is not present in the lattice);
   these lines are reported as segmentation/coverage failures, never as
   decoder successes.
5. The seam evidence is geometric (crossing ink, deviation from the
   ideal advance) because the frozen segmentation output does not record
   seam provenance/cost — documented approximation.

The framework itself (calibration metrics, determinism, config
validation, v1 byte-parity, replay≡runtime) is complete and tested; the
accuracy gap is an honest measured outcome of the current data scale,
not a definitional artifact.
