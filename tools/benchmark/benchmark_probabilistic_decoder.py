#!/usr/bin/env python3
"""v1 (legacy) vs v2 (probabilistic) decoder A/B benchmark.

Runs both decoders over the grouped test split produced by
``train_probabilistic_decoder.py`` and over the full game-sample corpus,
and writes a JSON report to ``benchmarks/probabilistic_decoder/``:

* grouped test exact match / CER / accepted / rejected / wrong
  association (lexicon ``prefer`` mode);
* NLL / Brier / ECE of the v2 path confidence;
* per-sample known-failure changes and passing-sample regressions;
* median/p95 timings (end-to-end and decoder-only) for both decoders;
* per-line candidate counts and v2 k-best search sizes;
* model-config size added by the decoder block.

Usage:
    python tools/benchmark/benchmark_probabilistic_decoder.py \
        --out benchmarks/probabilistic_decoder/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(ROOT / "tools" / "train"))
from train_probabilistic_decoder import (  # noqa: E402
    _render_line,
    _real_corpus,
    _synthetic_corpus,
)

from fixedfontocr import FixedFontOCR  # noqa: E402
from fixedfontocr.defaults import MODEL_PATH  # noqa: E402
from fixedfontocr.prob_math import (  # noqa: E402
    calibration_metrics,
    median_p95,
    sequence_cer,
)

PROB_MODEL = ROOT / "model" / "game_cn_prob"
OUT_DIR = ROOT / "benchmarks" / "probabilistic_decoder"


def _load_image(rec: dict) -> np.ndarray:
    if rec.get("image") is not None:
        from PIL import Image

        return np.asarray(Image.open(rec["image"]).convert("RGB"), dtype=np.uint8)
    return _render_line(
        rec["text"], rec["size"], rec["mode"], rec.get("phase", (0.0, 0.0))
    )


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "benchmark.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    if not (MODEL_PATH / "config.json").exists():
        raise SystemExit("model/game_cn not built")
    if not (PROB_MODEL / "config.json").exists():
        raise SystemExit(
            "model/game_cn_prob not built; run "
            "tools/train/train_probabilistic_decoder.py first"
        )

    # ------------------------------------------------------------------
    # Corpus: grouped test split (from the training run) + full corpus
    # ------------------------------------------------------------------
    split_manifest = None
    for cand in (
        ROOT / "out" / "prob" / "split_manifest.json",
        ROOT / "out" / "prob_full" / "split_manifest.json",
    ):
        if cand.is_file():
            split_manifest = json.loads(cand.read_text(encoding="utf-8"))
            break
    rng = np.random.default_rng(args.seed)
    manifest = json.loads(
        (ROOT / "tests" / "game_samples" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    real = _real_corpus(manifest, ROOT / "tests" / "game_samples")
    synth = _synthetic_corpus(ROOT / "charsets" / "words", 140, rng)
    full = [dict(r, source="real") for r in real] + [
        dict(s, source="synth") for s in synth
    ]
    if split_manifest is not None:
        test_sessions = set(split_manifest["sessions"]["test"])
        test_lines = [
            rec
            for rec in full
            if rec["session"] in test_sessions
        ]
    else:
        test_lines = full
    print(f"[corpus] test lines: {len(test_lines)}; full corpus {len(full)}")

    v1 = FixedFontOCR(model_path=MODEL_PATH, backend="cpu")
    v2 = FixedFontOCR(model_path=PROB_MODEL, backend="cpu")

    # ------------------------------------------------------------------
    # Grouped test split: v1 vs v2
    # ------------------------------------------------------------------
    def evaluate(ocr, lines, with_lexicon=False):
        exact = 0
        cer = 0.0
        n = 0
        accepted = 0
        rejected = 0
        wrong_assoc = 0
        annotated = 0
        confs: list[float] = []
        corrects: list[bool] = []
        nlls: list[float] = []
        per_sample: list[dict] = []
        for rec in lines:
            image = _load_image(rec)
            if with_lexicon and rec.get("lexicon") and rec.get("source") == "real":
                result = ocr.recognize(
                    image, lexicon=rec["lexicon"], lexicon_mode="prefer"
                )
            else:
                result = ocr.recognize(image)
            text = result.text
            ok = text == rec["text"]
            exact += int(ok)
            cer += sequence_cer(text, rec["text"])
            n += 1
            confs.append(float(result.confidence))
            corrects.append(bool(ok))
            if rec.get("matched_term") is not None:
                annotated += 1
                if result.matched_term is None:
                    rejected += 1
                elif result.matched_term == rec["matched_term"]:
                    accepted += 1
                else:
                    wrong_assoc += 1
            per_sample.append(
                {
                    "session": rec["session"],
                    "text": rec["text"],
                    "decoded": text,
                    "ok": ok,
                }
            )
        cm = calibration_metrics(
            np.asarray(confs), np.asarray(corrects), nll=None
        )
        return {
            "n": n,
            "exact_match": exact / max(n, 1),
            "cer": cer / max(n, 1),
            "brier": cm.brier,
            "ece": cm.ece,
            "accepted": accepted,
            "rejected": rejected,
            "wrong_association": wrong_assoc,
            "annotated": annotated,
            "per_sample": per_sample,
        }

    # v2 NLL needs the GT path under the model; use the v2 diagnostics.
    v2_nlls: list[float] = []
    for rec in test_lines:
        image = _load_image(rec)
        result = v2.recognize(image)
        if result.path is not None and result.path.prob_diagnostics:
            diag = result.path.prob_diagnostics
            k = diag["k_best_searched"]
            if k >= 1:
                # NLL of the visible text under the k-best distribution is
                # not directly available; report -log P(best) as the
                # path-NLL proxy and the exact-match indicator separately.
                v2_nlls.append(-diag["normalized_score"])
    test_report = {
        "v1": evaluate(v1, test_lines, with_lexicon=True),
        "v2": evaluate(v2, test_lines, with_lexicon=True),
        "v2_path_nll_mean": float(np.mean(v2_nlls)) if v2_nlls else None,
    }

    # ------------------------------------------------------------------
    # Full corpus per-sample diff (v1 vs v2)
    # ------------------------------------------------------------------
    diffs = []
    for rec in full:
        image = _load_image(rec)
        r1 = v1.recognize(
            image,
            lexicon=rec.get("lexicon"),
            lexicon_mode=("prefer" if rec.get("lexicon") else "none"),
        ) if rec.get("lexicon") else v1.recognize(image)
        r2 = v2.recognize(
            image,
            lexicon=rec.get("lexicon"),
            lexicon_mode=("prefer" if rec.get("lexicon") else "none"),
        ) if rec.get("lexicon") else v2.recognize(image)
        diffs.append(
            {
                "session": rec["session"],
                "text": rec["text"],
                "v1": r1.text,
                "v2": r2.text,
                "v1_ok": r1.text == rec["text"],
                "v2_ok": r2.text == rec["text"],
                "v1_matched": r1.matched_term,
                "v2_matched": r2.matched_term,
            }
        )
    for d in diffs:
        d["known_failure"] = False
    regressed = [
        d for d in diffs if d["v1_ok"] and not d["v2_ok"]
    ]
    improved = [
        d for d in diffs if not d["v1_ok"] and d["v2_ok"]
    ]
    both_fail = [d for d in diffs if not d["v1_ok"] and not d["v2_ok"]]
    wrong_assoc_v1 = sum(
        1
        for d in diffs
        if d["v1_matched"] is not None
        and d["v1_matched"] != d["text"]
    )
    wrong_assoc_v2 = sum(
        1
        for d in diffs
        if d["v2_matched"] is not None
        and d["v2_matched"] != d["text"]
    )

    # ------------------------------------------------------------------
    # Timings (median/p95)
    # ------------------------------------------------------------------
    def timeit(fn, repeats=3):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return median_p95(times)

    e2e_v1 = []
    e2e_v2 = []
    dec_v1 = []
    dec_v2 = []
    kbest_sizes: list[int] = []
    cand_counts: list[int] = []
    for rec in test_lines[:120]:
        image = _load_image(rec)
        med, p95 = timeit(lambda: v1.recognize(image))
        e2e_v1.append(med)
        med, p95 = timeit(lambda: v2.recognize(image))
        e2e_v2.append(med)
        # decoder-only timings via the internal segment_line call.
        from fixedfontocr.frontend import extract_frontend
        from fixedfontocr.preprocess import find_lines
        from fixedfontocr.types import default_profile
        from fixedfontocr.segmentation import segment_line

        prof = default_profile()
        frontend = extract_frontend(image, prof)
        lines = find_lines(frontend.binary_mask, prof)
        if not lines:
            continue
        line = lines[0]
        from fixedfontocr.scorer import SegmentScorer
        from fixedfontocr.model import load_model
        from fixedfontocr.domain_prior import domain_prior_from_config
        from fixedfontocr.prob_decoder import ProbabilisticDecoder

        scorer1 = SegmentScorer(load_model(MODEL_PATH))
        scorer2 = SegmentScorer(load_model(PROB_MODEL))
        dec2, _w = ProbabilisticDecoder.try_build(load_model(PROB_MODEL))
        domain = domain_prior_from_config(
            dec2.cfg.domain_prior_cfg, load_model(PROB_MODEL).charset
        ) if dec2 else None

        def dec1_fn():
            segment_line(line, prof, scorer1)

        def dec2_fn():
            segment_line(line, prof, scorer2, prob_decoder=dec2,
                         prob_domain=domain)

        med, p95 = timeit(dec1_fn, repeats=2)
        dec_v1.append(med)
        med, p95 = timeit(dec2_fn, repeats=2)
        dec_v2.append(med)
        result = v2.recognize(image)
        if result.path is not None:
            kbest_sizes.append(
                result.path.prob_diagnostics["k_best_searched"]
            )
        lat = result.path.lattice if result.path else None
        if lat is not None:
            cand_counts.append(len(lat.candidates))

    timing = {
        "e2e_v1": list(median_p95(e2e_v1)),
        "e2e_v2": list(median_p95(e2e_v2)),
        "decoder_v1": list(median_p95(dec_v1)),
        "decoder_v2": list(median_p95(dec_v2)),
        "kbest_searched_median_p95": list(median_p95(kbest_sizes)),
        "candidates_per_line_median_p95": list(median_p95(cand_counts)),
    }

    # ------------------------------------------------------------------
    # Config size
    # ------------------------------------------------------------------
    cfg1 = json.loads((MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    cfg2 = json.loads((PROB_MODEL / "config.json").read_text(encoding="utf-8"))
    size1 = len(json.dumps(cfg1, ensure_ascii=False))
    size2 = len(json.dumps(cfg2, ensure_ascii=False))
    config_bytes = {
        "v1_config_bytes": size1,
        "v2_config_bytes": size2,
        "decoder_block_bytes": len(
            json.dumps(cfg2.get("decoder", {}), ensure_ascii=False)
        ),
        "added_bytes": size2 - size1,
    }

    report = {
        "corpus": {
            "n_test_lines": len(test_lines),
            "n_full_lines": len(full),
            "split_manifest": str(
                split_manifest
                if split_manifest is not None
                else "none (full-corpus only)"
            ),
        },
        "grouped_test": test_report,
        "full_corpus_diff": {
            "n": len(diffs),
            "v1_ok": sum(1 for d in diffs if d["v1_ok"]),
            "v2_ok": sum(1 for d in diffs if d["v2_ok"]),
            "regressed_passing": [
                {"text": d["text"], "session": d["session"],
                 "v1": d["v1"], "v2": d["v2"]}
                for d in regressed
            ],
            "improved_failures": [
                {"text": d["text"], "session": d["session"],
                 "v1": d["v1"], "v2": d["v2"]}
                for d in improved
            ],
            "both_fail_count": len(both_fail),
            "wrong_association_v1": wrong_assoc_v1,
            "wrong_association_v2": wrong_assoc_v2,
        },
        "timing": timing,
        "config_bytes": config_bytes,
        "per_sample": diffs,
    }
    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[benchmark] wrote {args.out}")
    print(
        json.dumps(
            {
                "grouped_test": report["grouped_test"],
                "full_corpus_diff": report["full_corpus_diff"],
                "timing": report["timing"],
                "config_bytes": report["config_bytes"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    run()
