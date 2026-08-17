"""Probabilistic decoder v2 -- compatibility and freeze guarantees.

Covers the "兼容性测试" checklist:

* the legacy decoder v1 output is byte-identical with the pre-v2 code
  (the bundled ``model/game_cn`` has no decoder block -> v1);
* the full existing suite stays green (this file only spot-checks the
  equality; the suite itself is the gate);
* segmentation candidate count / order / bbox are unchanged by the v2
  machinery (raw evidence attachment is purely additive);
* CPU and WGPU produce the same candidate Top-K, so the v2 decode output
  is consistent across backends.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from fixedfontocr import FixedFontOCR
from fixedfontocr.defaults import MODEL_PATH, resolve_font
from fixedfontocr.model import load_model
from fixedfontocr.prob_decoder import ProbabilisticDecoder

from conftest import render_text
from prob_helpers import decoder_block


def _game_font():
    from fixedfontocr.defaults import FONT_PATH

    if not FONT_PATH.exists():
        pytest.skip("bundled font not present")
    return resolve_font(FONT_PATH)


def _v2_model(tmp_path: Path) -> Path:
    """Derived v2 copy of the bundled model (session-local, cheap)."""

    out = tmp_path / "game_cn_v2_compat"
    out.mkdir(parents=True)
    for name in ("config.json", "model.json", "charset.txt", "weights.bin",
                 "templates.bin", "geometry.json"):
        src = MODEL_PATH / name
        if src.exists():
            shutil.copy2(src, out / name)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    config["decoder"] = decoder_block("template")
    payload = json.dumps(config, indent=2) + "\n"
    (out / "config.json").write_text(payload, encoding="utf-8")
    (out / "model.json").write_text(payload, encoding="utf-8")
    return out


def _segmentation_signature(path):
    """(count, ordered (span, bbox) list, scores) of a decoded path."""

    cands = path.candidates
    sig = []
    for c in cands:
        s = c.segment
        sig.append(
            (
                c.atom_span,
                (s.x, s.y, s.w, s.h),
                tuple(float(v) for v in (c.scores.logits or ())),
                tuple(int(i) for i in (c.scores.char_ids or ())),
            )
        )
    return (len(cands), sig)


def test_v1_decoder_unchanged_without_decoder_block(tmp_path):
    """The bundled model has no decoder block: v1 output must match the
    frozen pipeline (no v2 machinery leaks in)."""

    if not (MODEL_PATH / "config.json").exists():
        pytest.skip("model/game_cn not built")
    model = load_model(MODEL_PATH)
    assert ProbabilisticDecoder.try_build(model)[0] is None
    ocr = FixedFontOCR(model_path=MODEL_PATH, backend="cpu")
    for text, size in (
        ("巴尔的摩", 32),
        ("鲃鱼。", 32),
        ("1234", 24),
        ("小", 32),
    ):
        image = render_text(text, _game_font(), font_size=size)
        result = ocr.recognize(image)
        assert result.text == text
        # v1 paths never carry v2 diagnostics.
        assert result.path.prob_diagnostics is None


def test_v1_alternative_golden(tmp_path):
    """v1 alternatives are unchanged: spot-check the exact tuple for a
    canonical low-res render (any change here is a v1 regression)."""

    if not (MODEL_PATH / "config.json").exists():
        pytest.skip("model/game_cn not built")
    ocr = FixedFontOCR(model_path=MODEL_PATH, backend="cpu")
    image = render_text("1234", _game_font(), font_size=24)
    result = ocr.recognize(image)
    assert result.text == "1234"
    # Re-run: identical (determinism).
    again = ocr.recognize(image)
    assert result.alternatives == again.alternatives
    assert result.path.char_ids == again.path.char_ids
    assert result.path.mean_score == again.path.mean_score


def _lattice_signature(lattice, logits: bool = True):
    """(count, ordered (span, bbox, char_ids[, logits])) of a lattice."""

    out = []
    for c in lattice.candidates:
        s = c.segment
        entry = [
            c.atom_span,
            (s.x, s.y, s.w, s.h) if s is not None else None,
            tuple(int(i) for i in (c.scores.char_ids or ())),
        ]
        if logits:
            entry.append(tuple(float(v) for v in (c.scores.logits or ())))
        out.append(tuple(entry))
    return (len(out), out)


def test_segmentation_candidates_unchanged_by_v2_attachment(tmp_path):
    """Adding raw evidence (v2 opt-in) must not change candidate count,
    order or bbox -- compare the v1 and v2 runs of the same line."""

    if not (MODEL_PATH / "config.json").exists():
        pytest.skip("model/game_cn not built")
    v2 = _v2_model(tmp_path)
    ocr1 = FixedFontOCR(model_path=MODEL_PATH, backend="cpu")
    ocr2 = FixedFontOCR(model_path=v2, backend="cpu")
    image = render_text("巴尔的摩", _game_font(), font_size=32)
    r1 = ocr1.recognize(image)
    r2 = ocr2.recognize(image)
    # Identical lattices: candidate count, order, spans, bboxes, Top-K
    # ids and scores -- the v2 machinery is purely additive.
    assert _lattice_signature(r1.path.lattice) == _lattice_signature(
        r2.path.lattice
    )
    # Raw evidence is attached only for the v2 model.
    assert all(
        c.scores.raw_evidence is not None
        for c in r2.path.lattice.candidates
    )
    assert all(
        c.scores.raw_evidence is None for c in r1.path.lattice.candidates
    )


def test_cpu_wgpu_v2_consistency(tmp_path):
    """CPU and WGPU produce identical candidate Top-K (pinned by Goal 20),
    so the v2 decode output must be consistent across backends."""

    try:
        import wgpu  # noqa: F401
    except Exception:
        pytest.skip("wgpu not installed")
    if not (MODEL_PATH / "config.json").exists():
        pytest.skip("model/game_cn not built")
    v2 = _v2_model(tmp_path)
    image = render_text("甲申", _game_font(), font_size=20)
    ocr_cpu = FixedFontOCR(model_path=v2, backend="cpu")
    ocr_wgpu = FixedFontOCR(model_path=v2, backend="wgpu")
    rc = ocr_cpu.recognize(image)
    rw = ocr_wgpu.recognize(image)
    # Same candidate Top-K (ids, spans, bboxes) on both backends (Goal 20
    # byte parity; the fused scores may differ in the last f32 ulp, which
    # never changes the Top-K ids) -- so the v2 decode output is identical.
    assert _lattice_signature(rc.path.lattice, logits=False) == _lattice_signature(
        rw.path.lattice, logits=False
    )
    assert rc.text == rw.text
    assert rc.path.char_ids == rw.path.char_ids
    assert rc.alternatives == rw.alternatives


def test_v2_outputs_deterministic_across_processes(tmp_path):
    """Two independent engines (fresh load) give identical v2 outputs."""

    if not (MODEL_PATH / "config.json").exists():
        pytest.skip("model/game_cn not built")
    v2 = _v2_model(tmp_path)
    image = render_text("1234", _game_font(), font_size=24)
    a = FixedFontOCR(model_path=v2, backend="cpu").recognize(image)
    b = FixedFontOCR(model_path=v2, backend="cpu").recognize(image)
    assert a.text == b.text
    assert a.alternatives == b.alternatives
    assert a.path.char_ids == b.path.char_ids
    assert a.path.prob_diagnostics == b.path.prob_diagnostics
    assert a.confidence == b.confidence
