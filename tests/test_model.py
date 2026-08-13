from __future__ import annotations

import json

import numpy as np

from fixedfontocr.fontgen import build_templates, write_model
from fixedfontocr.model import load_model


def test_model_roundtrip(tmp_path, font_path):
    charset = "0123456789abcdef"
    chars, templates = build_templates(font_path, list(charset))
    out = tmp_path / "model"
    write_model(out, chars, templates)

    model = load_model(out)
    assert model.input_size == 24
    assert model.charset == list(charset)
    assert model.templates.shape == (16, 72)
    assert model.config["version"] == 1

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["classes"] == 16
    assert config["dtype"] == "f32"


def test_load_rejects_mismatched_charset(tmp_path, font_path):
    chars, templates = build_templates(font_path, list("abc"))
    out = tmp_path / "model"
    write_model(out, chars, templates)
    (out / "charset.txt").write_text("abcd\n", encoding="utf-8")

    try:
        load_model(out)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched classes")


def test_templates_are_small(font_path):
    chars, templates = build_templates(font_path, list("0123456789") * 100)
    # 1000 classes * 72 bytes
    assert templates.nbytes == 1000 * 72
