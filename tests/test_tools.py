from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from fixedfontocr.cnn import forward, forward_with_activations  # noqa: E402
from fixedfontocr.fontgen import build_templates, write_model  # noqa: E402
from fixedfontocr.model import load_model, write_cnn_model  # noqa: E402
from tools.train import tinycnn_arch  # noqa: E402
from tools.dataset.collect_real_samples import collect_directory  # noqa: E402
from tools.train.export_model import load_checkpoint, make_test_vectors  # noqa: E402
from tools.dataset.generate_font_dataset import generate_dataset  # noqa: E402

from conftest import render_text  # noqa: E402


def test_generate_font_dataset(tmp_path, font_path):
    out = tmp_path / "synth.npz"
    charset = "01A"
    generate_dataset(
        font_path,
        list(charset),
        samples_per_char=3,
        output=out,
        seed=1,
    )
    data = np.load(out)
    assert data["x"].shape[1:] == (24, 24)
    assert data["x"].dtype == np.uint8
    assert set(np.unique(data["x"])) <= {0, 255}
    assert data["y"].dtype == np.int64
    assert list(data["chars"]) == list(charset)
    assert data["x"].shape[0] == data["y"].shape[0]
    assert len(str(data["font_sha256"][0])) == 64
    assert str(data["input_mode"][0]) == "binary"


def test_generate_font_dataset_soft(tmp_path, font_path):
    out = tmp_path / "synth_soft.npz"
    charset = "01A"
    generate_dataset(
        font_path,
        list(charset),
        samples_per_char=3,
        output=out,
        seed=1,
        soft=True,
    )
    data = np.load(out)
    assert data["x"].dtype == np.uint8
    # Soft glyphs keep intermediate foreground intensities (anti-aliasing).
    assert np.any((data["x"] > 0) & (data["x"] < 255))
    assert str(data["input_mode"][0]) == "soft"


def test_collect_real_samples(tmp_path, font_path):
    root = tmp_path / "real"
    (root / "12").mkdir(parents=True)
    (root / "0").mkdir()
    from PIL import Image

    Image.fromarray(render_text("12", font_path)).save(root / "12" / "line.png")
    Image.fromarray(render_text("0", font_path)).save(root / "0" / "one.png")
    out = tmp_path / "real.npz"
    collect_directory(root, out)
    data = np.load(out)
    assert data["x"].shape[1:] == (24, 24)
    assert sorted(str(c) for c in data["chars"]) == ["0", "1", "2"]
    assert data["y"].size == data["x"].shape[0]
    assert data["origins"].size == data["x"].shape[0]


def test_train_checkpoint_load(tmp_path):
    model = tinycnn_arch.TinyCNNBN(5)
    torch = __import__("torch")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "charset": list("01234"),
            "architecture": "tinycnn_bn",
            "input_size": 24,
            "version": 1,
        },
        tmp_path / "model.pth",
    )
    loaded, charset, font_sha256, input_mode = load_checkpoint(
        tmp_path / "model.pth"
    )
    assert charset == list("01234")
    assert isinstance(loaded, tinycnn_arch.TinyCNNBN)
    assert font_sha256 is None
    assert input_mode == "binary"  # legacy checkpoints default to binary


def test_train_build_dataset_merges_charsets(tmp_path, font_path):
    from tools.dataset.generate_font_dataset import generate_dataset
    from tools.train.train import build_dataset

    synth = tmp_path / "synth.npz"
    real = tmp_path / "real.npz"
    generate_dataset(font_path, ["0", "1"], 2, synth, seed=1)
    generate_dataset(font_path, ["1", "A"], 2, real, seed=2)
    x, y, charset, font_sha256, input_mode = build_dataset(synth, real)
    assert charset == ["0", "1", "A"]
    assert x.shape[0] == y.shape[0]
    assert set(np.unique(y)) <= {0, 1, 2}
    assert font_sha256
    assert input_mode == "binary"


def test_export_folds_batchnorm(tmp_path):
    torch = __import__("torch")
    torch.manual_seed(3)
    model = tinycnn_arch.TinyCNNBN(5)
    model.eval()
    weights = {
        name: arr.detach().numpy()
        for name, arr in tinycnn_arch.export_folded_weights(model).items()
    }
    x = np.random.default_rng(4).random((3, 1, 24, 24), dtype=np.float32)
    ref = model(torch.from_numpy(x)).detach().numpy()
    got = forward(x, weights)
    assert np.abs(ref - got).max() < 1e-4
    assert got.argmax(1).tolist() == ref.argmax(1).tolist()


def test_export_runtime_model_and_vectors(tmp_path):
    torch = __import__("torch")
    torch.manual_seed(5)
    charset = list("01234")
    model = tinycnn_arch.TinyCNNBN(len(charset))
    model.eval()
    weights = {
        name: arr.detach().numpy()
        for name, arr in tinycnn_arch.export_folded_weights(model).items()
    }
    out = tmp_path / "runtime"
    write_cnn_model(out, charset, weights)
    make_test_vectors(weights, out, None, num_samples=4, seed=9)

    for name in ("config.json", "weights.bin", "charset.txt", "test_vectors.npz"):
        assert (out / name).exists(), name
    loaded = load_model(out)
    assert loaded.classifier == "tinycnn"
    assert loaded.charset == charset

    vec = np.load(out / "test_vectors.npz")
    x = vec["input"].astype(np.float32)[:, None] / 255.0
    acts = forward_with_activations(x, weights)
    assert np.abs(vec["conv1"] - acts["conv1"]).max() < 1e-6
    assert np.abs(vec["pw2"] - acts["pw2"]).max() < 1e-6
    assert np.abs(vec["logits"] - acts["logits"]).max() < 1e-6
    assert np.array_equal(vec["char_id"], acts["logits"].argmax(1).astype(np.int32))


def test_export_hybrid_runtime_model(tmp_path, font_path):
    torch = __import__("torch")
    torch.manual_seed(6)
    charset = list("0123456789")
    model = tinycnn_arch.TinyCNNBN(len(charset))
    model.eval()
    weights = {
        name: arr.detach().numpy()
        for name, arr in tinycnn_arch.export_folded_weights(model).items()
    }
    tmpl_dir = tmp_path / "template"
    chars, templates = build_templates(font_path, charset, render_size=32)
    write_model(tmpl_dir, chars, templates, font_path=font_path)

    out = tmp_path / "hybrid_runtime"
    from tools.train.export_model import load_checkpoint

    torch.save(
        {
            "state_dict": model.state_dict(),
            "charset": charset,
            "architecture": "tinycnn_bn",
            "input_size": 24,
            "version": 1,
        },
        tmp_path / "model.pth",
    )
    _, ck_charset, _, _ = load_checkpoint(tmp_path / "model.pth")
    assert ck_charset == charset
    from fixedfontocr import write_hybrid_model

    write_hybrid_model(out, ck_charset, templates, weights, font_path=font_path)
    loaded = load_model(out)
    assert loaded.classifier == "hybrid"
    assert loaded.templates is not None
    assert loaded.weights is not None
    assert (out / "templates.bin").exists()
