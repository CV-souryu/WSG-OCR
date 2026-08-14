#!/usr/bin/env python3
"""Export a trained checkpoint to the runtime model directory.

The training network has BatchNorm; this exporter folds every BN layer into
the preceding convolution's weight and bias, so runtime inference only needs
Conv + ReLU (both numpy and WGSL).

Output:

    runtime_model/
    ├── model.json      # alias of config.json (runtime format)
    ├── config.json     # canonical FixedFontOCR config
    ├── weights.bin     # f32 CNN tensors (BN already folded)
    ├── charset.txt
    └── test_vectors.npz
        # input, every layer's activations, final logits and char_id;
        # WGPU unit tests compare against these directly.

Usage:
    python tools/train/export_model.py model.pth --output runtime_model
    python tools/train/export_model.py model.pth --templates template_model/ \\
        --output hybrid_model/ --template-threshold 0.9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from tools.train import tinycnn_arch  # noqa: E402
from fixedfontocr.cnn import forward_with_activations  # noqa: E402
from fixedfontocr.model import (  # noqa: E402
    load_model,
    write_cnn_model,
    write_hybrid_model,
)


def load_checkpoint(
    path: str | Path,
) -> tuple[tinycnn_arch.TinyCNNBN, list[str], str | None, str]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("architecture") != "tinycnn_bn":
        raise SystemExit(f"unsupported checkpoint architecture: {ckpt.get('architecture')}")
    charset = [str(c) for c in ckpt["charset"]]
    model = tinycnn_arch.TinyCNNBN(len(charset))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, charset, ckpt.get("font_sha256"), ckpt.get("input_mode", "binary")


def make_test_vectors(
    weights: dict[str, np.ndarray],
    output: Path,
    samples_npz: str | Path | None,
    num_samples: int = 8,
    seed: int = 0,
) -> None:
    """Write input/layer-output/logits/char_id vectors for WGPU tests."""
    if samples_npz is not None:
        data = np.load(samples_npz)
        x = np.asarray(data["x"], dtype=np.uint8)[:num_samples]
        if x.shape[1:] != (24, 24):
            raise SystemExit(f"expected 24x24 samples, got {x.shape[1:]}")
    else:
        rng = np.random.default_rng(seed)
        x = (rng.random((num_samples, 24, 24)) > 0.5).astype(np.uint8) * 255

    acts = forward_with_activations(
        x.astype(np.float32)[:, None, :, :] * (1.0 / 255.0), weights
    )
    np.savez_compressed(
        output / "test_vectors.npz",
        input=x,
        conv1=acts["conv1"],
        dw1=acts["dw1"],
        pw1=acts["pw1"],
        dw2=acts["dw2"],
        pw2=acts["pw2"],
        gap=acts["gap"],
        logits=acts["logits"],
        char_id=acts["logits"].argmax(axis=1).astype(np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="model.pth from tools/train.py")
    parser.add_argument("--output", default="runtime_model", help="output model dir")
    parser.add_argument(
        "--templates",
        help="template model dir to combine into a hybrid model "
        "(same charset required)",
    )
    parser.add_argument("--template-threshold", type=float, default=0.90)
    parser.add_argument(
        "--template-margin-threshold",
        type=float,
        default=0.04,
        help="minimum normalized template top-1/top-2 margin before a "
        "template match is trusted (P6 margin-aware hybrid gate)",
    )
    parser.add_argument("--cnn-threshold", type=float, default=0.0)
    parser.add_argument("--samples", help="npz whose x samples become test vectors")
    parser.add_argument("--num-test-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-test-vectors",
        action="store_true",
        help="do not write test_vectors.npz (smaller deployed runtime dir)",
    )
    args = parser.parse_args()

    model, charset, ckpt_sha, input_mode = load_checkpoint(args.checkpoint)
    font_sha256 = ckpt_sha
    weights = {
        name: arr.detach().numpy()
        for name, arr in tinycnn_arch.export_folded_weights(model).items()
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.templates:
        tmpl = load_model(Path(args.templates))
        if tmpl.classifier != "template":
            raise SystemExit("--templates must point at a template model")
        if tmpl.charset != charset:
            raise SystemExit("template charset differs from the CNN checkpoint")
        font_sha256 = font_sha256 or tmpl.config.get("font_sha256")
    if not font_sha256:
        raise SystemExit(
            "checkpoint/template has no font_sha256; retrain with the "
            "current pipeline before exporting"
        )
    normalize_spec = None
    if args.templates:
        normalize_spec = tmpl.config.get("normalize")
    if args.templates:
        write_hybrid_model(
            out,
            charset,
            tmpl.templates,
            weights,
            input_size=24,
            template_threshold=args.template_threshold,
            template_margin_threshold=args.template_margin_threshold,
            cnn_threshold=args.cnn_threshold,
            font_sha256=font_sha256,
            input_mode=input_mode,
            normalize_spec=normalize_spec,
        )
    else:
        write_cnn_model(
            out,
            charset,
            weights,
            input_size=24,
            font_sha256=font_sha256,
            input_mode=input_mode,
        )

    # The plan's runtime layout names it model.json; keep config.json too
    # for the existing loader.
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    (out / "model.json").write_text(
        json.dumps(config, indent=4) + "\n", encoding="utf-8"
    )

    if not args.skip_test_vectors:
        make_test_vectors(
            weights,
            out,
            args.samples,
            num_samples=args.num_test_samples,
            seed=args.seed,
        )
        print("  test_vectors.npz: input, layer activations, logits, char_id")
    else:
        print("  test_vectors.npz: skipped (--skip-test-vectors)")
    print(f"wrote runtime model ({len(charset)} classes) to {out}")
    print(f"  classifier: {config['classifier']}")


if __name__ == "__main__":
    main()
