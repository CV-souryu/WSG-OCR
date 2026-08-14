#!/usr/bin/env python3
"""Build the bundled hybrid recognizer (template + TinyCNN) in one command.

Uses the project defaults from ``fixedfontocr.defaults``:

    font     = fonts/SourceHanSansSC/SourceHanSansSC-Bold.otf
    charset  = charsets/sets/combined.txt
    output   = model/game_cn                  (hybrid model)
    template = model/game_cn_template         (template half)

Steps: generate a synthetic dataset, train the TinyCNN, then export the
hybrid runtime model (template fast path + CNN fallback).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fixedfontocr import defaults  # noqa: E402


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--font",
        type=Path,
        default=defaults.FONT_PATH,
        help="registered font under fonts/ (default: bundled game font)",
    )
    parser.add_argument("--charset", type=Path, default=defaults.CHARSET_PATH)
    parser.add_argument("--template-output", type=Path, default=defaults.TEMPLATE_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=defaults.MODEL_PATH)
    parser.add_argument("--synthetic", type=Path, default=ROOT / "data" / "cn" / "synth_combined.npz")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data" / "cn" / "game_cn.pth")
    parser.add_argument("--samples-per-char", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", help="cpu, mps, cuda, or auto")
    parser.add_argument(
        "--soft",
        action="store_true",
        help="train the CNN on soft foreground glyphs (Goal 7 low-res domain)",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="reuse an existing synthetic npz instead of regenerating it",
    )
    parser.add_argument(
        "--tune-samples",
        type=Path,
        default=None,
        help="real-screenshot npz used to tune Goal 10 visual weights and "
        "calibration after export",
    )
    args = parser.parse_args()

    font = defaults.resolve_font(args.font)
    if not args.charset.is_file():
        parser.error(f"charset not found: {args.charset}")

    if not args.skip_dataset:
        gen_cmd = [
            sys.executable,
            str(ROOT / "tools" / "dataset" / "generate_font_dataset.py"),
            str(font),
            str(args.synthetic),
            "--charset",
            str(args.charset),
            "--samples-per-char",
            str(args.samples_per_char),
            # Goal 7: train on the real 10-18 px UI domain instead of a
            # single clean 32 px render.
            "--render-size-min",
            "10",
            "--render-size-max",
            "18",
            "--supersample-min",
            "2",
            "--supersample-max",
            "3",
        ]
        if args.soft:
            gen_cmd.append("--soft")
        run(gen_cmd)

    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_model.py"),
            str(font),
            str(args.template_output),
            "--charset",
            str(args.charset),
            "--render-size",
            "32",
        ]
    )

    run(
        [
            sys.executable,
            str(ROOT / "tools" / "train" / "train.py"),
            "--synthetic",
            str(args.synthetic),
            "--output",
            str(args.checkpoint),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
        ]
    )

    run(
        [
            sys.executable,
            str(ROOT / "tools" / "train" / "export_model.py"),
            str(args.checkpoint),
            "--templates",
            str(args.template_output),
            "--output",
            str(args.output),
            "--font",
            str(font),
            "--skip-test-vectors",
        ]
    )

    if args.tune_samples:
        run(
            [
                sys.executable,
                str(ROOT / "tools" / "train" / "tune_visual.py"),
                "--samples",
                str(args.tune_samples),
                "--model",
                str(args.output),
            ]
        )

    print(f"\nhybrid recognizer ready: {args.output}")


if __name__ == "__main__":
    main()
