"""Custom model format: ``config.json`` + ``charset.txt`` + ``weights.bin``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cnn import (
    TINYCNN_V1_NAME,
    cnn_tensor_shapes,
    validate_v1_weights,
)
from .defaults import compute_font_sha256
from .preprocess import NormalizeSpec, compute_normalize_spec


@dataclass
class OCRModel:
    config: dict
    charset: list[str]
    templates: np.ndarray | None
    input_size: int
    classifier: str = "template"
    weights: dict[str, np.ndarray] | None = None
    font_sha256: str | None = None
    input_mode: str = "binary"
    normalize_spec: NormalizeSpec | None = None


# The tensor shape table (formerly CNN_TENSORS / cnn_tensor_shapes) now
# lives in fixedfontocr.cnn as the frozen Goal 5 V1 spec and is imported
# above so model loading/export can never drift from the forward pass.


def _load_charset(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    chars: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for ch in line:
            if ch.isspace():
                continue
            chars.append(ch)
    if not chars:
        raise ValueError(f"charset file {path} contains no characters")
    return chars


def load_model(model_path: Path) -> OCRModel:
    """Load a model directory produced by the template generator."""

    config_path = model_path / "config.json"
    charset_path = model_path / "charset.txt"
    weights_path = model_path / "weights.bin"
    if not config_path.exists():
        raise FileNotFoundError(f"missing {config_path}")
    if not charset_path.exists():
        raise FileNotFoundError(f"missing {charset_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"missing {weights_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    version = config.get("version", 1)
    if version != 1:
        raise ValueError(f"unsupported model version {version}")
    dtype = config.get("dtype", "f32")
    if dtype != "f32":
        raise ValueError(f"unsupported dtype {dtype!r} (only f32 is implemented)")
    input_size = int(config["input_width"])
    if int(config["input_height"]) != input_size:
        raise ValueError("input_width and input_height must match for the baseline")

    charset = _load_charset(charset_path)
    num_classes = int(config["classes"])
    if num_classes != len(charset):
        raise ValueError(
            f"config.classes={num_classes} but charset has {len(charset)} characters"
        )

    classifier = config.get("classifier", "template")
    if classifier not in ("template", "tinycnn", "hybrid"):
        raise ValueError(f"unsupported classifier {classifier!r}")
    architecture = config.get("architecture", TINYCNN_V1_NAME)
    if classifier in ("tinycnn", "hybrid") and architecture != TINYCNN_V1_NAME:
        raise ValueError(
            f"unsupported TinyCNN architecture {architecture!r}: only "
            f"{TINYCNN_V1_NAME} is frozen (Goal 5)"
        )
    input_mode = config.get("input_mode", "binary")
    if input_mode not in ("binary", "soft"):
        raise ValueError(f"unsupported input_mode {input_mode!r}")
    if classifier in ("tinycnn", "hybrid"):
        weights = _load_cnn_weights(weights_path, num_classes)
    else:
        weights = None
    if classifier == "hybrid":
        templates_path = model_path / "templates.bin"
        if not templates_path.exists():
            raise FileNotFoundError(f"hybrid model requires {templates_path}")
        templates = _load_template_weights(templates_path, num_classes, input_size)
    elif classifier == "template":
        templates = _load_template_weights(weights_path, num_classes, input_size)
    else:
        templates = None
    return OCRModel(
        config=config,
        charset=charset,
        templates=templates,
        input_size=input_size,
        classifier=classifier,
        weights=weights,
        font_sha256=config.get("font_sha256"),
        input_mode=input_mode,
        normalize_spec=NormalizeSpec.from_dict(config.get("normalize")),
    )


def _load_template_weights(
    path: Path, num_classes: int, input_size: int
) -> np.ndarray:
    """Load bitset templates: uint32 count, uint32 bytes_per_template, payload."""

    data = np.fromfile(path, dtype=np.uint8)
    if data.size < 8:
        raise ValueError(f"weights.bin too small: {data.size} bytes")
    count = int(data[:4].view("<u4")[0])
    bytes_per = int(data[4:8].view("<u4")[0])
    payload = data[8:]
    expected = num_classes * bytes_per
    if count != num_classes:
        raise ValueError(
            f"weights.bin has {count} templates but config declares {num_classes}"
        )
    if payload.size != expected:
        raise ValueError(
            f"weights.bin payload is {payload.size} bytes, expected {expected}"
        )
    if bytes_per != (input_size * input_size + 7) // 8:
        raise ValueError(
            f"weights.bin bytes_per_template={bytes_per} does not match "
            f"{input_size}x{input_size} bits"
        )
    rows = [payload[i * bytes_per : (i + 1) * bytes_per] for i in range(count)]
    return np.stack(rows)


def _load_cnn_weights(path: Path, num_classes: int) -> dict[str, np.ndarray]:
    """Load the fixed-order f32 CNN weights from ``weights.bin``."""

    data = np.fromfile(path, dtype="<f4")
    shapes = cnn_tensor_shapes(num_classes)
    expected = sum(int(np.prod(s)) for _, s in shapes)
    if data.size != expected:
        raise ValueError(
            f"weights.bin has {data.size} f32 values, expected {expected} "
            f"for {num_classes} classes"
        )
    weights: dict[str, np.ndarray] = {}
    offset = 0
    for name, shape in shapes:
        size = int(np.prod(shape))
        weights[name] = data[offset : offset + size].reshape(shape)
        offset += size
    validate_v1_weights(weights, num_classes)
    return weights


def write_cnn_weights(
    path: Path, weights: dict[str, np.ndarray], num_classes: int
) -> None:
    """Write CNN weights in the fixed tensor order expected by the loader."""

    validate_v1_weights(weights, num_classes)
    input_channels = int(np.asarray(weights["conv1.weight"]).shape[1])
    if input_channels != 1:
        raise ValueError(
            "the TinyCNN V1 model format is frozen to 1 input channel; "
            "2-channel soft+binary is an in-memory experiment only"
        )
    tensors: list[np.ndarray] = []
    for name, shape in cnn_tensor_shapes(num_classes):
        if name not in weights:
            raise ValueError(f"missing tensor {name}")
        arr = np.asarray(weights[name], dtype="<f4")
        if arr.shape != shape:
            raise ValueError(
                f"tensor {name} has shape {arr.shape}, expected {shape}"
            )
        tensors.append(arr.reshape(-1))
    np.concatenate(tensors).tofile(path)


def write_cnn_model(
    model_dir: Path,
    chars: list[str],
    weights: dict[str, np.ndarray],
    input_size: int = 24,
    font_path: str | Path | None = None,
    font_sha256: str | None = None,
    input_mode: str = "binary",
    normalize_spec: dict | None = None,
    render_size: int = 32,
) -> None:
    """Write ``config.json`` + ``charset.txt`` + CNN ``weights.bin``.

    When ``font_path`` is provided its SHA256 is stored in the model
    metadata, unless an explicit ``font_sha256`` is given. ``input_mode``
    records which glyph representation the CNN was trained with
    (``"binary"`` 0/255 masks or ``"soft"`` 0..255 foreground strength).
    """

    if input_mode not in ("binary", "soft"):
        raise ValueError(f"unsupported input_mode {input_mode!r}")
    model_dir.mkdir(parents=True, exist_ok=True)
    if font_path is not None and font_sha256 is None:
        font_sha256 = compute_font_sha256(font_path)
    config = {
        "input_width": input_size,
        "input_height": input_size,
        "classes": len(chars),
        "version": 1,
        "dtype": "f32",
        "classifier": "tinycnn",
        "architecture": TINYCNN_V1_NAME,
        "input_mode": input_mode,
    }
    if font_sha256:
        config["font_sha256"] = font_sha256
    if normalize_spec is None and font_path is not None:
        normalize_spec = compute_normalize_spec(
            font_path, chars, input_size, render_size
        ).to_dict()
    if normalize_spec:
        config["normalize"] = normalize_spec
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=4) + "\n",
        encoding="utf-8",
    )
    (model_dir / "charset.txt").write_text(
        "".join(chars) + "\n",
        encoding="utf-8",
    )
    write_cnn_weights(model_dir / "weights.bin", weights, len(chars))


def write_hybrid_model(
    model_dir: Path,
    chars: list[str],
    templates: np.ndarray,
    weights: dict[str, np.ndarray],
    input_size: int = 24,
    template_threshold: float = 0.90,
    template_margin_threshold: float = 0.04,
    cnn_threshold: float = 0.0,
    font_path: str | Path | None = None,
    font_sha256: str | None = None,
    input_mode: str = "binary",
    normalize_spec: dict | None = None,
    render_size: int = 32,
) -> None:
    """Write a hybrid model: template level-1 + TinyCNN level-2.

    Layout: ``config.json`` (``classifier: "hybrid"``), ``charset.txt``,
    ``templates.bin`` (bitset templates, same payload as a template model's
    ``weights.bin``) and ``weights.bin`` (f32 CNN tensors). The runtime
    pipeline tries the template first and only runs the CNN on glyphs whose
    template confidence is below ``template_threshold``.
    """

    if input_mode not in ("binary", "soft"):
        raise ValueError(f"unsupported input_mode {input_mode!r}")
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if font_path is not None and font_sha256 is None:
        font_sha256 = compute_font_sha256(font_path)
    config = {
        "input_width": input_size,
        "input_height": input_size,
        "classes": len(chars),
        "version": 1,
        "dtype": "f32",
        "classifier": "hybrid",
        "architecture": TINYCNN_V1_NAME,
        "template_threshold": float(template_threshold),
        "template_margin_threshold": float(template_margin_threshold),
        "cnn_threshold": float(cnn_threshold),
        "input_mode": input_mode,
    }
    if font_sha256:
        config["font_sha256"] = font_sha256
    if normalize_spec is None and font_path is not None:
        normalize_spec = compute_normalize_spec(
            font_path, chars, input_size, render_size
        ).to_dict()
    if normalize_spec:
        config["normalize"] = normalize_spec
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=4) + "\n",
        encoding="utf-8",
    )
    (model_dir / "charset.txt").write_text(
        "".join(chars) + "\n",
        encoding="utf-8",
    )
    if templates.shape[1] != (input_size * input_size + 7) // 8:
        raise ValueError("template byte width does not match input_size")
    if templates.shape[0] != len(chars):
        raise ValueError("templates and charset must have the same length")
    header = np.array([len(chars), templates.shape[1]], dtype="<u4")
    payload = np.concatenate([header.view(np.uint8), np.asarray(templates).reshape(-1)])
    (model_dir / "templates.bin").write_bytes(payload.tobytes())
    write_cnn_weights(model_dir / "weights.bin", weights, len(chars))
