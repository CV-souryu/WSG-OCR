"""Project-wide defaults: the bundled game font and its CN charset.

The charset is extracted from the CN game config dumps by
``tools/charset/export_names.py`` + ``tools/charset/extract_charset.py``;
``combined.txt`` (alias of ``all_charset.txt``) covers ship names, equipment
names, Chinese UI copy, and the ASCII letters/digits/punctuation set.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FONTS_DIR = PROJECT_ROOT / "fonts"
FONT_REGISTRY_PATH = FONTS_DIR / "registry.json"
FONT_PATH = PROJECT_ROOT / "fonts" / "SourceHanSansSC" / "SourceHanSansSC-Bold.otf"
CHARSET_PATH = PROJECT_ROOT / "charsets" / "sets" / "combined.txt"
MODEL_PATH = PROJECT_ROOT / "model" / "game_cn"
TEMPLATE_MODEL_PATH = PROJECT_ROOT / "model" / "game_cn_template"

# Goal 19: CPU Freeze.
#
# The numpy CPU pipeline is the canonical implementation of the OCR
# contract: the model format, the public API, the regression corpus and the
# benchmark suite below define the reference that any optional backend
# (e.g. WGPU, Goal 20) must match. Formal WGPU work starts only after this
# freeze; ``CPU_FREEZE_CONDITIONS`` mirrors the ten checkboxes in
# ``fonts/goal`` and each one is pinned by ``tests/test_goal19_cpu_freeze.py``.
CPU_FREEZE = True
CPU_FREEZE_VERSION = "1.0"
CPU_FREEZE_DATE = "2026-08-15"
CPU_FREEZE_CONDITIONS = (
    "segmentation_lattice_stable",
    "fragmented_glyphs_solved",
    "low_res_z17_stable",
    "baltimore_stable",
    "mixed_charset_stable",
    "topk_api_stable",
    "lexicon_decoder_stable",
    "partial_word_stable",
    "cpu_benchmark_complete",
    "regression_dataset_complete",
)


@functools.lru_cache(maxsize=64)
def compute_font_sha256(font_path: str | Path) -> str:
    """Return the lowercase SHA256 of a font file."""
    return hashlib.sha256(Path(font_path).read_bytes()).hexdigest()


def _load_font_registry() -> dict[str, str]:
    """Load ``fonts/registry.json`` as a relative-path -> SHA256 map."""
    if not FONT_REGISTRY_PATH.is_file():
        return {}
    data = json.loads(FONT_REGISTRY_PATH.read_text(encoding="utf-8"))
    return dict(data.get("fonts", {}))


def ensure_font_path(
    font_path: str | Path, require_registered: bool = True
) -> Path:
    """Resolve a font path and require it to live inside ``fonts/``.

    If a registry exists, the font's relative path must also be present in
    it. The registry entry's SHA256 is only verified by ``resolve_font`` so
    per-glyph rendering stays cheap.
    """

    path = Path(font_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"font not found: {resolved}")
    if not resolved.is_relative_to(FONTS_DIR):
        raise ValueError(
            f"font must be inside {FONTS_DIR}/, got {resolved}"
        )
    rel = resolved.relative_to(FONTS_DIR).as_posix()
    registry = _load_font_registry()
    if require_registered and registry and rel not in registry:
        raise ValueError(
            f"font {rel!r} is not registered; run tools/register_font.py"
        )
    return resolved


def resolve_font(font_path: str | Path) -> Path:
    """Return a font path after checking ``fonts/`` membership and SHA256."""

    resolved = ensure_font_path(font_path)
    rel = resolved.relative_to(FONTS_DIR).as_posix()
    registry = _load_font_registry()
    if registry:
        expected = registry.get(rel)
        if expected is None:
            raise ValueError(
                f"font {rel!r} is not registered; run tools/register_font.py"
            )
        actual = compute_font_sha256(resolved)
        if actual != expected:
            raise ValueError(
                f"font {rel!r} changed (sha256 {actual[:12]}... != "
                f"{expected[:12]}...); re-run tools/register_font.py"
            )
    return resolved


def read_charset(path: Path | str = CHARSET_PATH) -> str:
    """Return the charset file contents with all whitespace removed."""
    raw = Path(path).read_text(encoding="utf-8")
    return "".join(ch for ch in raw if not ch.isspace())
