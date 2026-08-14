#!/usr/bin/env python3
"""Export unique game strings from the CN config dumps as word lists.

The config JSONs are Unity ``JsonUtility`` exports with unquoted numeric dict
keys, so they are parsed with the tolerant scanner in
:mod:`extract_charset`.  One unique, non-empty string is written per line:

    charsets/words/ship_names.txt             - ship_h.json ``title``
                                                (original ship names)
    charsets/words/ship_names_harmonized.txt  - ship.json ``title``
                                                (harmonized ship names)
    charsets/words/equipment_names.txt        - equip.json ``title``
                                                (equipment names)
    charsets/words/ui_texts.txt               - language.json ``schinese``
                                                (Chinese UI strings)

The word lists keep first-seen order from the source JSONs and are consumed by
``tools/charset/extract_charset.py`` to build the OCR charsets.
Rich-text color codes like ``^C656565FF00000000`` are stripped from the
strings before deduplication.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

if __package__:
    from .extract_charset import load_unity_json
else:
    from extract_charset import load_unity_json


CHINESE_FIELD = "schinese"
COLOR_CODE = re.compile(r"\^C[0-9A-Fa-f]{8}(?:[0-9A-Fa-f]{8})?")


def strip_color_codes(text: str) -> str:
    """Remove rich-text ``^C`` color codes (8 or 16 hex digits)."""
    return COLOR_CODE.sub("", text)


def unique_strings(texts: list[str]) -> list[str]:
    """Return non-empty strings, deduplicated in first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        if isinstance(text, str) and text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def write_words(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(unique_strings(texts)) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "cn",
        help="directory containing the four config JSONs (default: data/cn)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "charsets" / "words",
        help="output directory for word lists (default: charsets/words)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    sources = {
        "ship_names.txt": ("ship_h.json", "title"),
        "ship_names_harmonized.txt": ("ship.json", "title"),
        "equipment_names.txt": ("equip.json", "title"),
    }
    for out_name, (filename, field) in sources.items():
        path = input_dir / filename
        if not path.is_file():
            parser.error(f"missing input file: {path}")
        data = load_unity_json(path)
        texts = [
            item[field]
            for item in data.get("sequence", [])
            if isinstance(item, dict) and isinstance(item.get(field), str)
        ]
        words = unique_strings(texts)
        write_words(output_dir / out_name, words)
        print(f"{filename:16s} {field:10s} {len(words):5d} unique strings -> {out_name}")

    language_path = input_dir / "language.json"
    if not language_path.is_file():
        parser.error(f"missing input file: {language_path}")
    language_data = load_unity_json(language_path)
    language_texts = []
    for item in language_data.get("sequence", []):
        if not isinstance(item, dict):
            continue
        text = item.get(CHINESE_FIELD)
        if isinstance(text, str):
            language_texts.append(strip_color_codes(text))
    language_words = unique_strings(language_texts)
    write_words(output_dir / "ui_texts.txt", language_words)
    print(f"{'language.json':16s} {'schinese':10s} {len(language_words):5d} unique strings -> ui_texts.txt")

    print(f"wrote word lists to {output_dir}/")


if __name__ == "__main__":
    main()
