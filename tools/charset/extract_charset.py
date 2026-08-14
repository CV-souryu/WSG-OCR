#!/usr/bin/env python3
"""Extract OCR charsets from the deduplicated word lists.

Run ``tools/charset/export_names.py`` first to regenerate the word lists
(one string per line in ``charsets/words/``) from the Unity ``JsonUtility``
config exports.  This script reads those word lists and writes one charset
per file as a single UTF-8 string under ``charsets/sets/``:

    words/ship_names.txt            -> sets/ship_names_charset.txt
    words/ship_names_harmonized.txt -> sets/ship_names_harmonized_charset.txt
    words/equipment_names.txt       -> sets/equipment_charset.txt
    words/ui_texts.txt              -> sets/ui_texts_charset.txt

It also writes:

    ui_texts_hanzi_charset.txt     - Han characters only from ui_texts
    ascii_letters.txt              - A-Z a-z
    digits.txt                     - 0-9
    ascii_punct.txt                - printable ASCII punctuation
    ascii.txt                      - letters + digits + punctuation
    all_charset.txt                - union of every word charset + ascii.txt
    combined.txt                   - same content as all_charset.txt
                                     (default alias)
"""

from __future__ import annotations

import argparse
import json
import string
from pathlib import Path


def load_unity_json(path: Path) -> dict:
    """Load a Unity JsonUtility export, quoting unquoted numeric keys."""
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            out.append(ch)
            i += 1
            while i < n:
                c = text[i]
                out.append(c)
                i += 1
                if c == "\\":
                    if i < n:
                        out.append(text[i])
                        i += 1
                elif c == '"':
                    break
        elif ch.isdigit() or (ch == "-" and i + 1 < n and text[i + 1].isdigit()):
            j = i + 1
            while j < n and text[j].isdigit():
                j += 1
            k = j
            while k < n and text[k] in " \t\r\n":
                k += 1
            if k < n and text[k] == ":":
                out.append('"')
                out.append(text[i:j])
                out.append('"')
                i = j
            else:
                out.append(ch)
                i += 1
        else:
            out.append(ch)
            i += 1
    return json.loads("".join(out))


def field_texts(data: dict, field: str) -> list[str]:
    seq = data.get("sequence")
    if not isinstance(seq, list):
        return []
    return [
        item[field]
        for item in seq
        if isinstance(item, dict) and isinstance(item.get(field), str) and item[field]
    ]


def unique_chars(texts: list[str]) -> str:
    chars = {ch for text in texts for ch in text if not ch.isspace()}
    return "".join(sorted(chars, key=ord))


def is_han(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x3400 <= cp <= 0x4DBF          # CJK Extension A
        or 0x4E00 <= cp <= 0x9FFF        # CJK Unified Ideographs
        or 0x20000 <= cp <= 0x2A6DF      # Extension B
        or 0x2A700 <= cp <= 0x2EBEF      # Extensions C-F
        or 0xF900 <= cp <= 0xFAFF        # CJK Compatibility Ideographs
        or 0x2F800 <= cp <= 0x2FA1F      # Compatibility Supplement
        or cp == 0x3007                  # 〇
    )


def write_charset(path: Path, chars: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chars + "\n", encoding="utf-8")


def read_words(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "charsets" / "words",
        help="directory containing the word lists (default: charsets/words)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "charsets" / "sets",
        help="output directory for charset files (default: charsets/sets)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    sources = {
        "ship_names_charset": "ship_names.txt",
        "ship_names_harmonized_charset": "ship_names_harmonized.txt",
        "equipment_charset": "equipment_names.txt",
        "ui_texts_charset": "ui_texts.txt",
    }

    extracted: dict[str, str] = {}
    for name, filename in sources.items():
        path = input_dir / filename
        if not path.is_file():
            parser.error(f"missing word list: {path} (run tools/charset/export_names.py first)")
        texts = read_words(path)
        chars = unique_chars(texts)
        extracted[name] = chars
        write_charset(output_dir / f"{name}.txt", chars)
        print(f"{filename:16s} {len(texts):5d} strings  {len(chars):5d} chars -> {name}.txt")

    language_hanzi = "".join(ch for ch in extracted["ui_texts_charset"] if is_han(ch))
    write_charset(output_dir / "ui_texts_hanzi_charset.txt", language_hanzi)

    ascii_punct = "".join(ch for ch in string.printable if not ch.isspace() and not ch.isalnum())
    ascii_all = string.ascii_letters + string.digits + ascii_punct
    write_charset(output_dir / "ascii_letters.txt", string.ascii_letters)
    write_charset(output_dir / "digits.txt", string.digits)
    write_charset(output_dir / "ascii_punct.txt", ascii_punct)
    write_charset(output_dir / "ascii.txt", ascii_all)

    allcharset = unique_chars(list(extracted.values()) + [ascii_all])
    write_charset(output_dir / "all_charset.txt", allcharset)
    write_charset(output_dir / "combined.txt", allcharset)

    print(f"{'ascii':15s} {'':10s} {'':5} strings  {len(ascii_all):5d} chars")
    print(f"{'all_charset':15s} {'':10s} {'':5} strings  {len(allcharset):5d} chars")
    print(f"{'combined':15s} {'':10s} {'':5} strings  {len(allcharset):5d} chars (alias)")
    print(f"wrote charsets to {output_dir}/")


if __name__ == "__main__":
    main()
