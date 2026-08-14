# 游戏文案字符集

本目录按用途分成两个子目录，全部使用英文文件名（无空格）：

```text
charsets/
├── words/    # 生成的一行一个词的词表（由 export_names.py 输出）
└── sets/     # 单行字符集：每个文件是一行 UTF-8 去重字符
```

## 生成流程

先由 `tools/charset/export_names.py` 从 `data/cn/` 下的 CN 配置 JSON 导出
去重词表（一行一个词）到 `charsets/words/`，再由
`tools/charset/extract_charset.py` 根据这些词表生成 `charsets/sets/` 下的
字符集：

```sh
python3 tools/charset/export_names.py
python3 tools/charset/extract_charset.py
```

字符集文件是一行 UTF-8 字符串，可直接传给
`tools/dataset/generate_font_dataset.py --charset`。

## sets/：单行字符集

| 文件 | 来源 | 唯一字符数 |
| --- | --- | ---: |
| `ship_names_charset.txt` | `words/ship_names.txt`（`ship_h.json` 的 `title`，船名） | 692 |
| `ship_names_harmonized_charset.txt` | `words/ship_names_harmonized.txt`（`ship.json` 的 `title`，和谐船名） | 761 |
| `equipment_charset.txt` | `words/equipment_names.txt`（`equip.json` 的 `title`，装备名） | 623 |
| `ui_texts_charset.txt` | `words/ui_texts.txt`（`language.json` 的 `schinese` 字段） | 1370 |
| `ui_texts_hanzi_charset.txt` | `ui_texts_charset` 中仅汉字 | 1249 |
| `ascii_letters.txt` | 标准 ASCII A-Z a-z | 52 |
| `digits.txt` | 标准 ASCII 0-9 | 10 |
| `ascii_punct.txt` | 标准 ASCII 英文标点 | 32 |
| `ascii.txt` | 英文 + 数字 + 标点 | 94 |
| `all_charset.txt` | 以上全部去重合并 | 1894 |
| `combined.txt` | `all_charset.txt` 的别名（项目默认） | 1894 |

## words/：一行一个词

| 文件 | 来源 | 唯一字符串数 |
| --- | --- | ---: |
| `ship_names.txt` | `ship_h.json` 的 `title`（船名） | 724 |
| `ship_names_harmonized.txt` | `ship.json` 的 `title`（和谐船名） | 779 |
| `equipment_names.txt` | `equip.json` 的 `title`（装备名） | 814 |
| `ui_texts.txt` | `language.json` 的 `schinese` 字段（中文 UI 文案，已剔除颜色代码） | 2906 |

说明：

- 词表保留源 JSON 首次出现的顺序，同一字符串只出现一次。
- `ui_texts.txt` 只保留中文 UI 文案；其他语言字段不再导出。
- 导出时会剔除 `^C` 开头的富文本颜色代码（如 `^C656565FF00000000`），
  只保留可见文案。
- 字符集保留所有非空白字符（含数字、英文、标点和日文/韩文/俄文符号等），
  因为这些字符会直接显示在游戏 UI 上。
- `combined.txt` 是项目默认字符集：
  `src/fixedfontocr/defaults.py`、`scripts/generate_model.py`、
  `tools/dataset/generate_font_dataset.py` 和 `tools/train/build_model.py`
  都会在未显式指定 `--charset` 时使用它。
