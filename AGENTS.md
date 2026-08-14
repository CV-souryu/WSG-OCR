# Project Rules

本文件是 SuperFeatureOCR 的项目级规则。所有 AIAgent 和开发者在修改本仓库前，
应先阅读并遵守本文件。

## Font Policy

1. `Fonts/` is the only font source.
2. No system-font discovery.
3. No fallback fonts.
4. Missing glyph = explicit error.
5. Training uses only `Fonts/`.
6. Font identity is stored by SHA256 in model metadata.
7. Accuracy outside registered `Fonts/` is out of scope.
8. Rendering augmentation is allowed; font-family augmentation is forbidden.

> 本仓库中的字体目录为 `fonts/`（见 README 与 `.gitignore`），上述规则中的
> `Fonts/` 即指该目录。
>
> 实现入口：`fonts/registry.json` 保存注册字体 SHA256；`defaults.resolve_font()`
> 统一校验；模板/训练/导出链路会把 `font_sha256` 写入模型元数据。
