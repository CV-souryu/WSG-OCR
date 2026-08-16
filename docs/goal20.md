# Goal 20: WGPU phase 2 — 设计与任务分解

> 状态：准备阶段完成（本文档 + `tests/test_goal20_wgpu.py` 契约骨架，
> 2026-08-16）。目标定义见 `fonts/goal` Goal 20；phase 1 架构与基准见
> [`wgpu.md`](wgpu.md)。

## 1. 目标与冻结约束

Goal 20 只把适合 GPU 的部分搬上 GPU：

```text
normalized glyph batch → TinyCNN → logits / Top-K        （GPU）
CC / Visual Lattice / Lexicon / Beam Search / OCRResult   （CPU）
```

约束（基准 = 当前仓库 CPU 实现）：

1. parity 基准是**当前仓库 HEAD 的 CPU 实现**，不是 Goal 19 的
   2026-08-15 版本 1.0 快照。准备时 HEAD =
   `2f46da72d51834d5b4f6f51fa47b574c7fb4c86d`（"perf: fix coarse-filter
   feature unpack axis"），Goal 19 freeze 之后 CPU 又落地了 7 个提交：
   `cb36be4`（单次分类/后端边界）、`9dab45b`（touching-glyph 分割）、
   `6e7c405`（dict 真实字形仲裁）、`e449580`/`cc690c2`/`9df06ce`/
   `2f46da7`（模板流式与 CPU perf）。`CPU_FREEZE = True` 仍是代码
   事实（CPU 为规范实现），但 WGPU 的参考点跟随 HEAD。
2. 任何 WGPU 改动不得改变 `backend="cpu"` 的输出：以当前全量回归
   （631 tests + 235-crop corpus + game_samples）在改动前后的 CPU 输出
   对比为准；`tests/test_goal20_wgpu.py` 另用固定权重的 CPU 黄金值
   pin 当前 CPU 内容（CPU 有意变更时先更新黄金值，再对 WGPU 重新
   parity）。
3. WGPU 可选：无 adapter / 无 `wgpu` 时测试 skip、`backend="auto"` 回退
   CPU（现有行为不变）。
4. 数值契约沿用 phase 1：FP32、每层 max abs error < 1e-4、argmax 100%
   一致、端到端 `backend="wgpu"` == `backend="cpu"`。

## 2. 现状盘点（phase 1 已完成，2026-08-16 复核）

| 项 | 现状 | 证据 |
| --- | --- | --- |
| Backend 接口 | `classify()` `[N,24,24]u8 → BackendResult(char_ids, scores)`；`forward_logits()` `→ [N,C] f32` | `src/fixedfontocr/backends.py:48-135` |
| WGPUBackend | 8 dispatch：normalize → conv1 → dw1 → pw1 → dw2 → pw2 → gap → linear_argmax；ReLU 已内联在各 conv shader；batch buffers 按容量持久化（~24.9 KB/glyph）；classify 读回 12 B/glyph、单次 map_sync | `backends.py:136-574` |
| forward_logits（DP 使用） | 走 8 个 per-layer 入口，**每个入口一次 `queue.submit` + 一次 `map_sync` + 新建 staging buffer**（≈8× 同步地板）；读回全 `[N,C]` | `backends.py:295-318、557-574` |
| 生产 DP 接线 | `FixedFontOCR` 构造 backend → 注入 `SegmentScorer`（P0-4 边界）→ `_cnn_batch` → `forward_logits` → CPU 侧 top2/top5/allowed 掩码/fusion | `api.py:83-93`、`scorer.py:908-946` |
| parity 基线 | `test_wgpu.py` + `test_wgpu_vectors.py` + `test_auto_backend.py` = 28 passed | 本机实测 2.60 s |
| 性能事实 | map_sync 地板 ~1.3-1.5 ms/call；compute ~0.1 ms；crossover（game_cn 1894 类）≈ batch 32；全 logits 读回 = N×C×4 B（40 候选 ≈ 300 KB/行） | `docs/wgpu.md` 基准 |

### 盘点结论（差距）

正确性上 phase 1 已达标，差距全部在 **DP 路径实际使用 WGPU 时的效率**：

1. `forward_logits` 每次调用 8 次 GPU 同步（≈ 8 × 1.4 ms ≈ 11 ms），DP 在
   GPU 上比 CPU 慢一个量级 → **T1/T4 是 DP 真正吃到 GPU 收益的前置条件**。
2. 全 logits 读回 `N×C×4` 字节（game_cn 一行 40 候选 ≈ 300 KB）→ T2。
3. CPU 侧 soft 图计算 + 逐 ROI normalize + glyph 上传 → T3。

## 3. 任务分解

### T1 — shader fusion（8 → 4 dispatch）✅ 完成（升级为 mega 单 dispatch）

原方案：8 个 dispatch 合并为 4 个 fused pass。实际落地采用了更强的
形式：**`mega.wgsl` 单 dispatch** —— 每 workgroup（64 线程）算一个
glyph 的整条 TinyCNN（normalize→conv1→dw1→pw1→dw2→pw2→gap→dense→
top-2 / 全 logits），中间量全部在 workgroup shared memory（15.6 KB），
host 零中间拷回：

- `classify()`：1 个 dispatch，读回 12 B/glyph；
- `forward_logits()`：1 个 dispatch，读回 `[N,C]`（DP 用）；
- 每层 f32 累加顺序逐层复刻 per-layer shader，28 个 parity 测试全绿
  （err ~2-3.4e-5）；
- 性能：`forward_logits` 12-17 ms（8 次 sync）→ 1.4-2.7 ms；
  classify compute 列 ~130 µs → ~33-50 µs；持久缓冲 24.9 KB/glyph →
  ~0.7-15.8 KB/glyph（按 C）。
- per-layer 入口原样保留作为 parity 原语；`last_dispatch_count` 观测
  字段已加，契约测试 pin `== 1`。

### T2 — DP 候选 Top-K 回灌（`classify_topk` + `logits_for`）

现状：DP 需要全 `[N,C]` logits —— Top-5、allowed_ids 掩码，以及
`_fuse_scores` 对 template-only id 取 logit（`scorer.py:264-308`）。

目标：每 glyph 只读回小记录：

1. `Backend.classify_topk(glyphs, allowed_mask=None)`：新 WGSL
   `linear_topk.wgsl`（per-glyph workgroup：全 C logits → 按 bitmask
   掩码 → Top-K 顺序归约），读回 `K × (u32 + f32)`/glyph。CPU 参考 =
   numpy topk over `forward_logits`（掩码语义一致）。
2. `Backend.logits_for(glyphs, char_ids)`：`[N,M]` 定点 gather
   （`logits_for.wgsl`：只算指定 (glyph, class) 的 dot），供 hybrid
   fusion 取 template-only id 的 logit。
3. `SegmentScorer._cnn_batch` 在后端提供能力时切换为上述两调用，
   fusion 逻辑逐行不变（`classify_topk` 缺省实现退回全 logits，模板
   模型无 CNN 不受影响）。

Top-K 边界保护：WGSL 额外读回第 k+1/k+2 名 logit；CPU 发现
`|logit_k − logit_{k+2}| < 1e-4` 时该批回退全 logits 读回（正确性
优先，罕见路径）。

验收：`classify_topk`/`logits_for` 与 CPU 参考 parity（ids 一致、
logits <1e-4）；hybrid 端到端 wgpu == cpu（digits/CJK/game_cn）；
读回字节量从 `N×C×4` 降到 ~`N×(K×8+12)`。

风险：Top-K 并列次序 CPU（argpartition）vs GPU（顺序 scan）可能不同 →
端到端 parity 测试把关 + 边界保护回退。

### T3 — GPU preprocessing（先 soft normalize）

目标（`wgpu.md` 3a）：RGB 图像一次上传，ROI crop + 灰度 +
nearest-neighbor resize + baseline 放置全在 WGSL，CPU 只留 CC/bbox。

1. 先 pin CPU 参考公式：`profile.soft_foreground(image)` 的逐像素公式
   （default profile 为灰度混合，实现时逐 profile 确认）；
   `normalize_grayscale` 的 `_resample` 是最近邻（`int(i*h/new_h)`
   截断，整数值 < 2^24 在 f32 精确 → WGSL 可 bit-exact）。
2. `preprocess_glyphs.wgsl`：storage RGB `[H*W*3]u8` + per-glyph ROI
   参数（x/y/w/h/baseline_offset/scale/baseline_row）+ 输出
   `uint8 [N,24,24]`（或直接写 f32 NHWC 输入缓冲，与 T1 stem 合并省
   一次 dispatch）。
3. API：`WGPUBackend.preprocess_glyphs(image, segments, spec)`，与 CPU
   `_soft_glyph_batch` 输出逐字节一致。

范围边界：binary mask / color_mask（HSL profile）留在 CPU（CC 本来
就要它）；soft 路径逐 profile 提供能力开关，未支持的 profile 回退
CPU 路径。

验收：uint8 输出与 CPU 逐字节一致（合成 ROI + 真实渲染两种输入）；
端到端 parity 不变。

风险：传输量上 RGB 整图（H×W×3）大于 glyph batch；3a 的真实收益是
CPU offload + 管线简化（tracker 多帧场景整图上传可摊销），基准必须
诚实记录传输量与每帧耗时。

### T4 — persistent/staged readback（摊薄同步地板）

现状：`classify` 已持久 staging + 单 sync；`forward_logits` 每层新建
staging + 每层 sync。

1. `forward_logits` 改为单 encoder：整条链一个 command encoder，末尾
   一次 copy 到持久 staging → 单次 `map_sync`；staging 按容量复用。
2. `staged=True`：双缓冲 readback + `map_async`：等待第 N 帧 map 的
   同时提交第 N+1 帧 compute（tracker/连续帧场景把 ~1.4 ms 地板摊到
   帧间）。
3. `benchmark_wgpu.py` 新增连续帧场景（如 10 帧同 ROI recognize），
   记录 per-frame steady-state。

验收：`classify(..., staged=True)` / `forward_logits(..., staged=True)`
与同步路径结果一致；连续帧 benchmark 每帧耗时下降；parity 全绿。

风险：wgpu-py 同步 API 限制（`map_async` 可用但轮询/回调需自管）；
Metal 单设备验证。

## 4. 全链路 GPU 方向（G2-G4，继 mega 之后）

用户指令升级：**全链路塞入 GPU，1 次发射出结果，0 CPU 中间拷回**。
mega（G1）已把 TinyCNN 整链做成单 dispatch。剩余三段：

- **G2 模板匹配进 GPU** ✅ 完成：`shaders/template_match.wgsl` 单
  dispatch、每候选 glyph 一个 workgroup（92.8k prototypes 分摊到 64
  线程：coarse filter + XOR/popcount + per-char min + ink-band 回扫 +
  确定性 Top-K + 胜者 prototype），输出与 CPU `TemplateV2Classifier`
  **逐字段精确相等**（距离是精确整数）。CPU 参考同步修正两处：
  Top-K 并列用 (dist, char_id) 升序确定化（原 argpartition 并列任选），
  以及 kk<k 时 padding 从 `[kk:]`（原 `[k:]` 留下 np.empty 垃圾 id）。
  `WGPUTemplateMatcher.match_batch` 单 dispatch 读回 60B/glyph；
  `backend="wgpu"` 恒用 GPU，`backend="auto"` 经
  `AutoTemplateMatcher` 按 batch 实测选路（crossover = batch 4）。
  实测（game_cn）：模板段 batch 4/8/16/32 = 1.3x/1.8x/4.8x/5.7x；
  **端到端 recognize：GPU 3.2-4.7x（auto 3.4-4.7x），首次真正领先
  CPU**。测试：`tests/test_goal20_template_gpu.py` 10 个（含逐字段
  精确 parity、allowed 子集、auto 包装、game_cn 端到端）。
- **G3 预处理进 GPU** ✅ 完成（soft 路径）：`shaders/preprocess_soft.wgsl`
  单 dispatch —— RGB 整图一次上传，ROI crop + 灰度（default profile
  f32 公式）+ 最近邻 resize + baseline 放置全在 WGSL，直接写 mega 的
  packed 输入格式；`WGPUBackend.preprocess_glyphs` 与 CPU
  `_soft_glyph_batch` **逐字节一致**（real crops 上验证），
  `forward_logits_from_image` 把 preprocess + mega 合入 **1 个 submit
  （2 dispatch）**、单次最终读回。`backend="wgpu"` 的 soft-mode 模型
  不再构建 CPU soft 图（dict 仲裁按需回退）。注意：game_cn 是
  binary input_mode，crops 上的收益来自 G1/G2（见下方实测）。
  测试：`tests/test_goal20_crops_gpu.py`（byte parity、fused logits
  parity、pinned + 全语料 dict 模式 parity）。
- **G4 视觉 DP 解码进 GPU** — 重新评估后**有意不做**，改为
  **评分链单次发射** ✅：`decode_dp` 用 f64 算术 + 1e-12 tie 容差
  （`score > prev + 1e-12` 比较），f32 WGSL 无法复刻该比较语义
  （近并列路径会选错），且 DP 本身是微秒级。fonts/goal 的边界本来
  就把 Decoder 留给 CPU。真正的收口是评分链：
  `WGPUScoringStage.score_line` 把模板 dispatch + mega-logits dispatch
  记入**同一个 command encoder**，模板记录与全 logits 拷进同一块
  staging，**1 个 submit、1 次 map_sync、1 次读回**，每行评分从
  2 sync 降为 1（crops 实测：初雪 26.4→21.1 ms，乌戈里尼
  50.8→49.5 ms）。DP/lexicon/NCC 仲裁留 CPU（见边界说明）。
  测试：`tests/test_goal20_crops_gpu.py` 新增 stage 逐位 parity +
  `last_submit_count == 1`；全语料 dict parity 在生产路径上直接
  覆盖 stage。

优先级：G2 ✅ → G3（与 mega 合并发射）→ G4（收口成一次 submit 出结果）。

## 5. 实施顺序与依赖

```text
G1 mega（✅）→ G2 模板 GPU（✅）→ G3 预处理（✅）→ G4 评分链单次发射（✅）→ T2 收尾
```

- G1 mega 已消灭 8-sync 墙；G2 已把端到端大头（模板段）搬上 GPU
  （端到端 3.2-4.7x）；G3 已把 soft 预处理搬上 GPU（byte-exact）；
  G4 已把模板+CNN 评分链收进 1 个 submit（1 次 map_sync）。
- 剩余：T2 读回字节裁剪（[N,C] → Top-K+gather）作为最后的带宽优化；
  视觉 DP/lexicon/CC 按 fonts/goal 边界留 CPU。
- T2（Top-K 读回裁剪）在 G4 之后做读回字节优化；T3 的 profile 公式
  风险被 G3 分解（先 default profile soft 路径）。
- 每项任务落地 = 独立 commit + 把 `tests/test_goal20_wgpu.py` 中对应
  xfail 翻成正式测试 + 更新本文档任务状态。

## 6. 验收标准（Goal 20 完成定义）

- [x] mega 单 dispatch：`classify`/`forward_logits` 各 1 个 dispatch
      （`last_dispatch_count == 1`），28 个 parity 测试全绿
- [x] G2 模板单 dispatch：`tests/test_goal20_template_gpu.py` 10 个
      全绿（逐字段精确 parity），端到端 GPU/auto 领先 3.2-4.7x
- [x] G3 预处理单 dispatch：`tests/test_goal20_crops_gpu.py` 4 个
      全绿（byte-exact + fused logits + crops dict 模式全语料 parity）
- [x] G4 评分链单次发射：`WGPUScoringStage.score_line` 1 submit /
      1 map_sync，stage 逐位 parity + `last_submit_count == 1`
      （`tests/test_goal20_crops_gpu.py`，共 6 个全绿）
- [ ] `tests/test_goal20_wgpu.py` 全绿（无 xfail）
- [ ] `tests/test_wgpu.py` + `tests/test_wgpu_vectors.py` +
      `tests/test_auto_backend.py` 全绿
- [ ] 全量回归（631 tests + 235-crop corpus + game_samples）在
      `backend="cpu"` 下与当前 HEAD CPU 输出一致
- [ ] 端到端 parity：digits / CJK / game_cn 上 `backend="wgpu"` ==
      `backend="cpu"`（text、char_ids、scores < 1e-4）
- [ ] 一次 submit 出 OCR 结果（G2+G3+G4 同 encoder），读回仅最终记录
- [ ] 基准更新并写回 `docs/wgpu.md`：dispatch 数、phase 耗时、
      crossover、连续帧 readback 摊薄
- [ ] README / `docs/architecture.md` 状态更新（Goal 20 complete）

## 7. 环境与基线（2026-08-16 复核）

| 项 | 值 |
| --- | --- |
| Python / wgpu | 3.13.7 / 0.32.0 |
| Adapter | Metal, Apple M4 (IntegratedGPU) |
| 基准 HEAD | `2f46da72d51834d5b4f6f51fa47b574c7fb4c86d`（当前仓库 CPU 实现） |
| 基线测试 | test_wgpu + test_wgpu_vectors + test_auto_backend = 28 passed, 2.60 s |
| 契约骨架 | `tests/test_goal20_wgpu.py`（3 个 now-pass + 5 个 xfail 待实现） |

### 实测（2026-08-16，game_cn 1894 类，median；mega 单 dispatch 后）

`classify()`（12 B/glyph 读回，1 个 dispatch）：

| batch | CPU | GPU | speedup |
| --- | --- | --- | --- |
| 8 | 296 µs | 1.50 ms | 0.2x |
| 32 | 713 µs | 1.47 ms | 0.5x |
| 128 | 2.52 ms | 1.45 ms | 1.7x |

`forward_logits()`（DP 生产路径，mega 单 dispatch，全 `[N,C]` 读回、
单次 sync；此前为 8 次 per-layer sync）：

| batch | CPU | GPU（旧 8-sync） | GPU（mega） | speedup（mega） |
| --- | --- | --- | --- | --- |
| 1 | 171 µs | 12.40 ms | 1.42 ms | 0.12x |
| 8 | 448 µs | 16.56 ms | 1.42 ms | 0.32x |
| 32 | 1.67 ms | 17.85 ms | 1.41 ms | 1.18x |
| 128 | 2.91 ms | 14.24 ms | 2.74 ms | 1.06x |

`classify()`（mega 单 dispatch，12 B/glyph 读回）：

| batch | CPU | GPU | speedup |
| --- | --- | --- | --- |
| 8 | 314 µs | 1.41 ms | 0.2x |
| 32 | 675 µs | 1.37 ms | 0.5x |
| 128 | 2.61 ms | 2.71 ms | 1.0x |

端到端 `recognize()`（渲染文本，G2 后 GPU/auto 首次真正领先；parity 全部
text 一致）：

| 文本 | CPU | GPU (wgpu) | auto | speedup |
| --- | --- | --- | --- | --- |
| 获得金币1000 | 58.3 ms | 18.2 ms | 17.1 ms | 3.2x / 3.4x |
| 巴尔的摩 | 40.1 ms | 15.8 ms | 16.9 ms | 2.5x / 2.4x |
| 获得金币1000×3（21 字） | 157.9 ms | 33.7 ms | 33.5 ms | 4.7x / 4.7x |

auto 选路：模板 crossover = (4, "wgpu")，CNN crossover = (64, "wgpu")。

### crops_items dict 模式实测（2026-08-16，+ real-glyph bank，三后端
text/matched_term 全部一致）

| crop | CPU | GPU (wgpu) | auto | speedup |
| --- | --- | --- | --- | --- |
| Z17 | 20.7 ms | 19.8 ms | 19.9 ms | 1.05x / 1.04x |
| Z28 | 18.7 ms | 17.9 ms | 18.6 ms | 1.04x / 1.00x |
| 初雪（小 crop） | 12.3 ms | 26.4 ms | 12.3 ms | 0.46x / 1.00x |
| 乌戈里尼（大 crop，~94 词条） | 92.0 ms | 50.8 ms | 49.4 ms | 1.81x / **1.86x** |

auto 后端在所有 crops 上无回归（小 crop 自动留在 CPU，大 crop 走 GPU），
全语料 238 个标注 crop 的 dict 模式 parity 由
`tests/test_goal20_crops_gpu.py::test_crops_gpu_dict_mode_parity_full_corpus`
逐张断言。

三个事实（对排期的影响）：

1. G1 mega 消灭了 forward_logits 的 8-sync 墙；G2 把端到端大头
   （模板段）搬上 GPU —— 端到端从打平变为 **GPU 领先 3.2-4.7x**。
2. 剩余 GPU 地板仍是单次 `map_sync`（~1.3-2.7 ms）：batch < 4 的
   模板匹配与 batch < 64 的 CNN 由 auto 选路留在 CPU，无回归。
3. G3（预处理）+ G4（视觉 DP 解码）落地后收口成「一次 submit 出
   OCR 结果」；T2/T4 的读回裁剪与 staged 双缓冲作为后续优化。
