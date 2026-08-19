# MTHv2 Chunk P1/P2 训练协议（2026-08-19）

## 1. 适用范围

`mthv2_layout_column_chunks16_v1` 可以直接用于 P1/P2，但它代表的是一个新的 **oracle-chunk** 协议：训练和验证输入是依据 MTHv2 textline 标注预先裁剪的页面块，每个块最多包含 16 个有序区域。该协议能解决原始整页页面超过 16 个 layout queries 的容量问题，但不再测量模型从整页图像中自行发现列位置的能力。

因此，chunk 结果必须与 whole-page 结果分开命名和报告，不能把 chunk 识别结果直接称为整页端到端列分割结果。MTHv2 的官方 `label_textline` 仍解释为“有序列候选区域”，不改称为未经校准的严格 column 真值。

当前服务器数据规模：

| split | source pages | chunks |
|---|---:|---:|
| train | 2159 | 5589 |
| validation | 240 | 593 |
| test | 800 | 1968 |

所有 chunk 继承源页面的 split，未跨 split 复制同一源页面。

## 2. 训练前必须完成的两项工程

### 2.1 源页面均衡采样

不能直接把 8150 个 chunk 当作普通独立页面随机采样。一个源页面最多有 26 个 chunk，直接按 chunk 采样会使密集页面获得约 26 倍的训练权重。

正式 loader 应使用以下任一等价实现：

1. 每个 optimizer step 先均匀采样 `source_page_id`，再在该页面的 chunks 中均匀采样一个 `chunk_index`；或
2. 对每个 chunk 使用权重 `1 / chunk_count` 的 weighted sampler。

训练日志必须记录 source-page exposure，而不是只记录 chunk exposure。若暂时不能改 sampler，当前 chunk 数据只能用于短 smoke，不能作为正式性能结论。

### 2.2 按源页面合并验证和测试

现有 evaluator 若把每个 chunk 当作独立 page，会让 chunk 多的页面重复计权。正式评估应：

1. 按 `source_page_id` 分组；
2. 按 `chunk_index` 排序；
3. 拼接各 chunk 的预测文本，再与原始整页 `page_text` 比较；
4. 页面 CER、编辑距离和 exact match 按源页面做 macro average；
5. chunk-level CER 只作为诊断指标；
6. 将 `source_crop_box_px` 加回局部 bbox，计算源页面坐标下的布局覆盖指标。

验证选择和 frozen test 都必须使用源页面级指标。当前 validation/test 分别对应 240/800 个源页面，而不是 593/1968 个独立页面。

## 3. 推荐主训练方法

### P1：布局预热

- 输入：chunk 图像和 OCR prompt；
- trainable：VLQA layout queries、query cross-attention、辅助 object/bbox/direction heads；
- frozen：视觉塔、Qwen、projector、zero-gated write-back；
- 损失：
  - object BCE = 1.0；
  - bbox L1 = 5.0；
  - bbox GIoU = 2.0；
  - direction CE = 1.0；
  - OCR loss = 0；
- steps：`12,000`；
- checkpoints：每 `3,000` steps；
- selection：validation layout F1 主指标，matched bbox IoU 次指标，direction accuracy 再次，较早 step 作为 tie-break；不能用 P1 OCR CER 选 checkpoint。

P1 的目标是让 query 在已裁剪的列候选块内学习区域存在性、局部 bbox 和书写方向。它不能证明模型已经学会整页列发现。

### P2：联合 OCR/布局训练

- 初始化：加载已通过 validation 选择和 checkpoint verification 的 P1 checkpoint；
- trainable：VLQA、projector、zero-gated write-back；
- 默认不解冻 Qwen decoder，`p2_train_scope=adapter_projector`；
- 损失：OCR CE = 1.0，布局使用 `layout_full`（1/5/2/1/1）；
- steps：`30,000`；
- checkpoints：每 `3,000` steps；
- selection：源页面 validation CER 主指标，去空白 CER 和 exact match 作为 tie-break；布局 F1、源页面 bbox IoU 和顺序指标作为并列诊断，不替代 OCR 主指标。

P2 的总预算为 30,000 steps，重点观察局部列块 OCR 是否受布局监督改善。只有在 adapter/projector P2 证明稳定后，才增加一个单独的 `decoder_adapter_projector` 扩展，不把 decoder 解冻混入主实验。

### 预算公平规则

P1→P2 主配置总共执行 `12,000 + 30,000 = 42,000` 个 optimizer steps。所有直接 P2 对照必须执行 `42,000` steps，而不是只执行 30,000 steps，否则 P1 预热收益与额外训练预算混在一起。

每个 run 固定：seed、effective batch size、gradient accumulation、图像处理、prompt、tokenizer、学习率策略、weight decay、解码参数和 source-page sampler。每个 checkpoint 间隔统一为 3,000 steps。

## 4. 对照组

### 主对照矩阵

| ID | 输入 | 训练 | 布局监督 | 目的 |
|---|---|---|---|---|
| `C0-page` | 原始整页 | 不训练的原始 GOT2 | 无 | whole-page 零样本参考；不与 chunk 指标直接混比 |
| `C0-chunk` | oracle chunks | 不训练的原始 GOT2 | 无 | 测量“已知裁剪输入”本身带来的收益 |
| `C1` | oracle chunks | projector-only，42k | 无 | 控制域适配和 projector 更新 |
| `C2` | oracle chunks | generic adapter + projector，42k | 无 | 控制新增参数量和普通视觉 adaptor |
| `C3` | oracle chunks | VLQA direct P2，42k | 无 | 控制 query/cross-attention/write-back 结构，但不使用布局损失 |
| `C4` | oracle chunks | VLQA direct P2，42k | object+bbox+direction | 测量直接布局监督收益 |
| `C5` | oracle chunks | VLQA P1 12k → P2 30k | object+bbox+direction | 测量 P1 预热相对 C4 的收益 |

其中：

- `C0-page` 只作为 whole-page 参考，不能和 `C0-chunk` 的 CER 直接做结论性差值；
- `C1`、`C2`、`C3`、`C4`、`C5` 必须使用完全相同的 chunk manifest 和源页面均衡采样；
- `C4` 与 `C3` 的差异归因于布局监督；
- `C5` 与 `C4` 的差异归因于 P1 预热，但前提是两者总 optimizer steps 都为 42k；
- `C2` 与 `C3` 的差异归因于 VLQA 结构，而不是单纯增加参数量。

### 可选扩展对照

以下实验不进入首轮主矩阵：

- `C6-decoder`：在 C5 P2 阶段解冻 decoder adaptor/projector，检查小样本过拟合和 OCR 收益；
- `C7-layout-loss-ablation`：分别使用 `object_only`、`object_bbox`、`object_direction_order`，拆分布局监督来源；
- `C8-page-under16`：只保留原始整页中区域数不超过 16 的页面，检验 chunk 裁剪是否改变任务难度；该集合不能代表完整 MTHv2。

外部列检测器、Mask R-CNN、双 GOT2 以及使用预测 mask 的在线裁剪属于独立两阶段路线，不并入 `C0`–`C8`。

## 5. 结果报告

每个 run 至少报告两套结果：

1. **源页面级主指标**：page CER、page edit distance、page exact match、源页面合并后的区域 recall/F1、源坐标 bbox IoU、reading-order pair accuracy、Kendall's tau；
2. **chunk 级诊断**：chunk CER、chunk exact match、每个 chunk 的区域 F1 和 bbox IoU。

另外报告：source-page 数、chunk 数、每源页面 chunk 数分布、source-page exposure、可训练参数量、峰值显存、吞吐量和单 chunk 延迟。

如果 C5 只改善 chunk CER 而不改善源页面 CER，不能宣称布局监督改善整页识别；如果 C5 只改善布局 F1 而 OCR 不改善，应保留为布局诊断收益，不扩大训练复杂度。

## 6. 首轮执行顺序

1. 实现 source-page balanced sampler；
2. 实现 grouped validation/test evaluator；
3. 运行 `C0-chunk` 诊断和 `C1` projector-only；
4. 并行或顺序运行 `C2`、`C3`、`C4`，均 42k steps；
5. 运行 `C5`：P1 12k，验证选点后 P2 30k；
6. 只在 C3/C4/C5 的源页面 validation 完成 checkpoint selection 后运行一次 frozen test；
7. 再决定是否做 `C6-decoder` 或回到 whole-page query capacity 扩展。

本协议不启动训练本身；它只定义 MTHv2 chunk 数据的正式训练和对照边界。
