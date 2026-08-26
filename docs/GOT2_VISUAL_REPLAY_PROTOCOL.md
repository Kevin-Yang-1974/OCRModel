# GOT2 Visual Replay and High-Resolution Layout Protocol

更新日期：2026-08-26

本文是新视觉解冻实验的执行协议。它只定义代码、数据和验证契约；在完成本地测试、数据生成、manifest 审计并得到确认前，不启动正式训练、validation、test 或远端长程作业。

## 1. 输入与分支边界

- 正式输入仍是 GOT2 原生 whole-page image + OCR prompt。
- Vary ViT 最终 `[B,1024,16,16]` 特征经过 `mm_projector_vary` 后形成固定的 256 个 Qwen OCR visual tokens。
- `layout_memory_resolution=16` 使用 16×16 memory；`layout_memory_resolution=64` 使用 neck 后的 `[B,256,64,64]` memory，并经过 layout decoder projection。
- 64×64 memory 只进入 layout queries、PVLD decoder、spatial coverage 和布局辅助头，不拼接到 Qwen。
- bbox、direction、reading order 只用于训练监督、离线解释和评测真值，不是正式推理输入。

## 2. 参数组

| 参数组 | P1 | P2 | P3 |
|---|---:|---:|---:|
| Vary ViT | 1e-6 | 5e-7 | 2e-7 |
| mm projector | 1e-5 | 5e-6 | 2e-6 |
| layout queries/decoder/heads | 1e-4 | 5e-5 | 1e-5 |
| Qwen decoder | frozen | 25% steps 后解冻，1e-6 | 5e-7 |
| residual gate | frozen zero | 1e-5 | 1e-6 |
| OCR lm head | frozen，除非 GOT2 tied-head 路径要求同步更新 | 同左 | 同左 |

所有组使用相同 optimizer、batch、seed、checkpoint 间隔和 optimizer steps。参数组名称写入 `layout_training_metrics.json`。

## 3. 损失与 replay

P1 采用：

```text
L_P1 = L_layout(primary + replay) + 0.25 * L_ocr(replay)
```

主页面只提供 layout supervision；replay 页面经过原始 whole-page OCR 路径，OCR labels 不得被置为 `IGNORE_INDEX`。固定 `primary:replay=7:1`，`replay_ocr_loss_weight=0.25`。

P2/P3 采用：

```text
L_total = L_ocr + 1.0 * L_layout
```

`L_layout` 包含 layout token、boundary、bbox、direction、count；coverage 只在 M2 启用。未经独立验证的 duplicate loss 不进入主协议。

训练日志必须包含：OCR replay token 数、OCR replay loss、vision/projector gradient norm、参数 update norm、feature drift、residual gate、页面 OCR 指标和布局指标。

## 4. S0-S4 合成数据

| Tier | 目标 |
|---|---|
| S0-layout | 规则字体和基础页面，用于 layout decoder 预热 |
| S1-real-crop | 真实单行/单列/区域 crop 嵌入整页 |
| S2-dense | 高列数、不规则列宽、局部压缩和密集区域 |
| S3-ancient-hard | 纸张纹理、透印、污渍、折痕、缺损、扫描阴影、模糊、局部遮挡 |
| S4-mixed | 多方向、混合阅读顺序、错位、倾斜、局部重叠和复杂边界 |

首版区域数目标分布：1–8 为 10%，9–16 为 15%，17–32 为 25%，33–64 为 30%，65–128 为 15%，>128 为 5%。`>32` 页面至少占 50%。

每条记录保存 image、HTML、page transcription、region bbox/order/direction、content/source hash、font/version、browser version/hash、degradation parameters、column count、region count 和 difficulty tier。

审计包括 source/content split isolation、重复与近重复页面、列数/区域数、bbox 面积与 aspect、direction、字体、退化强度、文本长度和 `>32` 页面比例。

### 数据规模硬门槛

由于 P1/P2/P3 会解冻视觉塔、projector、layout branch 和部分 Qwen，合成数据不能继续停留在 3,600 页或数千页规模。新主线的最低目标是：train 至少 10,000 张唯一渲染页面，且 train manifest 审计后的 region exposure 至少 1,000,000；默认五个 tier 使用 20,000/2,000/2,000 的 train/validation/test 页面配额，即 100,000/10,000/10,000 张页面。后续可扩展到百万级页面或更高 region exposure，但必须保持 source/content split 隔离。

这里的三个数量必须分开记录：

- unique content/source instances：独立内容和来源实例，决定小样本与泄漏协议；
- rendered pages：不同合法布局、字体和退化组合的页面数量；
- training exposures：optimizer steps × effective batch × replay schedule 的训练曝光次数。

同一 content_id 的无限布局复制只能增加 rendered page/exposure，不能增加独立 K-shot 内容实例。

## 5. P2 消融矩阵

| 组 | spatial memory/coverage | backward scale | predicted-layout routing |
|---|---|---|---|
| Legacy | off | 1.0 | off |
| M2 | on | 1.0 | off |
| M3 | off | 0.25 | off |
| M4 | off | 0.25 | on |
| All | on | 0.25 | on |

M4 只允许自由生成的 predicted layout evidence。validation 必须同时运行 normal、`alpha=0` 和 shuffled evidence；normal 未在 validation 上稳定优于两者时，不形成 M4 OCR 结论。

## 6. Selection 与 test

```text
checkpoint -> validation-only generation/ranking
           -> selected checkpoint
           -> next stage initialization
           -> selection-locked test
```

P1 选择布局质量，P2/P3 以页面 CER 为主并设置 layout non-collapse 约束。test 只能读取 validation 产生的 `selection.json`，不得参与训练、选点、阈值、prompt、后处理或结构判断。

## 7. 资源与 smoke 要求

64×64 memory 长度为 4096，是 16×16 的 16 倍。若 decoder cross-attention 显存过高，依次使用 memory projection、chunked attention 和 activation checkpointing；不得把高分辨率 memory 整体送入 Qwen。

必须完成以下 bounded smoke：

1. 16×16/64×64 forward 与 backward；
2. vision/projector/layout/Qwen 梯度非零性；
3. checkpoint 保存、重载和缺失 64×64 键的 deterministic initialization；
4. DDP 参数同步；
5. validation evaluator 与 padding mask；
6. memory/throughput 对比；
7. S3/S4 数据生成、manifest 审计和近重复检查。

未完成上述检查前，不得启动正式训练或任何新的远端长程作业。

## 8. GPU 准入与旧 checkpoint 清理

`run_variable_layout_a100.py` 默认自动查询启动瞬间所有物理 GPU，并将全部
`utilization.gpu < 50%` 的卡作为同一个多卡作业；达到或超过阈值的卡不等待、不抢占。
传入 `--gpu-ids` 时只查询和使用显式列表中的卡。无合格卡或查询失败会在启动子进程前退出，
并把准入模式和观测值写入 run metadata。

合成数据生成前如确有磁盘压力，只能对明确指定的历史 run 使用
`tools/maintenance/prune_old_checkpoints.py`。该工具默认 dry-run，保护运行中 run、
validation `selection.json` 引用的 checkpoint、final/source model 以及最近的 checkpoint；
只有显式 `--apply` 才会删除候选目录，不触碰未指定的 run、日志或 test 结果。
