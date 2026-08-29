# PVLD 主线配置登记

更新日期：2026-08-29

本文件登记共享 `main` 唯一活动的 whole-page GOT2/PVLD 流程。历史 pilot、M2/M3/M4/All、VLQA/VQLCA、Chunk、AncientDoc、BSCC 和 SOTA 配置位于 `archive/legacy-vlqa-chunk-20260829`。

## 输入与模型

| 项目 | 当前值 |
|---|---|
| OCR 输入 | GOT2 原生 whole-page image + OCR prompt |
| 布局输入 | 从整页视觉特征生成 global layout prompts；无外部 bbox 输入 |
| 布局 decoder | causal PVLD，变量长度 REGION/EOS 序列 |
| OCR 路由 | `visual_value_layout_routing`，Value 来自视觉 token |
| 辅助监督 | bbox、writing direction、reading order、count |
| 工程上限 | `max_regions=512`、`max_layout_tokens=2048`；不表示 Fixed-Slot query 数 |
| 随机种子 | `42`，如变更须写入 run metadata |

## 阶段与选点

| 阶段 | 数据和目的 | checkpoint 规则 |
|---|---|---|
| P1 | S3/S4 layout training，可使用 MTHv2 train replay | 只按锁定 validation 的布局指标选择 |
| P2 | 从 selected P1 启动 OCR＋layout 训练 | 只按锁定 validation 选择 |
| P3 | 从 selected P2 启动 MTHv2 train 域适配 | 只按锁定 validation 选择 |
| Test | selected P3 的 Real-OOD 评测 | selection 锁定后执行一次，不反向调参 |

当前编排器为 `tools/training/run_time_constrained_pvld_baseline.sh`，阶段 runner 为 `tools/training/run_variable_layout_a100.py`。P1/P2/P3 共用 S3/S4 400 页 validation lock；P3 test 使用 MTHv2 官方 test manifest。

## 数据锁

- S3/S4 validation：每个 tier 200 页，共 400 页；按 region-count bucket 与 complexity tertile 分层。
- MTHv2：train 用于 replay/P3，validation 不训练，test 仅最终 locked test。
- 所有 run 记录 manifest SHA-256、selection 文件、source checkpoint 和 checkpoint hash。
- `label_textline` 只称为有序 textline/region candidate，不称为严格 column 标注。

## GPU 与结果

启动前只检查本次命令允许范围内 GPU 的瞬时 utilization；严格低于 50% 才使用，忙卡不等待、不抢占。结果至少报告 CER、页面精确匹配、区域 F1、bbox IoU、方向/顺序、EOS/count 截断、稳定性、参数量、显存和吞吐。当前没有新的正式性能结论。
