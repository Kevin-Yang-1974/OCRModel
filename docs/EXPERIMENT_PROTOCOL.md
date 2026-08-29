# 当前实验协议

## 研究对象与输入

当前研究对象是面向多场景、小样本条件的通用符号识别模型。正式样本单位为完整页面，模型输入固定为 `whole_page_image + ocr_prompt`。布局区域、bbox、书写方向和阅读顺序只作为训练期辅助监督或离线评测真值，不作为推理输入；line-level 或 oracle chunk 只属于独立诊断，不能与页面结果混比。

## 阶段流程

```text
P1 layout training
 -> validation-only P1 selection
P2 OCR + layout training (from selected P1)
 -> validation-only P2 selection
P3 domain adaptation (from selected P2)
 -> validation-only P3 selection
 -> selection-locked test
```

每个阶段的选择只读取 validation。test 不得参与训练、checkpoint 选择、阈值、prompt 或后处理调整；每次选择必须记录 checkpoint、数据清单和协议哈希。

## 数据划分与少样本

划分先于页面渲染和增强，按书手、版本、馆藏、来源文档、内容 ID 或符号类型等来源组进行。禁止同一源页、同一内容、近重复页面或 crop 跨 split。领域级少样本限制独立来源组的标注页数/比例；稀有符号级 K-shot 限制每个符号的独立来源实例数，增强版本不增加 K。报告必须给出 split、来源组、页面数、区域数、manifest SHA-256 和近重复审计状态。

S3/S4 正式 validation 固定 400 页（每个 tier 200 页），P1/P2/P3 共用。MTHv2 `train` 仅用于 P1 replay/P3 训练，`validation` 不进入训练，`test` 仅用于最终 locked test；其 `label_textline` 是有序候选，不宣称严格列真值。详见 [`BRANCH_AND_DATA_LAYOUT.md`](BRANCH_AND_DATA_LAYOUT.md)。

## 目标函数与指标

P1 训练 causal PVLD 的布局 token、REGION/EOS、bbox、方向和 count 监督；P2/P3 联合 OCR 与布局损失，OCR Value 通过 `visual_value_layout_routing` 保持来自视觉 token。报告页面 CER、编辑距离、页面精确匹配、区域 P/R/F1、bbox IoU、方向/顺序指标、EOS/count 截断、训练稳定性、可训练参数量、峰值显存和吞吐。

## 历史方案边界

Fixed-Slot VLQA/VQLCA、Chunk、AncientDoc、BSCC、SOTA、M2/M3/M4/All 和双 GOT2 均不属于当前协议。其脚本、报告和测试保存在 `archive/legacy-vlqa-chunk-20260829`，仅用于复现和溯源。不同输入粒度、split、预算或 checkpoint 选择规则的结果不得直接比较，也不得把历史数值写成 PVLD 主线结论。
