# 项目状态

> 更新日期：2026 年 8 月 14 日
> 状态依据：当前代码、受限 A100 运行摘要与已完成的本地诊断

## 当前主线

当前工作聚焦于单个 GOT2 内的整页布局查询与小样本通用符号识别。推理输入只保留原始整页图像和 OCR prompt；bbox、阅读顺序和书写方向只用于训练辅助监督、可选解释输出或离线评测。

AncientDoc 现在作为真实古籍适配与验证数据源使用，不再作为独立主线口径。line-level 结果仅用于诊断和历史兼容。

## 已有入口

- `tools/training/run_layout_a100.py`：A100 训练与诊断总编排。
- `tools/training/run_ancientdoc_baseline_suite.sh`：古籍适配训练总入口。
- `tools/evaluation/run_ancientdoc_validation_suite.sh`：古籍验证总入口。
- `tools/preprocessing/prepare_ancientdoc_layout_dataset.py`：把 AncientDoc GOT 标注转成 `layout-page-jsonl`。

## 当前已能直接跑的 baseline

- `adapt` + AncientDoc OCR-only。
- `adapt` + AncientDoc + synthetic OCR replay。
- `adapt` + AncientDoc + synthetic layout replay。
- `joint-train` 的 loss 消融对照。

## 已完成的工程验证

- A100 训练链路、checkpoint 保存与重载、整页 validation 链路已经打通。
- AncientDoc 原始目录结构已经确认，数据文件位于 `/data4/hyf/backup/GOT-OCR2.0/reference-260707/AncientDoc`。
- `layout_page_dataset.py` 已支持真实 OCR-only 页面与 replay 混合输入。
- AncientDoc 的 C4/C5/C6 适配与统一 516 页测试已经完成，结果见 `docs/ANCIENTDOC_BASELINE_REPORT_20260814.md`。当前最佳为 C6，页面 CER 为 `0.934099`，但完全正确页面仅 `1/516`。

## 仍需正式完成

- 统一整页训练基线的正式 held-out 验证。
- AncientDoc 跨书籍、跨版本、跨馆藏及近重复隔离后的正式小样本验证。
- 结构创新 baseline 的独立实现与公平比较。

## 结果口径

AncientDoc 当前结果可作为页面级真实古籍适配基线：C4 明显优于原始 GOT2，C5 略差于 C4，C6 为当前最优。由于原始 split 尚未完成来源分组与近重复隔离审计，且没有多随机种子或逐页配对显著性检验，该结果不能外推为小样本泛化结论，也不能单独证明布局查询结构有效。
