# 项目状态

> 更新日期：2026 年 8 月 19 日

## 当前状态

- 正式数据：`ancientdoc_layout_260707_group_isolated_seed20260815`，train/validation/test=`1548/516/516`，五类跨 split 泄漏均为 0。
- 正式流程：C1/C4 训练、C4 validation-only 分支点选择、同起点 C5/C6 replay、C1/C5/C6 validation 选优和一次 frozen test 均已完成。
- C4 分支点：`checkpoint-6000`，validation 页面 CER `0.567493`；C5/C6 的起点路径、step、权重哈希一致，optimizer 和 scheduler 均为 fresh。
- Frozen test：`ancientdoc_12k_frozen_test_seed20260815_gpu0_retry1`，每项 516 页。C0/C1/C4/C5/C6 页面 CER 为 `1.328217/0.485815/0.457264/0.546852/0.515768`，去空白 CER 为 `1.113916/0.459547/0.438587/0.490939/0.464478`，完整页面 exact match 均为 `0/516`。
- 当前模型排序：C4、C1、C6、C5、C0；当前 AncientDoc 主 checkpoint 为 C4 `checkpoint-6000`。
- 离线验证：已对 C4-C1、C5-C4、C6-C4、C6-C5 执行逐页分析和 27 个书籍组、10,000 次 paired cluster bootstrap。只有 C5-C4 的 95% interval `[0.009897, 0.185221]` 排除 0；其余三项均跨 0。完整报告见 `docs/ANCIENTDOC_GROUP_ISOLATED_ANALYSIS_20260816.md`。
- GOT2 结构消融：A1–A5 已在 `formal_pdf_short_seed20260812` 完成 validation-only 选点和一次 Synthetic-ID frozen test。A1/A2/A3/A4/A5 test page CER 分别为 `0.086320/0.158236/0.069977/0.235795/0.070531`；A4/A5 complete layout F1 为 `0.609785/0.695752`。A3 OCR 最低；A5 OCR 与 A3 接近且布局更完整，但 A5 额外包含 P1 4000 steps，不能视为同总预算。完整记录见 `docs/GOT2_LAYOUT_ABLATION_RESULTS_20260817.md`。
- 多样化 synthetic→AncientDoc 新协议：已实现 ancient-photo-diverse preset、跨 split 生成/审计入口、synthetic A5 P1/P2 validation-only 选点、selection-locked C4 初始化，以及 C0/C1/C4/C5/C6 分阶段 selection/test。新 C5/C6 默认 AncientDoc:synthetic=`7:1`（12.5% replay），旧 frozen run 的 `3:1`（25%）结果不改写。代码已同步 A100；`layout_ablation_smoke_20260817_114133` 在 GPU 0 通过 1-step P1/P2、checkpoint 与 7:1 schedule smoke。正式多样化数据、长程训练和新 AncientDoc test 尚未运行。
- MTHv2 数据转换：已将官方 `label_textline` 转为有序区域标注，并把超过 `max_regions=16` 的页面按连续阅读顺序转换为 oracle chunks。当前数据为 train 2159 源页/5589 chunks、validation 240 源页/593 chunks、test 800 源页/1968 chunks；该输入协议不是 whole-page 端到端列发现，不能与 whole-page 指标直接比较。
- MTHv2 当前运行：A100 正在对 C1–C5 执行 chunk-level validation checkpoint 选择和 selection-locked test，run prefix 为 `mthv2_chunk_ablation_20260819_multi`。截至 2026-08-19 本次检查，五组日志均已记录 `validation_started`，但 `selection.json` 和 test `summary.json` 均未生成，尚无性能结果。
- MTHv2 报告边界：当前 evaluator 产出 chunk-level `layout_validation_metrics.json` 和 predictions；正式报告还需按 `source_page_id`、`chunk_index` 合并为 240 个 validation 源页和 800 个 test 源页。现有脚本明确标记 `grouped_source_page_evaluation=pending`，因此当前 chunk 结果只能作为中间诊断。

## 已完成的正式流程

1. 对全部 C4 周期 checkpoints 运行 validation-only selection，选择 `checkpoint-6000`。
2. 从该 C4-best 独立训练 C5/C6，并完成分支 provenance 检查。
3. C1/C5/C6 按 validation 选 best，C4 固定为分支点 C4-best。
4. 冻结五个模型后，在同一 516 页 test 上统一评估一次。

代码已提供 `train-core`、`select-c4`、`train-replay` 三阶段入口。旧 `C4-final→C5/C6` 流程已废弃。正式命令与紧凑回传见 `docs/SYNC_AND_RUN.md`。

## 归因边界

C5/C6 共享 C4-best 后，C6-C5 的页面 CER 为 `-0.031084`，说明 synthetic layout supervision 相对纯 OCR replay 减轻退化；但 C5-C4 和 C6-C4 分别为 `+0.089588` 和 `+0.058504`，两种 replay 均未超过 C4。C1 与 C4 的结构、可训练参数和上游训练历史仍不同；普通 GOT2 的同上游历史、同数据和近似参数预算 C2 尚未实现，因此不能把 C1-C4 差异完全归因于 VLQA。当前 frozen test 不再用于 replay 配置选择，后续调整只允许使用 validation。
