# 当前项目状态

更新日期：2026-08-29

## 已实现

- GOT2 whole-page 输入与 causal PVLD 布局 decoder 已接入；global layout prompts 从整页视觉特征生成，OCR Value 通过 `visual_value_layout_routing` 保持来自视觉 token。
- P1/P2/P3 阶段 runner、validation-only checkpoint selection、selection-locked test、400 页 validation lock 和 MTHv2 whole-page 转换工具已在本地实现。
- A100 forward/backward、checkpoint 重载、GPU 瞬时 utilization 准入和有限 smoke 链路已验证。
- S3/S4 合成页面生成、跨 split/近重复审计以及 MTHv2 `train/validation/test` manifest 约束已实现。

## 正在实施或待验证

- 正式 whole-page P1 → P2 → P3 长程训练、held-out validation、跨来源小样本评估和 selection-locked test 尚未形成新的性能结论。
- 需要在统一协议下比较 PVLD 与原始 GOT2、等参数普通 adaptor 及固定槽位兼容 baseline；在消融完成前不宣称结构收益。
- MTHv2 的 `label_textline` 仅表示有序区域候选，不提供严格 column ground truth；跨书手/版本/馆藏泛化需在有来源分组的数据上另行验证。

## 报告边界

历史 smoke、pilot、AncientDoc、BSCC、SOTA、Chunk 和 Fixed-Slot VLQA/VQLCA 结果只用于诊断或溯源，不能写入当前 PVLD 主线结论。不同输入粒度、split、训练预算或 checkpoint 选择规则的数字不得直接比较。

分支映射、数据划分和 MTHv2 用途见 [`BRANCH_AND_DATA_LAYOUT.md`](BRANCH_AND_DATA_LAYOUT.md)。发布目标是共享远程 `main`，旧方案目标是 `archive/legacy-vlqa-chunk-20260829`；运行命令见 [`SYNC_AND_RUN.md`](SYNC_AND_RUN.md)。个人发布命令保存在被忽略的本地 `docs/PUBLISHING.md`。
