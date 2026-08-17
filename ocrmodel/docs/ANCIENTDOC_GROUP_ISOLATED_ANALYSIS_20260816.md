# AncientDoc 分组隔离 frozen test 离线验证报告

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-16
- Verification Status: ANALYZED
- Version Label: ancientdoc_group_isolated_validation_v1

## 验证对象

- Frozen test run：`ancientdoc_12k_frozen_test_seed20260815_gpu0_retry1`
- 数据集：`ancientdoc_layout_260707_group_isolated_seed20260815`
- 页面数：516
- 来源组数：27
- 分析输入：既有 `layout_validation_predictions.jsonl` 与 test manifest
- 分析性质：CPU-only 离线分析；未重新推理、未训练、未占用 GPU
- 统计单位：以 `source_group_id` 为 cluster 的 paired nonparametric bootstrap，10,000 次，seed=`20260814`

页面 CER 由全部页面的字符编辑总数除以参考字符总数得到。差值统一定义为 right minus left；负值表示 right 更好。page-level bootstrap 仅作为同一 split 内诊断，主要不确定性判断采用 source-group cluster bootstrap，因为同一本书内页面并非独立样本。

## 配对结果

| 比较 | CER 差值 | 相对变化 | 编辑数差值 | right 改善/退化/持平页 | 书籍组 bootstrap 95% CI | 判断 |
|---|---:|---:|---:|---:|---:|---|
| C4-C1 | -0.028551 | -5.88% | -4418 | 244/253/19 | [-0.098053, 0.037284] | 区间跨 0，C4 总体点估计更好但来源级稳定性不足 |
| C5-C4 | +0.089588 | +19.59% | +13863 | 242/253/21 | [0.009897, 0.185221] | 区间完全大于 0，纯 OCR replay 退化在当前书籍集合中较稳定 |
| C6-C4 | +0.058504 | +12.79% | +9053 | 242/248/26 | [-0.013050, 0.140378] | 点估计退化，但来源级区间跨 0 |
| C6-C5 | -0.031084 | -5.68% | -4810 | 250/234/32 | [-0.088948, 0.016702] | 布局监督有减轻 replay 退化的趋势，但区间跨 0 |

四项 page-level bootstrap 95% CI 依次为 `[-0.081037, 0.024813]`、`[0.022002, 0.160136]`、`[0.005026, 0.112932]`、`[-0.095497, 0.029129]`。其中 C6-C4 在 page-level 区间下看似稳定退化，但按书籍整组重采样后区间跨 0，表明把 516 页当作独立样本会低估来源异质性。

## 来源异质性

- C4-C1：`選擇天鏡` 的差值为 `-0.482735`，而 `河嶽英靈集` 为 `+0.380452`。C4 的总体收益并非多数页面一致改善，而是不同书籍方向和幅度高度不均。
- C5-C4：`選擇天鏡` 退化 `+0.617436`，`黄獻臣-武經開宗` 退化 `+0.557167`；但 `河嶽英靈集` 改善 `-0.257137`。纯 OCR replay 的总体退化仍包含明显来源差异。
- C6-C4：`黄獻臣-武經開宗` 退化 `+0.454932`，`昌黎先生集` 退化 `+0.329470`，兵家类合计退化 `+0.243150`；`河嶽英靈集` 改善 `-0.275971`，总集类合计改善 `-0.172129`。
- C6-C5：`選擇天鏡` 改善 `-0.496752`，但 `武經七書講義全彙合参` 退化 `+0.127085`，`书谱` 退化 `+0.110778`。布局监督对 replay 的影响同样依赖来源。

这些分组结果只用于定位后续 validation 分析优先级，不用于在已冻结 test 上选择新配置。

## 错误模式

- C4 相比 C1 将 `CER >= 1` 页面从 44 减至 20、over-generation 从 23 减至 7、长重复字符从 20 减至 13，但 under-generation 从 7 增至 32。总体编辑数下降伴随错误类型转移。
- C5 相比 C4 将 `CER >= 1` 页面从 20 增至 49、over-generation 从 7 增至 28；新出现的 `CER >= 1` 和 over-generation 分类分别为 40 页和 26 页，是纯 OCR replay 退化的主要表征。
- C6 相比 C4 将 `CER >= 1` 页面从 20 增至 41、over-generation 从 7 增至 23；相较 C5，这两类错误总数有所下降，但仍高于 C4。
- 五个 frozen 模型的完整页面 exact match 均为 0/516，因此不能用 exact match 支撑模型间优劣；当前排序主要由聚合编辑距离决定。

## 统计警告

1. 四项配对区间属于同一组探索性比较，尚未进行多重比较校正。只有 C5-C4 的未校正 95% cluster interval 排除 0，不应把其余差值写成已证实改善或退化。
2. 只有 27 个来源组；cluster interval 明显宽于 page interval，来源数而非页面数是跨书籍不确定性的主要约束。
3. 当前只有一个训练 seed。bootstrap 衡量当前 test 来源构成的不确定性，不能替代多随机种子训练稳定性。
4. C1 与 C4 的结构、可训练参数和上游训练历史不同；即使 C4 点估计更低，也不能把 C4-C1 解释为 VLQA 的纯因果效应。
5. C5/C6 共享同一 C4 起点和 replay 数据协议，C6-C5 是当前最接近受控的布局监督比较；但其 cluster interval 仍跨 0。

## 统计谬误检查

- 覆盖：11/11。
- Simpson 悖论：`CAUTION`。未观察到所有子组方向一致而总体反转的严格 Simpson 条件，但多个书籍组与总体方向相反，聚合 CER 会掩盖强来源异质性。
- 生态谬误：`NOTE`。报告只对页面和书籍组作推断，不外推到单个字符类别、全部古籍或其他符号场景。
- Berkson/选择偏差：`CAUTION`。27 本测试书籍来自 AncientDoc 的受限来源集合，不能代表所有版本、馆藏或书手。
- Collider bias：`NOTE`。当前没有含控制变量的回归模型，未发现可检验的 collider 路径。
- Base-rate neglect：`NOTE`。主指标为编辑距离，不是敏感度/特异度；稀有符号召回仍未分析。
- Regression to the mean：`NOTE`。checkpoint 由 validation 选择，test 未参与选择；未按 test 极端值选模。
- Survivorship bias：`NOTE`。五个模型均覆盖同一 516 页，没有缺页；生成失败仍计入编辑距离。
- Look-elsewhere effect：`CAUTION`。四项 pair 和多个书籍/类别分组属于探索性分析，未做多重比较校正。
- Garden of forking paths：`CAUTION`。训练包含多 checkpoint 与 replay 选择；当前流程已锁定 validation-only 选优，但仍需预先登记后续 replay 搜索空间。
- Correlation is not causation：`CAUTION`。C4-C1 不是同预算同历史结构对照，不能作纯结构因果归因；C6-C5 虽更受控，但单 seed 且区间跨 0。
- Reverse causality：`NOTE`。模型配置先于冻结评估确定，不存在结果反向决定已完成训练配置；test 结果仍不得反向用于调参。

## 当前结论

- 当前 AncientDoc 主 checkpoint 保持 C4 `checkpoint-6000`，因为它在本次 frozen test 上取得最低聚合页面 CER。
- source-group 统计证据最明确的是：当前 synthetic OCR replay（C5）相对 C4 产生退化。
- C6 相对 C5 的点估计更好，但不能写成稳定的布局监督收益；C6 相对 C4 也不能写成稳定退化，因为 cluster interval 跨 0。
- 后续只在 validation 上分析 replay 比例、书籍组采样和 over-generation 抑制；当前 test 只保留作冻结报告与离线错误定位。

## 服务器产物

四组 v2 分析位于：

`$GOT_EVALUATION_RUNS/ancientdoc_12k_frozen_test_seed20260815_gpu0_retry1/test/analysis/`

- `c1_got2_ocr_only_vs_c4_cluster/`
- `c4_vs_c5_cluster/`
- `c4_vs_c6_cluster/`
- `c5_vs_c6_cluster/`

每个目录包含 `summary.json`、`analysis_summary.md`、`page_comparison.csv`、`group_comparison.csv`、`error_categories.json` 和 `worst_pages.md`。
