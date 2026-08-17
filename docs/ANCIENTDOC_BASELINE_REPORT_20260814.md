# AncientDoc 基线验证报告

> 2026-08-16 正式更新：书籍分组隔离的 12000-step 流程已经完成 validation-only 选优和一次 frozen test。本文件后半部分的 2026-08-14 原始 split、2000-step 结果仅作历史兼容记录，不再决定当前模型排序。

> 公平性更正：本报告中的“原始 GOT2”是 `c0_got2_zero_shot` 零样本推理参考，不是与 C4/C5/C6 相同训练预算的 baseline。该报告记录旧 split、2000 steps 的历史结果，不包含新实现的 C1，不能用于 C1/C4 公平比较或正式小样本结论。正式重跑必须使用 group-isolated 数据集，并同时报告 optimizer steps、样本/token 暴露、可训练参数和初始 checkpoint。

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-14
- Verification Status: ANALYZED
- Version Label: ancientdoc_validation_v1

## 2026-08-16 分组隔离 frozen test

- 数据集：`ancientdoc_layout_260707_group_isolated_seed20260815`
- train/validation/test：`1548/516/516`
- 跨 split 审计：五类书籍、来源及近重复泄漏检查均为 0
- Frozen test run：`ancientdoc_12k_frozen_test_seed20260815_gpu0_retry1`
- checkpoint 选择：C1 `12000`、C4 `6000`、C5/C6 `10000`，均只由 validation 决定
- C5/C6 分支：相同 C4 模型路径、step 和权重哈希，fresh optimizer，fresh scheduler，consistency=`consistent`

| 模型 | 页面 CER | 去空白 CER | 平均页面编辑距离 | 完全正确页面 |
|---|---:|---:|---:|---:|
| C0：原始 GOT2 zero-shot reference | 1.328217 | 1.113916 | 398.316 | 0/516 |
| C1：GOT2 OCR-only | 0.485815 | 0.459547 | 145.690 | 0/516 |
| C4：VLQA OCR-only | **0.457264** | **0.438587** | **137.128** | 0/516 |
| C5：C4＋synthetic OCR replay | 0.546852 | 0.490939 | 163.994 | 0/516 |
| C6：C4＋synthetic layout replay | 0.515768 | 0.464478 | 154.672 | 0/516 |

当前排序为 C4、C1、C6、C5、C0。C5-C4 与 C6-C4 的页面 CER 点差分别为 `+0.089588` 和 `+0.058504`，两种 replay 的聚合指标均未超过无 replay 的 C4。C6-C5 点差为 `-0.031084`，只提示“布局监督可能相对纯 OCR replay 减轻退化”；其 source-group cluster interval 跨 0，不能表述为稳定收益。C4-C1 为 `-0.028551`，但两者结构、参数预算和上游训练历史并不完全相同，不能将该差值单独归因于 VLQA。五项 exact match 均为 0，说明整页绝对质量仍不足。

该 test 已冻结并完成一次最终评估，后续不得据此调整 replay 比例、合成数据子集或损失权重。下一步只对既有 predictions 做逐页配对、分组误差和统计区间诊断；任何新配置只能在 validation 上选择，并需在预先锁定的新 test 或新随机种子上复验。

离线配对与 source-group cluster bootstrap 已于 2026-08-16 完成。只有 C5-C4 的 95% cluster interval `[0.009897, 0.185221]` 完全大于 0；C4-C1、C6-C4、C6-C5 的区间均跨 0。正式统计解释、来源异质性和错误模式见 `ANCIENTDOC_GROUP_ISOLATED_ANALYSIS_20260816.md`。

## 验证范围

- 结果目录：`/data3/yky/yangky_ocr_models/evaluation_runs/GOT/ancientdoc_validation_20260814`
- 数据：AncientDoc 转换数据的 `test` split，即原始 `split5`
- 页面数：516
- 输入协议：原始整页图像＋OCR prompt
- 布局信息：AncientDoc 没有布局标注；布局 metadata 不作为模型输入
- 结果来源：用户回传的 `summary.json` 紧凑指标
- 验证状态：只完成单次指标分析，没有独立重复运行、多随机种子区间或逐页配对显著性检验

## C4、C5、C6 实验配置

本报告记录的是首次 AncientDoc 页面级 split 实验：三项适配均为 2000 steps，train/validation/test 沿用原始 split1–3/split4/split5。它不是当前 12000-step、书籍分组隔离重跑的结果；两套协议的指标不得直接合并。当前可执行配置及新数据划分见 `tools/training/README.md`。

三项配置的模型输入均为原始整页图像和 OCR prompt。训练时语言模型与高分辨率视觉塔冻结，只更新完整 VLQA `layout_adapter` 与 `mm_projector_vary`。AncientDoc 没有布局标注，其页面只产生 OCR 监督。

| 编号 | 起始 checkpoint | 数据组成 | 损失设置 | 对照含义 |
|---|---|---|---|---|
| C4：AncientDoc OCR-only | 合成数据 P2 8000 VLQA checkpoint | 仅 AncientDoc | `L_ocr`，布局损失权重为 0 | 真实古籍 OCR-only 域适配基线 |
| C5：C4＋synthetic OCR replay | 首次实验的 C4 checkpoint | AncientDoc＋合成页面，按 3:1 交错 | AncientDoc 和合成页面都只计算 `L_ocr`，布局损失权重为 0 | 检验纯 OCR replay 是否缓解遗忘或改善泛化 |
| C6：C4＋synthetic layout replay | 与 C5 相同的首次实验 C4 checkpoint | AncientDoc＋同源合成页面，按 3:1 交错 | AncientDoc 计算 `L_ocr`；合成页面计算 `L_ocr + L_layout`，布局项含 object、bbox L1、bbox GIoU、direction | 在相同起点、数据源和比例下比较保留布局监督的 replay |

C5 与 C6 从同一 C4 checkpoint 独立启动，不是连续训练。C6 相对 C5 的差异集中在 synthetic replay 是否启用布局监督；C6 相对 C4 的差异则同时包含 replay 数据和布局监督，不能直接解释为纯布局结构收益。

## 结果

| 模型 | 页面 CER | 去空白 CER | 平均页面编辑距离 | 完全正确页面 | 相对原始 GOT2 的 CER 变化 |
|---|---:|---:|---:|---:|---:|
| C0：原始 GOT2 zero-shot reference | 1.495658 | 1.193926 | 441.289 | 0/516 | - |
| C4：AncientDoc OCR-only | 0.997051 | 0.815218 | 294.176 | 1/516 | -33.34% |
| C5：C4＋synthetic OCR replay | 1.021275 | 0.845694 | 301.324 | 1/516 | -31.72% |
| C6：C4＋synthetic layout replay | **0.934099** | **0.780829** | **275.603** | 1/516 | **-37.55%** |

当前排序为 C6、C4、C5、原始 GOT2。C4 相对原始 GOT2 的页面 CER 绝对下降 0.498607，相对下降 33.34%；C5 相对 C4 上升 0.024224，即相对变差 2.43%；C6 相对 C4 下降 0.062952，相对改善 6.31%，相对原始 GOT2 改善 37.55%。

## 结果解释

1. C4 相对 C0 zero-shot reference 有改善，但 C0 没有接受 AncientDoc 训练，因此这不是同预算训练比较，也不能据此隔离 VLQA 的贡献。
2. C5 与 C6 都从已完成的 C4 checkpoint 独立启动，不是先训练 C5 再训练 C6。因此 `C6-C5` 只比较两种 replay 方案，不能解释为连续训练收益。
3. C5 比 C4 略差，说明当前 synthetic OCR replay 的数据分布、混合比例或采样方式可能干扰真实古籍适配。该结果不支持保留当前 C5 配置作为主方案。
4. C6 是当前已经实现的 AncientDoc baseline 中最好的一项。它与“synthetic layout replay 可能提供更合适的正则化”一致，但尚未证明提升来自布局 queries 或布局监督本身。
5. C6 的完全正确页面仍只有 1/516。相对指标虽明显改善，整页转写的绝对质量仍不足，不能描述为可用或已解决古籍识别问题。

## 统计与协议限制

- 总体置信度：`CAUTION`。当前没有置信区间、逐页配对检验或多随机种子结果，模型之间的细小差异还没有稳定性证据。
- 当前 `train/validation/test` 直接采用 AncientDoc 原始 split ID 1–3/4/5。该划分尚未完成跨书籍、跨版本、跨馆藏和近重复隔离审计，不能据此声称小样本泛化或跨来源泛化。
- 已在同一 test split 查看四个配置。后续不得继续用 test 选择 replay 比例；参数选择应只使用 validation，最后在冻结配置和分组隔离 test 上评估一次。
- CER 可以因为插入错误超过 1。原始 GOT2 和 C5 的高 CER 表示严重的整页转写错误，并非超过 100% 的普通准确率。

## 统计谬误检查

- 覆盖：11/11。
- Simpson 悖论、生态谬误、碰撞变量偏差、基准率忽视、均值回归和反向因果：对当前仅含聚合 OCR 指标的结果不适用或无法判断。
- 选择偏差：`CAUTION`。AncientDoc 只是一个特定应用数据集，且分组独立性未确认。
- 幸存者偏差：四个模型均报告 516 页，未见页面数量缺失；仍需检查逐页预测中是否存在生成失败被替换为空串等情况。
- 多重寻找与分析分支：`CAUTION`。当前比较了多个训练配置和解码选择，但没有预注册选择规则。
- 因果归因：`CAUTION`。现有结果只能给 checkpoint 排序，不能证明布局监督导致了 C6 的改善。

## 2026-08-14 历史决策（已由正式重跑取代）

- 原始 split、2000-step 历史实验当时的最佳 baseline：C6；正式分组隔离 frozen test 的当前最佳已改为 C4 `checkpoint-6000`。
- C4 保留为必要的真实域 OCR-only 对照。
- C5 当前配置不继续追加训练；如需复查，只在 validation 上调整 replay 比例后重新比较。
- 在完成等参数量普通 adaptor 和无布局监督 queries 之前，不把 C6 的提升写成 VLQA 结构创新证据。

## 历史验证门槛完成情况

1. C4 与 C6 的逐页配对、来源分组和 cluster bootstrap 已完成，见正式离线验证报告。
2. 分组隔离数据的五类跨 split 泄漏审计已完成并通过。
3. validation-only checkpoint 选择和一次分组隔离 frozen test 已完成；该 test 不再用于调参。
4. 补齐等参数量普通 adaptor 和无布局监督 queries，之后再判断 C6 收益能否归因于布局结构。
