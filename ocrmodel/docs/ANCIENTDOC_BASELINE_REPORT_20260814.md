# AncientDoc 基线验证报告

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-14
- Verification Status: ANALYZED
- Version Label: ancientdoc_validation_v1

## 验证范围

- 结果目录：`/data3/yky/yangky_ocr_models/evaluation_runs/GOT/ancientdoc_validation_20260814`
- 数据：AncientDoc 转换数据的 `test` split，即原始 `split5`
- 页面数：516
- 输入协议：原始整页图像＋OCR prompt
- 布局信息：AncientDoc 没有布局标注；布局 metadata 不作为模型输入
- 结果来源：用户回传的 `summary.json` 紧凑指标
- 验证状态：只完成单次指标分析，没有独立重复运行、多随机种子区间或逐页配对显著性检验

## 结果

| 模型 | 页面 CER | 去空白 CER | 平均页面编辑距离 | 完全正确页面 | 相对原始 GOT2 的 CER 变化 |
|---|---:|---:|---:|---:|---:|
| 原始 GOT2 | 1.495658 | 1.193926 | 441.289 | 0/516 | - |
| C4：AncientDoc OCR-only | 0.997051 | 0.815218 | 294.176 | 1/516 | -33.34% |
| C5：C4＋synthetic OCR replay | 1.021275 | 0.845694 | 301.324 | 1/516 | -31.72% |
| C6：C4＋synthetic layout replay | **0.934099** | **0.780829** | **275.603** | 1/516 | **-37.55%** |

当前排序为 C6、C4、C5、原始 GOT2。C4 相对原始 GOT2 的页面 CER 绝对下降 0.498607，相对下降 33.34%；C5 相对 C4 上升 0.024224，即相对变差 2.43%；C6 相对 C4 下降 0.062952，相对改善 6.31%，相对原始 GOT2 改善 37.55%。

## 结果解释

1. C4 表明，在当前页面协议和数据划分下，使用 AncientDoc 真实 OCR 真值进行适配能明显改善原始 GOT2 的域内识别结果。
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

## 当前决策

- AncientDoc 当前最佳已实现 baseline：C6。
- C4 保留为必要的真实域 OCR-only 对照。
- C5 当前配置不继续追加训练；如需复查，只在 validation 上调整 replay 比例后重新比较。
- 在完成等参数量普通 adaptor 和无布局监督 queries 之前，不把 C6 的提升写成 VLQA 结构创新证据。

## 下一验证门槛

1. 对 C4 与 C6 做逐页配对错误分析，统计改善、持平和退化页面，并按文本长度、版式和书籍来源分组。
2. 审计 train/validation/test 的书籍、版本、馆藏、来源 ID、图片哈希及图文近重复泄漏。
3. 只在 validation 上选择 replay 比例，然后在分组隔离的 test 上执行一次冻结评估。
4. 补齐等参数量普通 adaptor 和无布局监督 queries，之后再判断 C6 收益能否归因于布局结构。
