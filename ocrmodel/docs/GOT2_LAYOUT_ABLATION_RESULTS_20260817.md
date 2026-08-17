# GOT2 整页布局结构消融结果

## 记录范围

- 日期：2026 年 8 月 17 日。
- 数据集：`formal_pdf_short_seed20260812`。
- 输入协议：`whole_page_image`＋OCR prompt；bbox、方向和顺序不作为 validation/test 模型输入。
- 实验前缀：`layout_ablation_formal_v1`，seed=`42`。
- 训练预算：A1–A4 均训练至 P2 8000 steps；A5 为 P1 4000 steps＋P2 8000 steps。A5 总训练步数和页面曝光量高于 A4，不能视为相同总预算。
- 选点协议：只按 validation page CER、去空白 page CER、较早 optimizer step 选择 checkpoint；test 不参与选点。
- 评测范围：本记录仅为 300 页 Synthetic-ID frozen test。不能外推到 Synthetic-OOD、Real-OOD、真实古籍、真实谱面或多场景泛化。
- A0 `got2_zero_shot` 未包含在本次回传结果中，本记录不补造 A0 指标。

各组完整 checkpoint 路径、config hash、weights hash 和候选列表以服务器对应的 `selection.json` 为准：

```text
$GOT_EVALUATION_RUNS/layout_ablation_formal_v1_<ablation>_seed42_selection/selection.json
```

## OCR 结果

| 组别 | validation 选中 step | validation page CER | validation 去空白 CER | test page CER | test 去空白 CER | test 编辑距离/参考字符 | exact match | 推理失败 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 `projector_only` | 8000 | 0.134710 | 0.131615 | 0.086320 | 0.084851 | 2493/28881 | 68/300 | 0 |
| A2 `generic_adapter_projector` | 4000 | 0.133172 | 0.131104 | 0.158236 | 0.146325 | 4570/28881 | 70/300 | 0 |
| A3 `vlqa_ocr_only` | 8000 | 0.135233 | 0.132912 | 0.069977 | 0.065576 | 2021/28881 | 74/300 | 0 |
| A4 `vlqa_layout_direct` | 6000 | 0.161966 | 0.148430 | 0.235795 | 0.204150 | 6810/28881 | 73/300 | 0 |
| A5 `vlqa_layout_p1_p2` | 8000 | 0.142530 | 0.141084 | 0.070531 | 0.066355 | 2037/28881 | 71/300 | 0 |

A1–A4 的最大 P2 训练预算相同，但 validation 分别选中了 8000、4000、8000 和 6000 steps；因此应同时报告最大训练预算和选中 checkpoint step，不能写成所有被测权重具有相同曝光量。

## 布局结果

A1/A2 不含 VLQA，布局指标为 `null`。A3 保留布局 heads，但布局 loss 全为 0；其 object head 把全部 16 个 slots 判为对象，导致每页固定产生 16 个区域，且没有 bbox 匹配，因此这些输出只说明未监督布局 heads 不可直接解释。

| 组别 | complete precision | complete recall | complete F1 | ordered object recall | ordered bbox mean IoU | matched bbox mean IoU | ordered direction accuracy | reading-order pair accuracy | Kendall tau |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A3 `vlqa_ocr_only` | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.118839 | null | 0.000000 | null | null |
| A4 `vlqa_layout_direct` | 0.622473 | 0.597603 | 0.609785 | 0.954338 | 0.515068 | 0.762578 | 0.920662 | 1.000000 | 1.000000 |
| A5 `vlqa_layout_p1_p2` | 0.699769 | 0.691781 | 0.695752 | 0.976027 | 0.593544 | 0.734961 | 0.993151 | 0.999530 | 0.999061 |

reading-order 指标只在存在足够匹配区域的可评测页面和区域对上计算。A3 没有匹配区域，因此其 reading-order 指标不可用，不能把 `null` 解释为零分。

## 受控比较

差值统一定义为右侧组减左侧组；CER 差值为负表示右侧更好。

| 比较 | test page CER 差值 | 当前协议下的观察 |
|---|---:|---|
| A2−A1 | +0.071916 | 普通 generic adaptor 没有带来容量收益，反而高于 projector-only。A2 虽在 validation 选中 4000 steps，但 test 退化，需按 tier/source 检查分布异质性。 |
| A3−A2 | -0.088259 | 在近似新增容量控制下，VLQA query/cross-attention/write-back 路径优于普通 adaptor；这一观察支持查询结构有贡献，但目前只有一个 seed 和 Synthetic-ID。 |
| A3−A1 | -0.016343 | VLQA OCR-only 优于 projector-only，说明收益不只来自 projector 域适配；仍需新 seed/OOD 复验稳定性。 |
| A4−A3 | +0.165818 | 直接加入布局辅助监督显著改善布局输出，但 OCR 明显退化，呈现强任务权衡或优化干扰；不能据此泛化为“布局监督必然有害”。 |
| A5−A4 | -0.165264 | P1→P2 相比直接 P2 基本恢复 OCR，同时 complete F1 提高 0.085968、ordered bbox IoU 提高 0.078476。该差异同时包含 P1 预热和额外 4000 steps/页面曝光量。 |
| A5−A3 | +0.000554 | A5 与 A3 的 OCR 基本持平，而 A5 获得可用布局输出；去空白 CER 差值为 +0.000779。当前单 seed 下不能宣称二者 OCR 存在稳定差异。 |

页面 exact match 与聚合 CER 的排序并不一致，例如 A4 exact match 为 73 页但 CER 最差，说明错误严重度集中在部分非 exact 页面。正式模型比较继续以预声明的聚合 page CER 为主，不用 exact match 反向改写选点规则。

## 当前结论与边界

1. 新增参数量本身不能解释 VLQA OCR 收益：A2 明显弱于 A3。
2. 无布局监督的 VLQA OCR 路径 A3 在本轮 Synthetic-ID 上取得最低 CER。
3. A4 表明直接从原始 GOT2 同时学习 OCR 和完整布局监督会产生明显 OCR–layout 权衡。
4. A5 表明 P1 布局预热在当前协议下可以缓解该权衡：OCR 接近 A3，同时布局优于 A4。
5. A5 相对 A4 不是同总预算比较；可归因为“现有 P1 预热流程整体贡献”，不能进一步拆成纯初始化效应与额外数据曝光效应。
6. 当前只有 seed 42 和 Synthetic-ID。尚不能声称真实域泛化、跨场景泛化、稳定统计优势或布局监督的普遍收益。
7. test 已冻结使用，不再用于选择新的 loss 权重、P1 步数或结构。二级 loss 消融和新配置应先在 validation 决策，再使用新预注册 test 或新 seed 复验。

## 后续验证

1. 补齐 A0 zero-shot 的同协议 validation/test 记录，完成合成域适配基准链。
2. 对 A2 validation/test 反转和 A4 OCR 退化进行 validation-only 的 tier、区域数、方向和文本长度分组分析。
3. 在 validation 上执行 `object_only`、`object_bbox`、`object_direction_order` 等二级 loss 消融，保持 A3/A4 结构完全相同。
4. 预注册至少一个新 seed，并补充 Synthetic-OOD 和 Real-OOD；不同输入粒度不得进入同一结果表。

