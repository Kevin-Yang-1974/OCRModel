# 项目状态

> 更新日期：2026 年 8 月 13 日
>
> 状态依据：代码、受限运行摘要与已完成的本地/A100 诊断
>
> 注意：本页区分已实现、工程验证、待验证假设和正式实验结果

## 1. 当前任务

项目已进入工程实现与实验验证阶段。研究目标是小样本条件下的多场景通用符号识别，当前代码路线是在单个 GOT2 内加入端到端整页布局 queries。

主模型推理输入只有原始整页图像和 OCR prompt。区域 bbox、阅读顺序和书写方向用于训练辅助监督、可选解释性输出或评测，不作为推理条件。line-level 图像只用于字符诊断、AnandaSky 兼容对照或合成页面内容来源。

## 2. 当前候选

Visual Layout Query Adapter（VLQA）从 GOT2 的整页视觉 token 中读取固定上限的有序区域 queries，预测区域存在性、bbox 和书写方向，再以零初始化残差门将布局信息写回原 256 个视觉 token。首轮使用 GOT2 现有的 16×16 最终视觉网格；64×64 中间特征只在定位误差证明 16×16 不足后考虑。

页面 OCR 自回归交叉熵是唯一直接优化转写的主损失。布局损失包括区域存在性、bbox L1/GIoU 和方向分类；阅读顺序通过有序 query 与真值区域的目标分配监督。布局损失下降不能替代页面 OCR 和阅读顺序指标。

## 3. 已实现

- HTML/Playwright 整页生成器，支持 `S0-html-text`、`S1-html-crop` 和 `S2-hard`。
- DOM bbox、实际字体、图片/内容哈希、页面转写与跨 split 泄漏审计。
- 页面级 manifest dataset/collator 和固定上限的布局监督张量。
- 16×16 VLQA、object/bbox/direction 辅助头、零门控写回和 FP32 布局损失。
- 原始 GOT2、P1 checkpoint 和部分 VLQA checkpoint 的布局键完整性审计。
- `P1` 布局预热、`P2` OCR 联合训练开发入口，以及受限 A100 编排器。
- 训练分项、query/raw-logit/bbox 范围、梯度、残差门和 checkpoint 重载诊断。
- prompt-only 整页 validation dataset/collator、GPU evaluator 和统一页面指标代码；布局标注只进入离线评测。
- 正式 A100 编排模式：`pretrain` 从原始 GOT2 只运行 P1，`joint-train` 从完整 P1 VLQA checkpoint 只运行 P2；两者都要求 train/validation/test manifest 联合审计。
- 原始 GOT2 与 VLQA 对照脚本 `tools/evaluation/compare_got2_vlqa.py`；它强制同一 test 页面、OCR prompt、tokenizer 和贪心解码，VLQA 只接受 P2 checkpoint。
- 离线错误分析、threshold sweep、slot alignment、fixed offset sweep 和 analysis bundle 汇总入口；这些脚本只读取 comparison run 与 analysis JSON，不重新推理、不占 GPU。
- P2 loss-supervision 消融入口 `tools/training/run_formal_layout_ablation.sh`；当前支持 `full`、`no-direction`、`no-bbox`、`object-only` 和 `ocr-only-adapter` 五个 loss 权重 preset。
- 消融后处理短入口 `tools/evaluation/run_formal_layout_post_analysis.sh`；对一个完成的 P2 checkpoint 串行执行 formal comparison、三项离线诊断和 analysis bundle，并拒绝覆盖已有 comparison run。

## 4. 已获得的工程证据

| 运行 | 证明范围 | 结果与限制 |
|---|---|---|
| `layout_pilot_20260811_023528` | CUDA forward/backward、整页 batch、P1→P2 加载、保存和完整模型重载 | 链路通过；仅 2 页且每阶段重复 500 epoch，无 validation，不是有效 pilot |
| `layout_overfit_20260811_110317` | 固定 P1、2 条记录、200 steps 的可拟合性诊断 | `overfit_assessment.status=fail`；不能进入 P2 或正式训练 |
| `layout_overfit_20260811_113817` | 修复后固定 P1、2 条记录、200 steps 的尺度与学习信号诊断 | 初始化修复有效；bbox 有学习信号但 200 steps 未达到实现门槛 |
| `layout_overfit_20260812_002747` | 固定 P1、2 条记录、1000 steps 的实现诊断 | `overfit_assessment.status=pass`；只证明实现可拟合，不是 validation 或性能结果 |
| `layout_validate_20260812_012924` | 首轮 prompt-only validation 启动检查 | evaluator 的 tokenizer 文件名白名单错误拒绝原始 GOT/Qwen tokenizer；未进入 checkpoint 加载、forward 或 generation，不是模型失败或性能结果 |
| `layout_validate_20260812_014816` | 修复后两页 prompt-only validation 链路 | 输入协议正确；同一 P1 overfit checkpoint/`train` split 上 16/16 区域匹配、bbox mean IoU `0.956666`、方向和阅读顺序均为 `1.0`；不是正式性能结果 |
| `layout_pretrain_20260813_003142` | 首轮正式合成数据 P1，`formal_pdf_short_seed20260812`，1000 steps，300 页 validation | 页面 CER `0.782540`，去空白 CER `0.676580`；完整区域 precision/recall/F1 `0.464259/0.363951/0.408031`；matched bbox mean IoU `0.683627`；有序槽位 bbox mean IoU `0.320758`；证明 P1 正式入口与 held-out validation 已跑通，不是 OCR 性能结论 |
| `layout_joint-train_20260813_012356` | 首轮正式合成数据 P2，接 P1 checkpoint，2000 steps，300 页 validation | 页面 CER `0.198253`，去空白 CER `0.188622`；完整区域 precision/recall/F1 `0.354590/0.372617/0.363380`；matched bbox mean IoU `0.678319`；有序槽位 bbox mean IoU `0.276798`；OCR 较 P1 明显改善，但布局 F1 和槽位定位未改善，不能据此归因到布局查询或外推真实跨域 |
| `layout_pretrain_4000_20260813` | 长程正式合成数据 P1，4000 steps，300 页 validation | OCR 与 P1 1000 相同，页面 CER `0.782540`；布局完整区域 precision/recall/F1 `0.373609/0.407279/0.389718`；matched bbox mean IoU `0.703180`；有序槽位 bbox mean IoU `0.329817`；P1 长训未明显改善 ordered-slot 指标 |
| `layout_joint-train_8000_20260813` | 长程正式合成数据 P2，接 P1 4000 checkpoint，8000 steps，300 页 validation | 页面 CER `0.139487`，去空白 CER `0.138923`；完整区域 precision/recall/F1 `0.770047/0.754477/0.762183`；matched bbox mean IoU `0.761135`；有序槽位 bbox mean IoU `0.652210`；相对 P2 2000，OCR 和布局均显著继续收敛，说明训练步数不足是首轮布局低分的重要原因 |
| `got2_vlqa_compare_20260813_020251` | 首轮正式合成 test，同一 `formal_pdf_short_seed20260812/test` split，原始 GOT2 vs P2 VLQA | 300 页；VLQA - baseline 页面 CER delta `-0.087047`，去空白 CER delta `-0.058194`，page exact match rate delta `+0.19`；VLQA 布局完整区域 precision/recall/F1 `0.367591/0.385845/0.376497`，matched bbox mean IoU `0.682537`，有序槽位 bbox mean IoU `0.269150`；证明合成 test 同协议 OCR 对照优于原始 GOT2，但尚不能归因到布局查询或外推真实跨域 |
| `got2_vlqa_compare_p2_8000_20260813` | 长程正式合成 test，同一 `formal_pdf_short_seed20260812/test` split，原始 GOT2 vs P2 8000 VLQA | 300 页；VLQA - baseline 页面 CER delta `-0.260725`，去空白 CER delta `-0.211409`，page exact match rate delta `+0.213333`；VLQA 布局完整区域 precision/recall/F1 `0.796957/0.777397/0.787056`；matched bbox mean IoU `0.758097`；有序槽位 bbox mean IoU `0.647827`；reading-order pair accuracy `0.991988`，Kendall tau `0.983977`；确认 P2 8000 的 validation 改善同步转化为 test 改善 |
| `got2_vlqa_compare_20260813_020251/analysis` | 离线 test 错误分析，300 页 baseline/VLQA predictions 对齐 | VLQA 总编辑距离 `7042`，baseline 为 `9556`，减少 `2514`；按页 `246` 页改善、`22` 页持平、`32` 页变差；exact match `68` vs `11`，增加 `57` 页；layout failure type 为 `miss_and_extra=285`、`extra_regions=11`、`missed_regions=1`、`layout_ok=3`，提示布局主问题是同页漏检与多检并存 |
| `got2_vlqa_compare_20260813_020251/analysis/threshold_sweep` | 离线 object threshold sweep，不重新推理 | 最优 complete region F1 在 `object_threshold=0.2`，precision/recall/F1 `0.362086/0.412100/0.385478`；相比默认 test F1 `0.376497` 只小幅提高；`miss_and_extra=281/300`，false negative `1030`，false positive `1272`；说明低布局 F1 不是单纯阈值问题，更应检查 query 槽位分配、预测区域数和 layout loss 权重 |
| `got2_vlqa_compare_20260813_020251/analysis/slot_alignment` | 离线 slot alignment 诊断，不重新推理 | 1752 个区域中 ordered hits `372`，best hits `737`，slot-misaligned hits `365`；ordered hit rate `0.212329`，best hit rate `0.420662`；mean ordered IoU `0.269150`，mean best IoU `0.439486`；best query offset 中 `+1=504`，提示 query 槽位和阅读顺序存在明显错配，但 best hit rate 仍只有约 `42%`，说明也有真实定位失败 |
| `got2_vlqa_compare_20260813_020251/analysis/slot_offset_sweep` | 离线 fixed slot-offset sweep，不重新推理 | 最佳固定偏移为 `offset=+1`，hit rate `0.227169`，mean IoU `0.279240`；相比默认 ordered hit rate `0.212329` 和 mean IoU `0.269150` 只小幅提升，排除简单全局 off-by-one 作为主因 |
| `got2_vlqa_compare_p2_8000_20260813/analysis` | P2 8000 离线 test 错误分析，300 页 baseline/VLQA predictions 对齐 | VLQA 总编辑距离 `2026`，baseline 为 `9556`，减少 `7530`；从 predictions 重新计算的页面 CER `0.070150`，baseline 为 `0.330875`；按页 `264` 页改善、`16` 页持平、`20` 页变差；exact match `75` vs `11`，增加 `64` 页；layout failure type 为 `miss_and_extra=169`、`layout_ok=124`、`missed_regions=4`、`extra_regions=3`，相对 P2 2000 的 `layout_ok=3` 和 `miss_and_extra=285` 明显改善 |
| `got2_vlqa_compare_p2_8000_20260813/analysis/threshold_sweep` | P2 8000 离线 object threshold sweep，不重新推理 | 最优 complete region F1 在 `object_threshold=0.3`，precision/recall/F1 `0.790046/0.788242/0.789143`；相比默认 test F1 `0.787056` 仅轻微提升；预测区域数 `1748` 与真值区域数 `1752` 基本匹配，exact region count pages `244/300`；说明当前剩余布局误差不主要来自阈值或系统性区域数量偏差 |
| `got2_vlqa_compare_p2_8000_20260813/analysis/slot_alignment` | P2 8000 离线 slot alignment 诊断，不重新推理 | 1752 个区域中 ordered hits `1349`，best hits `1412`，slot-misaligned hits `63`；ordered hit rate `0.769977`，best hit rate `0.805936`；mean ordered IoU `0.647827`，mean best IoU `0.681584`；best query offset 中 `0=1494`，明显高于 `+1=91` 和 `-1=52`，说明 P2 2000 阶段明显的槽位错配在 P2 8000 已基本收敛 |
| `got2_vlqa_compare_p2_8000_20260813/analysis/analysis_bundle` | P2 8000 离线 analysis bundle 汇总，不重新推理 | `threshold_f1_gain_over_default=0.002087`，`threshold_is_main_issue=false`；slot alignment gap `0.035959`，`slot_misalignment_is_main_issue=false`；top query offset 为 `0=1494`；建议优先检查 group/page priorities、抽查剩余 `miss_and_extra` 页面、启动同预算消融，并在缺少定位证据前暂缓 64×64 分支 |

首次 overfit 的第 1 步 object 与 direction logit 绝对值上限均约为 1680，bbox 已饱和到约 0/1，bbox mean IoU 为 0。末 20 步 object loss 为 3.5654、bbox L1 为 1.9023、bbox GIoU 为 1.8513、object accuracy 为 0.5375、bbox mean IoU 仍为 0。query 梯度存在，说明监督链路没有简单断开；direction accuracy 为 1.0 也不能作为证据，因为两页有效区域只有一个方向类别。

异常在首次优化前已经出现，因此“训练后期发散”不是主要解释。当前根因假设是：原始 GOT2 checkpoint 不含 VLQA 权重，而旧加载路径没有可靠初始化全部自定义参数。针对该假设，本地代码已加入：

- 原始 checkpoint 后显式完整重置 VLQA；
- VLQA 键“全无、全有或拒绝部分状态”的严格审计；
- 辅助预测前最终 LayerNorm；
- bbox 输出层小尺度初始化；
- component smoke 和训练首步 raw-logit 尺度保护。

这些改动已由同协议 A100 复测确认有效；修复组合尚未做单项消融。

`layout_overfit_20260812_002747` 的末 20 步均值为：object loss `0.00201755`、bbox L1 `0.00531464`、bbox GIoU `0.05508578`、direction loss `0.00154159`、object/direction accuracy `1.0`、bbox mean IoU `0.94497279`。尾段范围也保持在实现阈值内。该 run 仍只有两页，不能外推到正式数据或泛化性能。

`layout_validate_20260812_012924` 在 evaluator 自定义的 tokenizer 文件名预检处退出。该预检只接受四类标准文件，但 GOT/Qwen 可以使用 `qwen.tiktoken` 和自定义 tokenizer 代码；同一原始模型目录此前已由 P1 训练入口直接通过 `AutoTokenizer.from_pretrained` 成功加载。修复已删除文件名白名单，改为对本地候选目录执行 Transformers 离线实际加载并保留限长的逐候选错误摘要，随后由 `layout_validate_20260812_014816` 验证通过。

修复后的 `layout_validate_20260812_014816` 已在同一 P1 checkpoint 上完成两页 `train` split 的 prompt-only 链路验证。模型输入为 `whole_page_image` 和 `ocr_prompt`，`layout_metadata_as_model_input=false`。OCR 页面 CER 为 `0.215909`（38/176 字符编辑），去空白 CER 为 `0.086420`（14/162），页面 exact match 为 `0/2`；布局 16 个标注区域全部匹配，完整区域 precision/recall/F1 均为 `1.0`，有序槽位 bbox mean IoU 为 `0.956666`，方向准确率、reading-order pair accuracy 和 Kendall tau 均为 `1.0`。该 checkpoint 来自同两页 P1 overfit，且 P1 的 `ocr_loss_weight=0`、写回门为 0，因此这些 OCR/布局数字只证明重载与评测链路，不代表 VLQA 已改善页面识别或泛化。

## 5. 尚未完成

- 系统 Edge、字体文件、版本、哈希和缺字覆盖锁定；
- 第一批真实 crop 的 `S1-html-crop` 内容清单；
- `A0`–`A6` 中等参数量 adaptor、无布局监督 queries 和旧 oracle/pseudo-layout region-token adapter 的结构消融入口；
- 对 P2 8000 `analysis/group_error_analysis.csv`、`analysis/error_analysis.md` 和 `analysis/slot_alignment/slot_alignment.md` 的人工解读与最差页面抽查；
- 小样本真实整页适配、跨域评估和 64×64 可选路径；
- 任何足以把收益归因到 VLQA 结构本身，或支持真实跨域泛化的实验结果。

## 6. 下一步门槛

1. 从 `layout_pretrain_4000_20260813/p1/model` 启动单个 P2 loss-supervision 消融 sanity run，建议优先 `no-direction` 或 `no-bbox`，并保持 `formal_pdf_short_seed20260812`、P2 `8000` steps、同一 comparison 与离线 analysis 链路。
2. 读取 P2 8000 的 `analysis/group_error_analysis.csv`、`analysis/error_analysis.md`、`analysis/slot_alignment/slot_alignment.md` 和 `analysis_bundle.md`，定位剩余 `miss_and_extra=169`、OCR 退化 `20` 页和 slot-gap 最大页面的共同特征。
3. 继续补齐 `A0`–`A6` 中尚未实现的结构消融入口：等参数量 adaptor、无布局监督 queries 和旧 oracle/pseudo-layout region-token adapter。只有完整 VLQA 同时优于普通容量对照和无布局监督 queries，才能把收益归因于布局监督结构。
4. 在 P2 8000 诊断稳定和消融结果完成后，再启动 P3 真实页适配和跨域评估。
4. 锁定正式字体包，建立 `S1-html-crop` 内容清单和缺字覆盖报告。
5. 没有正式定位误差证据前不接入 64×64 分支；不能只因首轮 P2 槽位 IoU 偏低就直接扩大结构。

## 7. 结果表述边界

- 可以表述：整页生成、审计、CUDA forward/backward、P1→P2 与 checkpoint 链路已打通。
- 可以表述：正式合成数据 `formal_pdf_short_seed20260812` 的 P1 1000 steps 与 P2 2000 steps 已完成，且均完成 300 页 validation；P2 相比 P1 明显降低合成 validation OCR CER。
- 可以表述：P1 4000/P2 8000 长程训练已完成；P2 8000 validation 页面 CER `0.139487`，布局 F1 `0.762183`，有序槽位 bbox IoU `0.652210`，相对 P2 2000 显著改善。
- 可以表述：P2 8000 在同一合成 test split 上相对原始 GOT2 的页面 CER delta 为 `-0.260725`，布局 F1 为 `0.787056`，有序槽位 bbox IoU 为 `0.647827`；长程训练收益已同步体现在 validation 和 test。
- 可以表述：`got2_vlqa_compare_20260813_020251` 显示 P2 VLQA 在同一合成 test split 上相对原始 GOT2 的页面 CER delta 为 `-0.087047`，页面 exact match rate delta 为 `+0.19`。
- 可以表述：P2 8000 离线错误分析显示 300 页中 VLQA 有 `264` 页 OCR 改善、`16` 页持平、`20` 页变差，exact match 从 `11` 增至 `75` 页，layout failure 中 `layout_ok=124`、`miss_and_extra=169`。
- 可以表述：P2 8000 threshold sweep 的最佳 layout F1 为 `0.789143`，只比默认 `0.787056` 轻微提高，说明默认阈值已经接近最优。
- 可以表述：P2 8000 slot alignment 显示 ordered hit rate `0.769977`、best hit rate `0.805936`、slot-misaligned hit rate `0.035959`，best query offset 以 `0=1494` 为主，说明 P2 2000 阶段明显的槽位错配在长程训练后已基本收敛。
- 可以表述：P2 8000 analysis bundle 进一步确认 threshold 和 slot mismatch 都不是当前主因，下一步应转向 hard-page 抽查和同预算消融。
- 可以表述：P2 2000 的离线错误分析、threshold sweep、slot alignment 和 fixed offset sweep 是短步数历史诊断，可用于解释为什么需要 P2 8000，但不再作为当前主 checkpoint 的剩余错误归因。
- 可以表述：现有 `run_formal_layout_ablation.sh` 只覆盖 loss-supervision 消融，不能替代等参数结构对照。
- 必须表述：`layout_overfit_20260812_002747` 和 `layout_validate_20260812_014816` 都只涉及两页同分布链路/实现诊断；`layout_pretrain_20260813_003142`、`layout_joint-train_20260813_012356`、`layout_pretrain_4000_20260813`、`layout_joint-train_8000_20260813`、`got2_vlqa_compare_20260813_020251` 和 `got2_vlqa_compare_p2_8000_20260813` 都是合成数据阶段结果，尚未完成消融和真实跨域验证。
- 不得表述：VLQA 已稳定改善布局、阅读顺序或小样本泛化；也不得在缺少等参数量和无布局监督消融时把 test OCR 改善直接归因到布局查询。
- AncientDoc 旧 split 有书籍级重叠，只能作为历史兼容基线。
- line-level 与 whole-page 结果不可直接比较。
- 双 GOT2 两阶段系统必须单独报告两次模型成本和错误传播。
