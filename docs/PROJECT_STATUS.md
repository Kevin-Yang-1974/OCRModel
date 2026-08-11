# 项目状态

> 更新日期：2026 年 8 月 12 日
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

## 4. 已获得的工程证据

| 运行 | 证明范围 | 结果与限制 |
|---|---|---|
| `layout_pilot_20260811_023528` | CUDA forward/backward、整页 batch、P1→P2 加载、保存和完整模型重载 | 链路通过；仅 2 页且每阶段重复 500 epoch，无 validation，不是有效 pilot |
| `layout_overfit_20260811_110317` | 固定 P1、2 条记录、200 steps 的可拟合性诊断 | `overfit_assessment.status=fail`；不能进入 P2 或正式训练 |
| `layout_overfit_20260811_113817` | 修复后固定 P1、2 条记录、200 steps 的尺度与学习信号诊断 | 初始化修复有效；bbox 有学习信号但 200 steps 未达到实现门槛 |
| `layout_overfit_20260812_002747` | 固定 P1、2 条记录、1000 steps 的实现诊断 | `overfit_assessment.status=pass`；只证明实现可拟合，不是 validation 或性能结果 |
| `layout_validate_20260812_012924` | 首轮 prompt-only validation 启动检查 | evaluator 的 tokenizer 文件名白名单错误拒绝原始 GOT/Qwen tokenizer；未进入 checkpoint 加载、forward 或 generation，不是模型失败或性能结果 |
| `layout_validate_20260812_014816` | 修复后两页 prompt-only validation 链路 | 输入协议正确；同一 P1 overfit checkpoint/`train` split 上 16/16 区域匹配、bbox mean IoU `0.956666`、方向和阅读顺序均为 `1.0`；不是正式性能结果 |

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

- 在正式隔离的 held-out split 上，用明确的 VLQA checkpoint 完成真实整页 validation，并核对生成、布局输出和 summary；当前两页链路验证不能替代该步骤；
- 正式 Chromium、字体文件、版本、哈希和缺字覆盖锁定；
- 第一批真实 crop 的 `S1-html-crop` 内容清单；
- 正式 validation/test split 与跨来源泄漏审计；
- `A0`–`A6` 同划分、同预算的消融启动器；
- 小样本真实整页适配、跨域评估和 64×64 可选路径；
- 任何足以支持 VLQA 性能提升的实验结果。

## 6. 下一步门槛

1. 建立按来源组隔离的正式 validation/test split，并完成 manifest 泄漏审计；当前 `train` 两页只作为链路证据。
2. 用原 GOT2 页面 baseline 和明确的 VLQA checkpoint 在正式 split 上执行 prompt-only validation，继续只回传紧凑完成 JSON。
3. 锁定正式 Chromium/字体包，建立 `S1-html-crop` 内容清单和缺字覆盖报告。
4. 按同一数据、预算和指标执行 `A0`–`A6` 消融；P2/P3 和正式训练在 protocol 锁定前不启动。
5. 没有正式定位误差证据前不接入 64×64 分支；不能用两页 overfit bbox 结果决定分辨率升级。

## 7. 结果表述边界

- 可以表述：整页生成、审计、CUDA forward/backward、P1→P2 与 checkpoint 链路已打通。
- 必须表述：`layout_overfit_20260812_002747` 和 `layout_validate_20260812_014816` 都只涉及两页同分布链路/实现诊断；正式 held-out validation 尚未完成。
- 不得表述：VLQA 已改善 OCR、布局、阅读顺序或小样本泛化。
- AncientDoc 旧 split 有书籍级重叠，只能作为历史兼容基线。
- line-level 与 whole-page 结果不可直接比较。
- 双 GOT2 两阶段系统必须单独报告两次模型成本和错误传播。
