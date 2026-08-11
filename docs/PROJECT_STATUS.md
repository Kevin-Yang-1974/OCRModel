# 项目状态

> 更新日期：2026 年 8 月 11 日
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

## 4. 已获得的工程证据

| 运行 | 证明范围 | 结果与限制 |
|---|---|---|
| `layout_pilot_20260811_023528` | CUDA forward/backward、整页 batch、P1→P2 加载、保存和完整模型重载 | 链路通过；仅 2 页且每阶段重复 500 epoch，无 validation，不是有效 pilot |
| `layout_overfit_20260811_110317` | 固定 P1、2 条记录、200 steps 的可拟合性诊断 | `overfit_assessment.status=fail`；不能进入 P2 或正式训练 |

首次 overfit 的第 1 步 object 与 direction logit 绝对值上限均约为 1680，bbox 已饱和到约 0/1，bbox mean IoU 为 0。末 20 步 object loss 为 3.5654、bbox L1 为 1.9023、bbox GIoU 为 1.8513、object accuracy 为 0.5375、bbox mean IoU 仍为 0。query 梯度存在，说明监督链路没有简单断开；direction accuracy 为 1.0 也不能作为证据，因为两页有效区域只有一个方向类别。

异常在首次优化前已经出现，因此“训练后期发散”不是主要解释。当前根因假设是：原始 GOT2 checkpoint 不含 VLQA 权重，而旧加载路径没有可靠初始化全部自定义参数。针对该假设，本地代码已加入：

- 原始 checkpoint 后显式完整重置 VLQA；
- VLQA 键“全无、全有或拒绝部分状态”的严格审计；
- 辅助预测前最终 LayerNorm；
- bbox 输出层小尺度初始化；
- component smoke 和训练首步 raw-logit 尺度保护。

这些改动只完成了本地实现，根因尚未由同协议 A100 复测确认。

## 5. 尚未完成

- 修复版 P1 两样本 overfit 复测；
- 正式 Chromium、字体文件、版本、哈希和缺字覆盖锁定；
- 第一批真实 crop 的 `S1-html-crop` 内容清单；
- validation loader 与页面 CER、bbox、方向、区域召回和阅读顺序统一指标；
- `A0`–`A6` 同划分、同预算的消融启动器；
- 小样本真实整页适配、跨域评估和 64×64 可选路径；
- 任何足以支持 VLQA 性能提升的实验结果。

## 6. 下一步门槛

1. 在原来的两页数据和固定 P1 配置上复测 `--mode overfit`。
2. 确认 `initialization.mode=fresh_explicit_reset`、`source_layout_tensor_count=0`，并检查首步 raw logits 不再出现 1680 级异常。
3. 若首步尺度恢复但仍不能拟合，依次检查有序槽位目标、object 正负样本、bbox 参数化和可训练范围，不扩大数据。
4. overfit 通过后再锁定浏览器/字体、构建 S1 内容并实现 validation。
5. validation 完成后再执行 `A0`–`A6` 同预算消融；没有真实页面迁移证据前不接入 64×64 分支。

## 7. 结果表述边界

- 可以表述：整页生成、审计、CUDA forward/backward、P1→P2 与 checkpoint 链路已打通。
- 必须表述：首次受控 P1 两样本 overfit 失败，初始化修复待复测。
- 不得表述：VLQA 已改善 OCR、布局、阅读顺序或小样本泛化。
- AncientDoc 旧 split 有书籍级重叠，只能作为历史兼容基线。
- line-level 与 whole-page 结果不可直接比较。
- 双 GOT2 两阶段系统必须单独报告两次模型成本和错误传播。
