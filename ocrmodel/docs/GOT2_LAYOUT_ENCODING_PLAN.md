# GOT2 整页端到端布局查询方案

> 更新日期：2026 年 8 月 12 日
>
> 文档性质：单模型 VLQA 路线的架构、目标函数与消融执行依据
>
> 当前状态：整页生成与审计及 A100 工程链路已经打通；`layout_overfit_20260812_002747` 已通过固定 1000 steps 的 P1 两样本实现诊断，object/direction/bbox 均达到实现门槛；tokenizer 预检修复后的 `layout_validate_20260812_014816` 已在两页 `train` split 上完成 prompt-only 链路验证，但正式 held-out validation、消融和效果验证仍未完成；本文参数仍为首轮候选

## 1. 当前任务与输入口径

本仓库以 GOT2 原生整页 OCR 能力为基础，研究在小样本条件下显式强化页面中的列位置、阅读顺序和书写方向。正式训练、验证和推理的主输入为原始整页图像；模型在推理时只接收页面图像和 OCR prompt，不要求外部提供 bbox、列序或方向。

GOT 原论文同时支持 slice 与 whole-page 图像，并将单张 $1024\times1024$ 图像压缩为 $256\times1024$ 视觉 token [1]。当前活动源码的标准 demo 也直接读取整张图像，经 `BlipImageEvalProcessor(image_size=1024)` 缩放后送入 Vary ViT。因此，整页输入是 GOT2 的原生能力，不是新增假设。

旧版方案把页面先裁成单行或单列，再把原页面 bbox、列序和方向作为外部字段送回识别器。该路线保留为显式元数据对照，但不再是当前主方案。line-level 图像仍可用于以下用途：

1. 作为 HTML 合成页面的真实内容来源；
2. 诊断单行或单列字符识别能力；
3. 与 AnandaSky 进行独立的 line-level 兼容比较；
4. 构造页面级转写标签，但不替代整页主输入。

“第一个 GOT2 做列识别/分割，再送入第二个 GOT2 做识别”属于仓库外的独立两阶段系统对照。本仓库路线只使用一个 GOT2，在其内部从整页视觉特征生成软布局查询；两条路线不合并实现或报告。

## 2. 核心瓶颈

原始 GOT2 已包含二维绝对和相对位置编码，能够从整页图像生成具有页面坐标语义的视觉 token。但页面被高度压缩为 $16\times16=256$ 个最终视觉 token，OCR 自回归损失也不保证模型形成可解释、稳定的列级表示。在小样本、复杂竖排、多栏、边注或跨版式条件下，模型可能出现以下问题：

1. 多个列或区域在高压缩视觉表示中混合；
2. OCR loss 能优化字符序列，却不能直接约束列定位、书写方向和阅读顺序；
3. 页面模板变化时，隐式阅读顺序可能依赖训练分布中的位置捷径；
4. 直接增加普通视觉 adaptor 即使提高 OCR，也无法证明收益来自布局建模。

因此，本方案要验证的不是“GOT2 是否具有坐标”，而是：**在整页视觉 token 上增加受布局监督的少量可学习查询，能否以较小参数和计算代价，提高小样本条件下的页面识别与阅读顺序鲁棒性。**

## 3. 原论文方法、项目借鉴与新增设计

| 来源 | 原方法解决的问题 | 本项目借鉴内容 | 不直接照搬的部分 |
|---|---|---|---|
| GOT [1] | 用高压缩视觉编码器和自回归解码器统一整页、切片、格式化及交互式 OCR | 整页输入、Vary ViT、256 个视觉 token、Qwen 解码器和原 OCR prompt | 不改变 GOT2 为检测后逐行识别器 |
| AnandaSky [2] | 面向历史汉字文献的 line-level 高分辨率转写 | 连接层预热、冻结视觉塔和小样本训练经验 | 其 line-level 输入和未压缩视觉前缀不作为本仓库整页主接口 |
| DETR [3] | 用可学习 object queries 和集合预测进行端到端目标定位 | 固定上限的区域 queries、有效位预测、bbox 辅助监督和可选匹配 | 不把独立目标检测结果作为必须的推理输入 |
| BLIP-2 Q-Former [4] | 用少量可学习 queries 从冻结视觉编码器提取信息 | query-to-visual 交叉注意力和低维瓶颈 | 不替换 GOT2 的 Qwen，也不把查询预训练任务整体照搬 |
| LayoutReader [5] | 学习文档文本块的阅读顺序 | 阅读顺序监督和排序指标 | 原方法依赖已有文本块与 bbox；本项目先从页面视觉特征生成查询 |
| LayoutLMv3、LiLT [6,7] | 建模文本、图像与二维布局 | 二维位置监督、字段 mask 和跨模态对齐思想 | 显式 bbox token 只保留为 oracle/pseudo-layout 对照 |
| LayTokenLLM [8] | 将显式 bbox 压缩为少量 layout token | 显式 bbox 压缩基线 | 不作为当前端到端主结构 |
| DocLayout-YOLO [9] | 通过多样合成数据增强文档版面分析 | 合成模板、背景和扰动多样性 | 其检测器不是本仓库的主识别结构 |

本项目新增并正在验证的候选结构暂称 **Visual Layout Query Adapter（VLQA）**：布局 queries 直接读取整页视觉特征，使用布局辅助头进行语义约束，再通过零初始化门控残差写回 GOT2 的 256 个视觉 token。该结构已有首版代码，A100 forward/backward、P1→P2 工程链路、固定 1000 steps 两页实现诊断和两页 prompt-only checkpoint 重载链路均已通过；后者使用同一 `train` split 的 P1 overfit checkpoint，只能证明工程链路，不能表述为真实页面性能或泛化提升。

## 4. 总体架构

### 4.1 原始 GOT2 路径

设第 $i$ 个页面图像为 $P_i$，页面级目标转写为 $\mathbf y_i$。原始路径为：

```text
page image -> 1024x1024 processor -> Vary ViT -> 256 visual tokens
           -> mm_projector_vary -> Qwen -> page transcription
```

记最终视觉表示为：

$$\mathbf{V}_i=\mathrm{Projector}(\mathrm{VaryViT}(P_i))\in\mathbb{R}^{N_v\times D},\qquad N_v=256,\ D=1024.$$

Vary ViT 在两次 stride-2 卷积前还存在更高分辨率的中间特征。记可供布局查询读取的特征为：

$$\mathbf{H}_i\in\mathbb{R}^{N_h\times D_h}.$$

首轮最小实现令 $\mathbf H_i=\mathbf V_i$，即直接使用现有 $16\times16$ token；只有在定位指标证明分辨率不足时，才暴露下采样前的 $64\times64$ 特征作为查询 Key/Value。高分辨率特征只供布局分支读取，不全部送入 Qwen。

### 4.2 有序布局 queries

设置 $K$ 个区域查询和一个可选全局查询：

$$\mathbf{Q}^{(0)}\in\mathbb{R}^{K\times d_a}.$$

$K$ 表示单页可建模区域数上限，由训练数据的列或版面区域分布确定，不预设为固定最优值。首轮采用**按阅读顺序编号的有序查询**：第 $k$ 个有效 query 对应真值排列 $\pi_i(k)$ 中的第 $k$ 个阅读区域；页面区域不足 $K$ 时，其余位置使用 `no-object` mask。这样把阅读顺序监督落实到 query 与区域的目标分配中，并避免无序查询的排列歧义。

令 $\mathbf{\Pi}_i$ 为继承自整页视觉网格的二维位置表示，布局读取为：

$$\mathbf{G}'_i=\mathbf{Q}^{(0)}+\mathrm{MHA}(\mathrm{LN}(\mathbf{Q}^{(0)}),\mathrm{LN}(\mathbf{H}_i+\mathbf{\Pi}_i)\mathbf{W}_K,\mathrm{LN}(\mathbf{H}_i)\mathbf{W}_V).$$

$$\mathbf{G}_i=\mathbf{G}'_i+\mathrm{FFN}(\mathrm{LN}(\mathbf{G}'_i))\in\mathbb{R}^{K\times d_a}.$$

$\mathbf G_i$ 由页面图像直接产生，不接收外部 bbox、列序或方向。由于有序槽位的 query ID 已携带顺序，不再用“从 query 预测自己的槽位编号”作为有效辅助任务；该任务可以脱离图像直接完成，不能证明模型学习了阅读顺序。若后续需要支持更复杂、无固定顺序的异构版面，可将有序槽位改为 DETR 式集合预测，再在匹配后增加独立顺序预测；该方案不是首轮最小实现。

### 4.3 训练期布局辅助头

对第 $k$ 个 query，先使用最终归一化 $\widetilde{\mathbf{G}}_i=\mathrm{LN}(\mathbf{G}_i)$ 隔离 query 主干的异常尺度，再在训练阶段预测：

$$\hat{p}^{\mathrm{obj}}_{i,k}=\mathrm{sigmoid}(\mathbf{w}_{\mathrm{obj}}^{\top}\widetilde{\mathbf{G}}_{i,k}+b_{\mathrm{obj}}).$$

$$\hat{\mathbf{b}}_{i,k}=\mathrm{sigmoid}(\mathbf{W}_{\mathrm{box}}\widetilde{\mathbf{G}}_{i,k}+\mathbf{b}_{\mathrm{box}})\in[0,1]^4.$$

$$\hat{\mathbf{p}}^{\mathrm{dir}}_{i,k}=\mathrm{softmax}(\mathbf{W}_{\mathrm{dir}}\widetilde{\mathbf{G}}_{i,k}+\mathbf{b}_{\mathrm{dir}}).$$

其中 bbox 使用归一化中心点和宽高或归一化角点，两种参数化必须在实现中固定一种；`writing_direction` 的最小类别集合为 `horizontal_ltr`、`horizontal_rtl`、`vertical_rtl`、`vertical_ltr` 和 `unknown`。阅读顺序由有序 query 的目标分配监督，不设置可由固定 query ID 平凡完成的顺序分类头。bbox、方向和顺序真值只用于训练监督及评测，不是主模型推理输入。辅助头可在部署时保留用于解释和质量控制，也可在只需 OCR 输出时丢弃。

无序 query 对照采用 DETR 式匹配后，才允许增加独立顺序预测 $\hat{o}_{i,k}=\mathrm{sigmoid}(\mathbf{w}_{\mathrm{ord}}^{\top}\mathbf{G}_{i,k}+b_{\mathrm{ord}})$；此时顺序损失必须在匹配后的区域上计算。

`col` 不必作为独立网络输入。对规则多栏页面，可由有效区域的 bbox、方向和顺序得到列位置；若复杂版面实验表明几何规则不足，再增加区域类型或列组预测头。

### 4.4 零门控写回视觉序列

布局 queries 不直接替换 GOT2 的 256 个视觉 token。以原视觉 token 为 Query、布局表示为 Key/Value，得到布局残差：

$$\mathbf{C}_i=\mathrm{MHA}(\mathrm{LN}(\mathbf{V}_i)\mathbf{W}_Q,(\widetilde{\mathbf{G}}_i+\mathbf{E}^{\mathrm{ord}})\mathbf{W}_K^{G},\widetilde{\mathbf{G}}_i\mathbf{W}_V^{G})\in\mathbb{R}^{N_v\times d_a}.$$

$$\Delta\mathbf{V}_i=\mathrm{LN}(\mathbf{C}_i)\mathbf{W}_O\in\mathbb{R}^{N_v\times D}.$$

$$\mathbf V_i^{\mathrm{layout}}=\mathbf V_i+\tanh(\alpha)\Delta\mathbf V_i.$$

$\mathbf E^{\mathrm{ord}}$ 是 query 槽位或预测顺序 embedding，$\alpha$ 初始化为 0。初始化时模型严格退化为原始 GOT2；输出仍为 $256\times1024$，不改变 `<im_start>`、`<im_end>`、image patch token 数量或 Qwen position ID。

首轮不把 $\mathbf G_i$ 额外拼接到 Qwen 上下文，以免把上下文长度变化与结构收益混在一起。若残差写回无效，再把 query 拼接方案作为后续独立消融。

### 4.5 端到端前向路径

主候选可写为：

$$\left(\mathbf{H}_i,\mathbf{V}_i\right)=\mathrm{VaryViT}(P_i),\qquad \mathbf{G}_i=\mathrm{LayoutQuery}\left(\mathbf{Q}^{(0)},\mathbf{H}_i\right).$$

$$\left(\hat{\mathbf{b}}_i,\hat{\mathbf{p}}^{\mathrm{obj}}_i,\hat{\mathbf{p}}^{\mathrm{dir}}_i\right)=\mathrm{AuxHeads}(\mathbf{G}_i),\qquad \mathbf{V}_i^{\mathrm{layout}}=\mathrm{GatedWriteback}(\mathbf{V}_i,\mathbf{G}_i).$$

$$p_{\theta}\left(\mathbf{y}_i\mid P_i\right)=\prod_{t=1}^{T_i}p_{\theta}\left(y_{i,t}\mid \mathbf{y}_{i,1:t-1},\mathbf{V}_i^{\mathrm{layout}}\right).$$

推理接口仍为“整页图像＋OCR prompt”。模型内部可以输出布局预测用于审计，但不要求用户或上游模块提供 bbox。

## 5. 目标函数

### 5.1 页面 OCR 主损失

设 $a_{i,t}\in\{0,1\}$ 表示该 token 是否属于需要监督的答案，页面 OCR 自回归交叉熵为：

$$\mathcal L_{\mathrm{ocr}}=-\frac{1}{\sum_{i,t}a_{i,t}}\sum_i\sum_{t=1}^{T_i}a_{i,t}\log p_{\theta}\left(y_{i,t}\mid \mathbf{y}_{i,1:t-1},\mathbf{V}_i^{\mathrm{layout}}\right).$$

$\mathcal L_{\mathrm{ocr}}$ 是唯一直接优化页面符号转写的主损失。bbox、方向或顺序指标不能替代页面 CER、编辑距离和完整页面精确匹配率。

### 5.2 布局查询监督

对有效区域 mask $m_{i,k}$，使用：

$$\mathcal{L}_{\mathrm{obj}}=\mathrm{BCE}(\hat{p}^{\mathrm{obj}}_{i,k},m_{i,k}).$$

$$\mathcal L_{\mathrm{box}}=\frac{1}{\max(1,\sum_{i,k}m_{i,k})}\sum_{i,k}m_{i,k}\left(\|\hat{\mathbf b}_{i,k}-\mathbf b_{i,k}\|_1+\lambda_{\mathrm{giou}}\mathcal L_{\mathrm{GIoU}}(\hat{\mathbf b}_{i,k},\mathbf b_{i,k})\right).$$

$$\mathcal L_{\mathrm{dir}}=-\frac{1}{\max(1,\sum_{i,k}m_{i,k}^{\mathrm{dir}})}\sum_{i,k}m_{i,k}^{\mathrm{dir}}\log \hat p^{\mathrm{dir}}_{i,k,d_{i,k}}.$$

主候选的阅读顺序监督体现在 $\mathbf G_{i,k}$ 与第 $k$ 个真值阅读区域的有序分配，不另设 $\mathcal L_{\mathrm{ord}}$。无序 query 对照在完成区域匹配后可使用：

$$\mathcal L_{\mathrm{ord}}=\frac{1}{\max(1,\sum_{i,k}m_{i,k}^{\mathrm{ord}})}\sum_{i,k}m_{i,k}^{\mathrm{ord}}\rho(\hat o_{i,k}-\tilde o_{i,k}),\qquad \tilde o_{i,k}=o_{i,k}/\max(1,R_i-1),$$

其中 $\rho$ 为 Smooth L1。该损失只属于无序 query 消融，不进入首轮有序 query 主配置。

### 5.3 总目标

$$\mathcal L_{\mathrm{total}}=\mathcal L_{\mathrm{ocr}}+\lambda_{\mathrm{obj}}\mathcal L_{\mathrm{obj}}+\lambda_{\mathrm{box}}\mathcal L_{\mathrm{box}}+\lambda_{\mathrm{dir}}\mathcal L_{\mathrm{dir}}.$$

所有权重均由验证集选择，并检查主损失、辅助损失和对应梯度量级。布局辅助指标良好最多证明 queries 被几何监督约束；只有页面 OCR、阅读顺序和跨模板泛化优于等参数量视觉 adaptor，才能支持布局查询对识别有效。

## 6. 训练策略

完整合成数据协议见 `SYNTHETIC_LAYOUT_TRAINING_PLAN.md`。阶段划分如下：

1. `P0` 原始整页 GOT2 基线：不增加 VLQA，固定数据划分、解码设置、页面分辨率和训练预算。
2. `P1` 布局查询预训练：使用 HTML/真实 crop 重排页面，冻结 Vary ViT 和 Qwen，残差门保持 0，只训练 queries、交叉注意力和布局辅助头。
3. `P2` 合成页面联合训练：打开零门控残差，训练 VLQA、辅助头和 `mm_projector_vary`，以页面 OCR 损失为主；Vary ViT 与 Qwen 先冻结。
4. `P3` 真实页面小样本适配：混合真实整页与少量合成页面，保持视觉塔冻结，使用 Qwen LoRA 或低学习率 decoder；LoRA 是训练手段，不作为结构创新。
5. `P4` 分辨率扩展：只有当 $16\times16$ query 定位不足且错误分析指向空间分辨率时，才让 queries 读取 $64\times64$ 中间特征或接入 GOT2 原生 multi-crop。

动态合成数据统一用 optimizer steps、有效 batch size、唯一内容曝光次数和随机种子控制预算，不以 epoch 作为唯一比较口径。筛选阶段可先使用一个种子；正式比较至少使用三个种子。

## 7. 轻量化设计

1. Vary ViT 主体默认冻结，优先训练少量 queries、交叉注意力、辅助头和连接器。
2. 高分辨率中间特征只作为布局 queries 的 Key/Value，不拼接进 Qwen，避免上下文与显存随 $N_h$ 增长。
3. 布局残差写回后仍保持 256 个 image token，维持原 Qwen 接口。
4. query 数量 $K$、瓶颈维度 $d_a$ 和 FFN 扩展率必须做小规模消融；参数量、峰值显存、吞吐量和单页延迟以实现后实测为准。
5. 若完整 bbox、方向和顺序监督不优于更简单的方向或全局页面查询，优先保留简单结构，不扩大模块。

## 8. 数据与标签口径

每个正式样本以页面为单位，至少包含：

```json
{
  "image": "relative/page/path.png",
  "page_id": "page_xxx",
  "source_group_id": "book_or_collection_xxx",
  "page_text": "按真值阅读顺序拼接的页面转写",
  "regions": [
    {
      "region_id": "region_000",
      "bbox": [x0, y0, x1, y1],
      "reading_order": 0,
      "writing_direction": "vertical_rtl",
      "text": "可选的区域转写"
    }
  ]
}
```

`bbox`、`reading_order` 和 `writing_direction` 是训练标签与评测依据，不是推理输入。真实页面若只有页面转写而无区域框，仍可用于 OCR loss；对应布局辅助项通过 mask 跳过。现有 line-level 转写可以按已知页面归属和阅读顺序重组成合成页面标签，但不得让同一内容或同一原页面跨 split。

训练、验证和测试必须按书手、版本、馆藏、来源文档或符号类型隔离。页面模板、背景和内容独立采样，防止固定文字、书籍或符号类别与固定列位绑定。

## 9. 实验协议

### 9.1 必须比较的模型

| 编号 | 模型 | 目的 |
|---|---|---|
| `A0` | `got2_zero_shot` | 原始 GOT2 零样本参考 |
| `A1` | `projector_only` | 合成域与 projector 适配贡献 |
| `A2` | `generic_adapter_projector` | 等参数普通 adaptor 容量对照 |
| `A3` | `vlqa_ocr_only` | VLQA 查询和写回结构贡献 |
| `A4` | `vlqa_layout_direct` | 不经 P1 的布局辅助监督贡献 |
| `A5` | `vlqa_layout_p1_p2` | P1 预热贡献，单独报告 P1/P2 与总曝光量 |

`A1`–`A4` 必须使用相同原始 checkpoint、整页输入、数据划分、页面转写、解码设置、有效 batch size 和 P2 预算。A5 的 P2 与 A4 相同，但必须额外报告 P1 steps 与页面曝光量。旧显式字段方案仍可作独立扩展对照，不再占用本轮编号。

独立双 GOT2 系统只有在输入页面、数据划分、训练预算和页面级输出指标一致时才能比较最终结果；不得用其单列 CER 直接对比本仓库路线的页面 CER。

### 9.2 评测指标

- 页面级 CER、编辑距离、完整页面精确匹配率；
- 区域或列级 CER，仅作为错误定位诊断；
- 稀有符号 K-shot 召回率；
- bbox IoU/mAP 或区域召回率；
- 书写方向准确率；
- 阅读顺序成对准确率和 Kendall's $\tau$；
- 跨书手、跨版本、跨馆藏、跨符号类型和跨页面模板泛化；
- 可训练参数量、峰值显存、吞吐量和单页延迟。

小样本继续拆成两类：领域级少样本限制新书手、新版本、新馆藏或新场景的标注页面数/比例；稀有符号级 K-shot 限制每个低频符号的训练实例数。所有结果必须明确页面粒度、划分单位、训练步数、输入分辨率和是否使用区域级辅助标签。

### 9.3 关键消融与反事实检查

1. 将 VLQA 残差门设为 0，检查是否退化为同一 GOT2 基线。
2. 去除全部布局辅助损失，形成 `A2`。
3. 分别比较 bbox-only、bbox＋direction，以及在相同区域辅助监督下的有序与无序 query 目标分配；不设置没有区域损失支撑的 `order-only` 主配置。
4. 打乱 query 的顺序 embedding，检查页面输出顺序是否相应恶化。
5. 将有序 queries 改为无序 queries，检查显式顺序约束的贡献。
6. 比较 $16\times16$ 与 $64\times64$ query Key/Value，但保持 Qwen 的 256 token 不变。
7. 在未见模板和未见内容上可视化 query attention，检查是否只记忆固定列位。

## 10. 与当前代码的接入位置

- 整页图像处理和 256 token 接口：`src/GOT-OCR-2.0/GOT/model/GOT_ocr_2_0.py`；
- Vary ViT 的二维位置和中间特征：`src/GOT-OCR-2.0/GOT/model/vision_encoder/vary_b.py`；
- VLQA、辅助头、零门控写回和布局损失：`src/GOT-OCR-2.0/GOT/model/layout_query.py`；
- 页面级 dataset/collator：`src/GOT-OCR-2.0/scripts/layout_page_dataset.py`；
- `P1/P2` 开发训练入口：`src/GOT-OCR-2.0/scripts/train_GOT_layout.py`；
- A100 受限 smoke 编排与 checkpoint 验证：`tools/training/run_layout_a100.py`、`src/GOT-OCR-2.0/scripts/verify_layout_checkpoint.py`；
- HTML 页面生成和 split 审计：`tools/preprocessing/generate_synthetic_layout.py`、`tools/preprocessing/audit_synthetic_layout.py`；
- 现有整页兼容数据参考：`src/GOT-OCR-2.0/scripts/ancientdoc_dataset.py`；
- 现有标准整页 demo：`src/GOT-OCR-2.0/GOT/demo/run_ocr_2.0.py`。

上述代码保持权重、数据、checkpoint 和日志位于源码树外。生成器、DOM bbox 和 manifest 已通过本机真实浏览器 smoke。A100 run `layout_pilot_20260811_023528` 已完成 VLQA CUDA 反向传播、整页 batch、P1、P1→P2 加载、checkpoint 保存与完整模型重载；但它只有 2 页、每阶段 500 epoch 且没有 validation，只能确认工程链路。

A100 run `layout_overfit_20260811_110317` 随后执行了 P1 两样本 200 steps 诊断，结果为 fail。object/direction logits 在第 1 步即达到约 1680，bbox 同时饱和到约 0/1；末 20 步 bbox mean IoU 仍为 0、object accuracy 仅为 0.5375。query 梯度存在，因此不是简单的监督断链。问题由此定位到原始 GOT2 checkpoint 缺少 VLQA 权重时的加载后初始化与预测尺度路径。修复版在加载后显式完整初始化新 VLQA，对 safetensors 中的布局键执行“全无、全有或拒绝部分状态”的严格审计，在辅助头前增加最终 LayerNorm，并使用小尺度 bbox 输出层初始化。

A100 run `layout_overfit_20260811_113817` 已验证该修复组合：完整模型报告 `fresh_explicit_reset`、`0/45` 个源/预期布局张量和 1.0 的参数绝对值上限；首步 object、direction 和 bbox raw logits 分别仅为 0.1592、0.3984 和 0.0010，bbox 初值为 0.25–0.75。200 步后 object loss/accuracy 与 direction loss/accuracy 均通过阈值；bbox L1 从 0.9761 降至尾段均值 0.1159，mean IoU 从 0.0726 升至 0.3496，表明 bbox 监督和梯度有效但步数不足。

A100 run `layout_overfit_20260812_002747` 已完成固定 P1、2 条记录、1000 steps 的实现诊断并返回 `overfit_assessment.status=pass`。末 20 步均值为 object loss `0.00201755`、bbox L1 `0.00531464`、bbox GIoU `0.05508578`、direction loss `0.00154159`、object/direction accuracy `1.0`、bbox mean IoU `0.94497279`，bbox 尾段范围也通过阈值。该 run 仍只是实现诊断，不是 validation 或性能结果。

建议实现顺序：

1. 页面 manifest 与 HTML 合成器：首版完成，本地浏览器 smoke 通过；
2. 页面级 dataset/collator、CPU JSON 预检和 A100 batch：完成；
3. 基于现有 $16\times16$ token 的最小 VLQA 与辅助头 forward/backward：完成；
4. `P1→P2` 保存、加载和完整模型重载：完成；
5. FP32 分项日志与 P1 两样本 `overfit`：完成；
6. 显式完整初始化、checkpoint 键审计、预测前归一化与 raw-logit 诊断：A100 验证通过；
7. 固定 1000 steps 的 bbox 收敛检查：A100 已通过；
8. validation loader、prompt-only evaluator 和页面级指标：本地已实现；`layout_validate_20260812_014816` 已完成两页链路验证，正式 held-out split 待执行；
9. `A0/A1/A2/A4` 单步 smoke；
10. 仅在定位失败后暴露 $64\times64$ 中间特征。

## 11. 风险与停止条件

1. 原始 GOT2 已能隐式处理整页布局，新增 queries 可能没有额外收益；必须以 `A0/A1/A2` 排除容量和重采样影响。
2. 布局指标良好但页面 OCR 和阅读顺序无改善，说明辅助监督没有有效服务识别，应停止扩大 query 模块。
3. `A4` 与 `A2` 无稳定差异，不能声称 bbox、方向或顺序监督有效。
4. 收益只存在于已见 HTML 模板，在真实页面或未见模板上消失，应判定为合成域差异或位置捷径。
5. 最终 $16\times16$ token 可能不足以定位密集细列；只有错误分析支持该判断时才接入 $64\times64$ 特征，不能直接用更大模块掩盖目标设计问题。
6. 有序 query 的固定槽位可能不适合复杂异构页面；若 `no-object`、漏检或顺序冲突频繁，再评估集合匹配，不提前增加复杂度。
7. 页面转写长度可能超过 Qwen 上下文或造成训练截断；必须报告截断比例，并将超长页面、multi-crop 与普通页面分层评估。
8. 若完整 VLQA 在同预算下不优于等参数量视觉 adaptor，应保留简单 adaptor 或原 GOT2，不把无效结构包装为布局创新。
9. 若 P1 不能在两个固定模板页面上把 object/direction 分类和 bbox 定位拟合到预设阈值，应先修复数值、初始化、冻结范围或目标实现，不得扩大数据或启动 P2。

## 12. 当前结论

当前主候选已从“line crop＋显式 bbox/列序输入的 region-token adapter（旧 PCLA）”修订为“整页 GOT2＋端到端 Visual Layout Query Adapter”。页面视觉 token 提供原页面坐标参考，learnable queries 从视觉特征中产生布局表示；bbox、方向和阅读顺序只作为训练期辅助监督或评测标签，推理时不要求外部 metadata。显式 region-token adapter（旧 PCLA）、外部检测器路线和双 GOT2 路线均只作为独立对照。

该方案已完成设计修订、首版工程链路、加载后初始化修复、1000 steps 两页实现诊断和两页 prompt-only checkpoint 重载验证。后一次验证使用同一 `train` split 的 P1 overfit checkpoint，不能替代正式 held-out validation。能否提高页面 OCR、阅读顺序和小样本跨域泛化，仍必须由正式 split 和新版统一预算 `A0`–`A5` 消融决定。

## 参考文献

1. Wei, H. et al. *General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model*. arXiv:2409.01704, 2024.
2. Brisson, C., Kahfy, A., Constant, F., & Bui, M. *AnandaSky: A Vision-Language Model for Line-Level Transcription of Historical Sinographic Documents*. LT4HALA, 2026.
3. Carion, N. et al. “End-to-End Object Detection with Transformers.” *ECCV*, 2020. DOI: [10.1007/978-3-030-58452-8_13](https://doi.org/10.1007/978-3-030-58452-8_13)。
4. Li, J. et al. “BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.” *ICML*, 2023. [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)。
5. Wang, Z. et al. “LayoutReader: Pre-training of Text and Layout for Reading Order Detection.” *EMNLP*, 2021. DOI: [10.18653/v1/2021.emnlp-main.389](https://doi.org/10.18653/v1/2021.emnlp-main.389)。
6. Huang, Y. et al. “LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking.” *ACM Multimedia*, 2022. DOI: [10.1145/3503161.3548112](https://doi.org/10.1145/3503161.3548112)。
7. Wang, J., Jin, L., & Ding, K. “LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding.” *ACL*, 2022. DOI: [10.18653/v1/2022.acl-long.534](https://doi.org/10.18653/v1/2022.acl-long.534)。
8. Zhu, Z. et al. “A Simple yet Effective Layout Token in Large Language Models for Document Understanding.” *CVPR*, 2025. DOI: [10.1109/CVPR52734.2025.01349](https://doi.org/10.1109/CVPR52734.2025.01349)。
9. Wang, D. et al. “DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception.” arXiv:2410.12628, 2024. [代码](https://github.com/opendatalab/DocLayout-YOLO)。
