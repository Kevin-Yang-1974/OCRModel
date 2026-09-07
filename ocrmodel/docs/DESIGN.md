# 设计说明

## 核心瓶颈

整页符号识别既需要保留局部形态，又需要建模区域、阅读顺序和书写方向。小样本适配时，若仅依赖文本生成损失，稀疏布局信号容易被高频内容信号覆盖；若把标注布局作为推理输入，则破坏整页端到端协议。

## 总体架构

GLM-OCR 的视觉编码器先生成整页 visual tokens。`PreMergeLayoutAdapter` 使用可学习 seed queries 对这些 tokens 做交叉 attention，得到布局 queries；随后预测区域、顺序和方向，并在视觉内容进入原合并器前回写布局上下文。布局真值不出现在 `forward` 接口中。

## 关键模块

当前确认实验优先比较 `attention` 与 `geometry`。`geometry` 在内容相似度中加入预测区域—patch 距离；`layout_ot` 保留为后续探索对照。半松弛传输固定每个 query 的质量，但只通过 KL 近端更新鼓励 token 边缘接近均匀覆盖，允许不同区域占用不同数量的视觉 tokens。该实现仍是项目候选设计，不是已验证结论。

三个对照共用 query 生成器与预测头：`content_only` 不回写布局，`attention` 只使用内容 score，`geometry` 使用相同几何 score 但以 softmax 取代 OT。这样可分离 query 参数、几何偏置和运输约束的贡献。

## 目标函数

主任务使用 GLM-OCR 原文本生成目标。本模块新增 bbox Smooth L1、顺序回归、方向分类和 token-region assignment NLL。数据层需先完成 target-query matching；无标注 query 通过 mask 忽略。传输熵仅作可选正则项，默认权重为零。

## 训练策略

首轮三种子结果显示 attention/geometry 均在 step 256 达到最低 validation CER，后续虽有 OCR loss 下降，但残差 gate 与生成触顶率持续增加。该现象当前解释为小样本反复训练下的残差扰动累积与解码漂移，不是 NaN 型故障；这一解释仍需本轮稳定性实验验证。

稳定性确认保持主干、原 patch merger、输入协议和 `auxiliary_weight=0.2` 不变，只训练 pre-merger adapter。学习率先用 64 steps 线性 warmup 至 `5e-5`，再 cosine 衰减，并在 step 1024 到达 `5e-6`。原始 `content_gate` 仍作为 checkpoint 参数保存，实际回写系数采用 `clamp(tanh(content_gate), -0.03, 0.03)`；未配置上限的旧 checkpoint 继续使用原始 `tanh` 语义。受控小残差尺度的设计依据来自 ReZero 的零初始化残差思想和 CaiT LayerScale 的小尺度残差注入，但 `0.03` 是依据本项目首轮最佳 checkpoint 的观察值设定，属于项目修改而非两篇论文的原始超参数[4-5]。

同一 128 页 train、64 页 validation 上运行 attention/geometry × seeds `42/43/44` 共六组，每组 1024 steps；另运行一次不更新参数的严格 prompt-only `content_only` 基线。以三种子平均 validation CER 选择“模式＋step”，不读取 64 页 test。只有稳定性阈值全部满足后，候选才可进入后续 selection-locked test。

## 轻量化边界

首轮仅训练 adapter、query seeds 和辅助头，以减少小样本过拟合与存储开销。这是训练范围控制，不单独作为结构创新。后续是否解冻视觉高层或合并器，由统一消融结果决定。

## 接入与风险

预期接入点是 GLM-OCR 视觉 tokens 进入视觉—语言合并器之前。具体 tensor 契约必须在锁定上游版本和 checkpoint 后核对，当前原型不声称已与实际 checkpoint 兼容。主要失败风险是 query collapse、OT 数值敏感、辅助监督压制识别目标，以及不同对照计算量不对等。

## 方法来源边界

直接采用部分：GLM-OCR 官方 checkpoint `ca5d8b3e287e52589e37c28385d9655ee4372f9d`、CogViT 视觉编码器、空间下采样、`GlmOcrVisionPatchMerger` 和 GLM-0.5B decoder。官方 Transformers 实现明确在视觉 block 后先下采到 1536 维 token，再调用 patch merger[1-3]。

结合项目修改部分：`LayoutAwarePatchMerger` 保留官方 merger，但在其前对下采样 token 执行布局融合。数据使用整页 `Text Recognition:` prompt，不使用官方 SDK 的 PP-DocLayout-V3 裁剪两阶段推理。

项目新增部分：整页布局 queries、布局条件化半松弛 OT、三类结构对照与布局辅助目标。这些是待实验候选，不归因为 GLM-OCR 原论文方法。

## 参考文献

[1] Duan, S., Xue, Y., Wang, W., et al. GLM-OCR Technical Report. arXiv:2603.10910, 2026. https://arxiv.org/abs/2603.10910

[2] zai-org. GLM-OCR model card, revision `ca5d8b3e287e52589e37c28385d9655ee4372f9d`. https://huggingface.co/zai-org/GLM-OCR

[3] Hugging Face. `modeling_glm_ocr.py`, Transformers 5.3 series. https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/glm_ocr/modeling_glm_ocr.py

[4] Bachlechner, T., Majumder, B. P., Mao, H., et al. ReZero is All You Need: Fast Convergence at Large Depth. arXiv:2003.04887, 2020. https://arxiv.org/abs/2003.04887

[5] Touvron, H., Cord, M., Sablayrolles, A., et al. Going Deeper With Image Transformers. ICCV, 2021. https://openaccess.thecvf.com/content/ICCV2021/html/Touvron_Going_Deeper_With_Image_Transformers_ICCV_2021_paper.html
