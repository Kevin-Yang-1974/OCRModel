# GLM-OCR 整页布局适配分支

本工程面向小样本条件下的多场景通用符号识别。当前代码是在 GLM-OCR 视觉内容合并之前，由整页视觉 token 生成布局 queries 的独立机制原型。确认实验优先比较 geometry 与 attention，并开关布局辅助监督。

- `content_only`：不做布局融合的核心对照。
- `attention`：仅使用内容相似度的普通 attention 对照。
- `geometry`：在 attention 中加入模型预测区域与视觉 patch 网格位置的几何偏置。
- `layout_ot`：使用固定 query 质量、松弛 token 边缘的布局条件化熵正则最优传输。

推理接口不接收布局真值。`patch_positions` 是视觉编码器内部的归一化 patch 网格坐标，不是 bbox 标注。bbox、顺序、方向和 token-region 对应仅进入辅助损失。

已实现：独立 Python 包、四种模式的前向/反向通路、辅助目标、R1/R2 split 泄漏审计、128 页 seed 42 机制筛选，以及三种子 attention/geometry 首轮确认实验。首轮结果在 256 steps 后一致退化，并伴随残差 gate 增大和生成触顶率上升。

正在实施：稳定性确认保持 `auxiliary_weight=0.2`，比较 attention/geometry 各三种子；使用 64-step warmup、峰值 `5e-5`、cosine 衰减至 `5e-6`，并将有效残差 scale 限制在 `±0.03`。另设一次不更新参数的严格 prompt-only `content_only` 基线。低频字符召回目前只是训练频次诊断，不等同于受控 R2 K-shot。

待验证：上述约束能否消除后程 CER 与生成触顶率退化，以及 geometry 是否在三种子同预算下稳定优于 attention；半松弛 OT 和受控 R2 K-shot 留待后续阶段。

本地检查和 BSCC 入口见 `docs/RUN.md`；设计边界见 `docs/DESIGN.md`；实验状态见 `docs/EXPERIMENT_REGISTER.md`。
