# 协作说明

## 研究口径

1. 当前研究对象是多场景、小样本条件下的通用符号识别。古籍、谱面和复杂文献是重点验证场景，不是能力边界。
2. 正式主协议使用原始整页图像。bbox、阅读顺序和书写方向只作为训练监督、可选解释性输出或评测真值。
3. 当前主候选是单个 GOT2 内部的 VLQA。双 GOT2 两阶段系统、AnandaSky line-level 模型和 AncientDoc 旧 split 均为独立对照或兼容诊断。
4. 不把尚未验证的实现写成性能提升。P1 两样本 overfit 和同页 prompt-only validation 链路已通过；当前应先建立隔离的正式 split 和原 GOT2 baseline，再进行 held-out validation 与统一消融。

## 修改范围

- 活动 GOT2 源码只位于 `src/GOT-OCR-2.0`。
- 通用运行工具放在 `tools`，不要把一次性 run 命令写入活动入口。
- 机器路径只写入未提交的 `config/paths.env` 或命令行参数。
- 权重、数据、checkpoint、完整预测和日志必须放在仓库外。
- `references/legacy-ancientdoc-eval` 是历史兼容评估的只读快照；如需修正，应在 `tools/evaluation` 新增或修改包装器。

## 变更要求

每个模型或训练变更应说明：

1. 要解决的具体瓶颈；
2. 接入位置和所需监督；
3. 相对原 GOT2 的新增参数与计算；
4. 对应的对照和消融；
5. 失败风险与可观察诊断。

常规数据增强、LoRA、训练技巧或简单 MLP 替换不能单独表述为结构创新。结构与损失变更应能局部接入现有 GOT2 baseline，并可独立消融。

## 本地检查

从仓库根目录运行：

```powershell
py -3 -m compileall -q src tools
py -3 -m pytest -q tests
```

本地没有 PyTorch 时，依赖 PyTorch 的测试可以跳过；必须在变更说明中明确记录。涉及 GPU 的变更还需先通过受限 `smoke` 或 `overfit`，但不得将其表述为 validation。

## 实验记录

正式实验至少记录以下字段：

- Git commit 与本地修改状态；
- 模型、数据 manifest 和配置哈希；
- 页面输入粒度、图像尺寸和 prompt；
- split 单位及泄漏审计结果；
- seed、有效 batch size、训练步数和可训练模块；
- 参数量、显存、吞吐或单页延迟；
- 页面 OCR、布局和少样本指标；
- 失败状态及最后少量相关错误，不提交完整日志。

运行目录和 `summary.json` schema 应保持向后可读。改变字段名或输出位置时，应同步更新测试、`docs/SYNC_AND_RUN.md` 和 `docs/PROJECT_STATUS.md`。

## Git

提交只包含同一项任务相关文件。不要提交本机配置、密钥、远端地址、数据或运行产物，也不要改写已有历史。共享远程只发布本目录内容；父项目中的申报材料、文献和个人记录不属于本仓库。
