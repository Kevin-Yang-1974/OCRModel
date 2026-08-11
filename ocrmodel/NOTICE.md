# 来源与第三方材料说明

## GOT-OCR2.0

`src/GOT-OCR-2.0` 保留 GOT-OCR2.0 的包结构，并在此基础上加入本项目代码。上游项目与论文：

- Code: <https://github.com/Ucas-HaoranWei/GOT-OCR2.0>
- Paper: H. Wei et al., “General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model,” arXiv:2409.01704, 2024.

`src/GOT-OCR-2.0/pyproject.toml` 将上游包标记为 Apache Software License。当前共享目录没有附带完整的仓库级许可证文本，因此对外公开或再分发前应核对上游仓库许可证，并由项目负责人确定新增代码的许可。

本项目相对当前上游布局的主要新增或改动包括：

- `GOT/model/layout_query.py` 中的 Visual Layout Query Adapter；
- GOT2 视觉 token 与布局模块的接线；
- 页面级 layout dataset、P1/P2 训练及 checkpoint 审计；
- HTML/Playwright 整页合成、数据审计和受限实验编排工具。

## AnandaSky

仓库只提供面向外部 AnandaSky 安装的环境与诊断包装器，不包含 AnandaSky 模型权重。论文：C. Brisson, A. Kahfy, F. Constant, and M. Bui, “AnandaSky: A Vision-Language Model for Line-Level Transcription of Historical Sinographic Documents,” LT4HALA, 2026。

## 历史 AncientDoc 兼容评估

`references/legacy-ancientdoc-eval` 保存项目早期共享环境中的四个评估脚本快照，用于核对历史 AncientDoc split5 解码和指标口径。它们不是当前活动源码，可能包含旧路径、旧参数和历史注释；不得直接作为正式整页 VLQA 入口。活动包装器位于 `tools/evaluation`。

模型权重、数据集和历史完整预测均未包含在共享仓库中，其访问和使用仍受各自来源条款约束。
