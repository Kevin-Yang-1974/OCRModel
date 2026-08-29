# 来源与第三方材料说明

## GOT-OCR2.0

`src/GOT-OCR-2.0` 保留 GOT-OCR2.0 的包结构，并在此基础上加入本项目代码。上游项目与论文：

- Code: <https://github.com/Ucas-HaoranWei/GOT-OCR2.0>
- Paper: H. Wei et al., “General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model,” arXiv:2409.01704, 2024.

`src/GOT-OCR-2.0/pyproject.toml` 将上游包标记为 Apache Software License。当前共享目录没有附带完整的仓库级许可证文本，因此对外公开或再分发前应核对上游仓库许可证，并由项目负责人确定新增代码的许可。

本项目相对当前上游布局的主要新增或改动包括：

- `GOT/model/layout_prompt_decoder.py` 中的 LAVP/PVLD 布局重建模块；
- `GOT/model/layout_query.py` 中保留的历史 Fixed-Slot Visual Layout Query Adapter/VQLCA；
- GOT2 视觉 token 与布局模块的接线；
- 页面级 layout dataset、P1/P2 训练及 checkpoint 审计；
- HTML/Playwright 整页合成、数据审计和受限实验编排工具。

## AnandaSky

仓库只提供面向外部 AnandaSky 安装的环境与诊断包装器，不包含 AnandaSky 模型权重。论文：C. Brisson, A. Kahfy, F. Constant, and M. Bui, “AnandaSky: A Vision-Language Model for Line-Level Transcription of Historical Sinographic Documents,” LT4HALA, 2026。

## 历史 AncientDoc 兼容评估

历史 AncientDoc、VLQA/VQLCA、Chunk 和 SOTA 评估脚本不属于当前共享主线；整理前版本保存在远程 `archive/legacy-vlqa-chunk-20260829` 分支，仅用于历史结果复现和口径核对，不得直接作为当前 PVLD 入口。

模型权重、数据集和历史完整预测均未包含在共享仓库中，其访问和使用仍受各自来源条款约束。
