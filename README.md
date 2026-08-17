# 小样本通用符号识别

本仓库实现面向多场景、小样本条件的通用符号识别实验。当前主路线以 GOT2 原生整页 OCR 为基础，在单个 GOT2 内加入端到端 Visual Layout Query Adapter（VLQA），从整页视觉特征生成区域查询，并以区域位置、阅读顺序和书写方向作为训练期辅助监督。

正式训练、验证和推理只输入原始整页图像与 OCR prompt。bbox、阅读顺序和书写方向不是主模型推理输入。古籍、谱面、公式、表格和复杂文献是验证场景，不限定模型的适用范围。

## 当前状态

更新于 2026 年 8 月 15 日。

| 模块 | 状态 | 当前结论 |
|---|---|---|
| 整页 HTML/Playwright 合成与 manifest 审计 | 已实现并通过本地 smoke | 可生成 `S0-html-text`、`S1-html-crop` 和 `S2-hard` schema v2 数据；正式字体包仍未锁定 |
| 页面级 dataset/collator 与 VLQA | 已实现首版 | 支持有序 queries、object/bbox/direction 辅助损失和零门控写回 |
| A100 forward/backward 与 P1→P2 保存重载 | 工程链路已打通 | 两样本 overfit、checkpoint 重载和 prompt-only validation 均已通过；这些运行仍只作为工程证据 |
| 正式合成 P1/P2 | 已完成 P1 4000/P2 8000 | P2 8000 validation 页面 CER `0.139487`、布局 F1 `0.762183`；属于合成数据结果 |
| 原始 GOT2 与 P2 8000 VLQA 对照 | 已完成 300 页合成 test | VLQA 页面 CER 相对 baseline 下降 `0.260725`，布局 F1 `0.787056`；结构归因仍需消融 |
| AncientDoc C0/C1/C4/C5/C6 对照 | C1 已完成，C4 在途，replay 待 C4-best | C4 周期 checkpoints 必须先在 validation 选 best；C5/C6 从同一 C4-best 独立启动，旧 C4-final 自动分支已禁用 |
| 等参数量与无布局监督结构对照 | 尚未实现 | 未完成前不能把 P2/C6 收益单独归因于 VLQA 布局结构 |

当前合成数据主 checkpoint 为 `layout_joint-train_8000_20260813/p2/model`。它已经完成同协议合成 validation/test，但等参数量 adaptor、无布局监督 queries 和完整 loss 消融仍未完成。AncientDoc 旧实验中的原始 GOT2 现明确记为 C0 zero-shot reference，不再作为同预算训练 baseline；C1 原生 GOT2 整页 OCR-only 适配已经实现。旧 `seed20260814` group split 因比例失衡已废弃，正式重跑使用新 allocator 生成的 `seed20260815` split。完整状态见 [PROJECT_STATUS.md](docs/PROJECT_STATUS.md)，命令见 [SYNC_AND_RUN.md](docs/SYNC_AND_RUN.md)，配置和公平性边界见 [训练入口说明](tools/training/README.md)。

## 目录

```text
.
├─ config/                         可提交的配置模板；实际路径配置不提交
├─ docs/                           状态、架构、实验协议、依赖和运行手册
├─ references/
│  └─ legacy-ancientdoc-eval/      历史兼容评估的只读参考快照
├─ src/GOT-OCR-2.0/               唯一活动 GOT2 源码树
├─ tests/                          CPU 单元测试与编排器纯函数测试
└─ tools/
   ├─ environment/                 环境检查、锁文件和受限运行包装器
   ├─ evaluation/                  评估工具；历史兼容入口不属于正式协议
   ├─ inference/                   GOT2/AnandaSky 诊断推理工具
   ├─ preprocessing/               整页合成、审计和兼容预处理
   ├─ sync/                        参数化的代码同步工具
   └─ training/                    VLQA A100 诊断与训练编排器
```

模型权重、数据集、checkpoint、完整预测、日志和 run 目录必须位于仓库外。个人迁移记录、废弃 SSH 脚本和历史结果也不属于共享源码。

## 快速检查

以下命令均从本仓库根目录执行。

只做语法检查不需要 CUDA、PyTorch 或模型权重：

```powershell
py -3 -m compileall -q src tools
```

安装本地 CPU 工具依赖并运行测试：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r tools\environment\requirements-local-dev.txt
.\.venv\Scripts\python.exe -m pytest -q tests
```

页面合成的安装、schema 和审计命令见 [整页布局合成工具](tools/preprocessing/README.md)。服务器环境、同步和受限实验命令见 [SYNC_AND_RUN.md](docs/SYNC_AND_RUN.md)。

AncientDoc 日常训练使用 `prepare → train-core → select-c4 → train-replay`。C4-best 只由 validation 决定并作为 C5/C6 的共同分支点；随后 C1/C5/C6 再在 validation 选优，冻结后 test 只评估一次。

## 实验边界

| 入口 | 用途 | 能否作为当前主结果 |
|---|---|---|
| `tools/training/run_layout_a100.py --mode overfit` | 两样本实现正确性检查 | 否 |
| `tools/training/run_layout_a100.py --mode smoke` | 环境、单步训练与 checkpoint 链路检查 | 否 |
| `tools/training/run_layout_a100.py --mode validate` | 整页 prompt-only 推理与统一页面指标 | 可用于明确 checkpoint 和 split 的评估；结论范围由数据隔离和对照完整性决定 |
| `tools/training/run_layout_a100.py --mode pretrain` | 原始 GOT2 起点的正式 P1 布局预训练 | 已完成合成数据 P1 4000；P1 不用于 OCR 改善结论 |
| `tools/training/run_layout_a100.py --mode joint-train` | P1 checkpoint 起点的正式 P2 OCR＋布局联合训练 | 已完成合成数据 P2 8000；仍需结构消融和真实分组隔离评估 |
| `tools/evaluation/compare_got2_vlqa.py` | 同一整页 test manifest 上比较原始 GOT2 与 P2 VLQA | 已完成 P2 8000 合成 test 对照 |
| `tools/training/run_ancientdoc.sh train-core|select-c4|train-replay` | AncientDoc 分阶段适配 | C1/C4 训练、C4 分支点选择、同起点 C5/C6 replay；不自动 test |
| `tools/evaluation/run_ancientdoc.sh` | C0/C1/C4/C5/C6 validation 选优与冻结 test | 单模型 batch 推理、动态多卡队列；输出 `selection.json`、`summary.json` 与 `report.md` |
| `src/GOT-OCR-2.0/scripts/run_linelevel_smoke.sh` | 既有单行/单列工程诊断 | 否 |
| AncientDoc 旧 split 训练与兼容评估 | 历史页面兼容基线 | 否；存在书籍级重叠 |
| 统一整页划分下的 `A0`–`A5` 消融 | 正式结构归因实验 | A0 zero-shot、A1 projector-only、A2 等参数量 generic adaptor、A3 VLQA OCR-only、A4 direct layout、A5 P1→P2 已有统一训练、validation 选点和 selection-locked test 入口；正式效果仍待完整运行 |

AnandaSky 是 line-level 高分辨率转写基线。双 GOT2 的“先分割、再识别”是独立两阶段系统对照。两者只有在页面输入、数据划分、训练预算和页面级指标一致时才可与本仓库路线比较。

## 文档

- [PROJECT_STATUS.md](docs/PROJECT_STATUS.md)：已实现、待验证和下一步。
- [tools/training/README.md](tools/training/README.md)：C0/C1/C4/C5/C6 的起始 checkpoint、数据组成、预算、公平性边界和分支关系。
- [ANCIENTDOC_BASELINE_REPORT_20260814.md](docs/ANCIENTDOC_BASELINE_REPORT_20260814.md)：首次 2000-step 原始 split 指标、解释和协议限制。
- [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)：正式输入、少样本划分、消融和指标。
- [GOT2_LAYOUT_ENCODING_PLAN.md](docs/GOT2_LAYOUT_ENCODING_PLAN.md)：VLQA 架构、目标函数和接入位置。
- [SYNTHETIC_LAYOUT_TRAINING_PLAN.md](docs/SYNTHETIC_LAYOUT_TRAINING_PLAN.md)：整页合成与分阶段训练方案。
- [DEPENDENCIES.md](docs/DEPENDENCIES.md)：本地与服务器依赖边界。
- [SYNC_AND_RUN.md](docs/SYNC_AND_RUN.md)：参数化同步、环境检查和实验命令。
- [PUBLISHING.md](docs/PUBLISHING.md)：从父仓库安全发布仅含本目录的共享分支。
- [CONTRIBUTING.md](CONTRIBUTING.md)：协作与提交要求。
- [NOTICE.md](NOTICE.md)：上游代码和历史参考材料说明。

## 来源与许可

`src/GOT-OCR-2.0` 基于 GOT-OCR2.0 上游源码，并包含本项目的 VLQA、页面数据集和训练改动。上游与参考材料的来源见 [NOTICE.md](NOTICE.md)。仓库尚未单独声明覆盖全部项目新增内容的分发许可；对外公开或再分发前应由项目负责人确认许可范围。
整页布局结构消融的统一训练、validation 选点和 frozen test 入口见 `tools/training/run_layout_ablation_suite.sh`，运行命令见 `docs/SYNC_AND_RUN.md`。
