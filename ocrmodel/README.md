# 小样本通用符号识别

本仓库实现面向多场景、小样本条件的通用符号识别实验。当前主路线以 GOT2 原生整页 OCR 为基础，在单个 GOT2 内加入端到端 Visual Layout Query Adapter（VLQA），从整页视觉特征生成区域查询，并以区域位置、阅读顺序和书写方向作为训练期辅助监督。

正式训练、验证和推理只输入原始整页图像与 OCR prompt。bbox、阅读顺序和书写方向不是主模型推理输入。古籍、谱面、公式、表格和复杂文献是验证场景，不限定模型的适用范围。

## 当前状态

更新于 2026 年 8 月 12 日。

| 模块 | 状态 | 当前结论 |
|---|---|---|
| 整页 HTML/Playwright 合成与 manifest 审计 | 已实现并通过本地 smoke | 可生成 `S0-html-text`、`S1-html-crop` 和 `S2-hard` schema v2 数据；正式字体包仍未锁定 |
| 页面级 dataset/collator 与 VLQA | 已实现首版 | 支持有序 queries、object/bbox/direction 辅助损失和零门控写回 |
| A100 forward/backward 与 P1→P2 保存重载 | 工程链路已打通 | 只证明代码链路可运行，不构成性能证据 |
| P1 两样本 overfit | 1000 steps 实现诊断通过 | `layout_overfit_20260812_002747` 的 object/direction、bbox L1/GIoU/IoU 均达到实现门槛；仍不是性能结果 |
| validation loader 与统一页面指标 | 两页链路验证通过，正式 split 待执行 | `layout_validate_20260812_014816` 确认整页 prompt-only checkpoint 重载、generation 和统一指标；使用 overfit checkpoint/同一 `train` split，不能作为泛化结果 |

当前 P1 两样本实现诊断已经通过；随后 `layout_validate_20260812_014816` 在同一个 VLQA checkpoint 和两页 `train` split 上完成了整页 prompt-only validation 链路。布局区域 precision/recall/F1 均为 `1.0`，有序槽位 bbox mean IoU 为 `0.956666`，方向和阅读顺序指标均为 `1.0`；OCR 页面 CER 为 `0.215909`、去空白 CER 为 `0.086420`。这些数字来自 P1 同页 overfit checkpoint，P1 不优化 OCR，不能作为正式性能或泛化结果。下一步是锁定隔离的 held-out split 和原 GOT2 baseline；在正式划分、消融和跨域验证完成前，不启动未解锁的 pilot。完整状态和证据边界见 [PROJECT_STATUS.md](docs/PROJECT_STATUS.md)。

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

## 实验边界

| 入口 | 用途 | 能否作为当前主结果 |
|---|---|---|
| `tools/training/run_layout_a100.py --mode overfit` | 两样本实现正确性检查 | 否 |
| `tools/training/run_layout_a100.py --mode smoke` | 环境、单步训练与 checkpoint 链路检查 | 否 |
| `tools/training/run_layout_a100.py --mode validate` | 整页 prompt-only 推理与统一页面指标 | 暂不作为正式性能结论；需明确 VLQA checkpoint 和正式 split |
| `src/GOT-OCR-2.0/scripts/run_linelevel_smoke.sh` | 既有单行/单列工程诊断 | 否 |
| AncientDoc 旧 split 训练与兼容评估 | 历史页面兼容基线 | 否；存在书籍级重叠 |
| 统一整页划分下的 validation 与 `A0`–`A6` 消融 | 正式实验 | validation 代码已实现；正式 split、A100 结果和消融待完成 |

AnandaSky 是 line-level 高分辨率转写基线。双 GOT2 的“先分割、再识别”是独立两阶段系统对照。两者只有在页面输入、数据划分、训练预算和页面级指标一致时才可与本仓库路线比较。

## 文档

- [PROJECT_STATUS.md](docs/PROJECT_STATUS.md)：已实现、待验证和下一步。
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
