# Whole-page GOT2 符号识别

本目录是面向多场景、小样本条件的通用符号识别代码。当前共享主线只维护 GOT2 原生 whole-page 输入、PVLD 布局查询和 P1 → P2 → P3 训练流程。古籍、谱面、公式、表格等数据是验证场景，不限制模型的通用性表述。

## 当前主线

正式训练、验证和推理均接收 `whole_page_image + ocr_prompt`。bbox、书写方向和阅读顺序只作为训练期辅助监督或离线评测真值，不作为模型推理输入。

```text
P1 layout training
  -> validation-only checkpoint selection
P2 OCR + layout training
  -> validation-only checkpoint selection
P3 domain adaptation
  -> validation-only checkpoint selection
  -> selection-locked test
```

P2 必须从同一实验组已选中的 P1 checkpoint 初始化，P3 必须从已选中的 P2 checkpoint 初始化。test 不参与训练、checkpoint 选择、阈值或后处理调整。

## 活动入口

| 用途 | 文件 |
|---|---|
| P1/P2/P3 编排 | `tools/training/run_time_constrained_pvld_baseline.sh` |
| 阶段 runner、GPU 准入、checkpoint 衔接 | `tools/training/run_variable_layout_a100.py` |
| P1/P2/P3 validation selection | `tools/evaluation/select_layout_ablation_checkpoint.py` |
| selection-locked test | `tools/evaluation/evaluate_layout_ablation_test.py` |
| 400 页 validation lock | `tools/preprocessing/prepare_time_constrained_validation.py` |
| 合成页面生成与审计 | `tools/preprocessing/generate_synthetic_layout.py`、`audit_synthetic_layout.py` |
| MTHv2 whole-page 转换 | `tools/preprocessing/prepare_mthv2_layout_dataset.py` |
| GOT2/PVLD 架构 | `src/GOT-OCR-2.0/GOT/model/`、`src/GOT-OCR-2.0/scripts/` |

时间受限入口默认使用 S3/S4 validation 中按 tier、region-count bucket 和 complexity tertile 固定的 400 页；P1、P2、P3 共用该 validation manifest。MTHv2 的 `train` 仅用于 replay/P3 训练，`validation` 不进入训练，`test` 只在 P3 selection 后执行一次 locked test。完整 split 规则见 [`docs/BRANCH_AND_DATA_LAYOUT.md`](docs/BRANCH_AND_DATA_LAYOUT.md)。

## 目录边界

```text
src/GOT-OCR-2.0/   唯一活动 GOT2/PVLD 源码
tools/             当前训练、评估、预处理和同步入口
config/            可提交的配置模板（实际 paths.env 不提交）
docs/              当前协议、分支和运行说明
tests/             当前源码与编排器的 CPU 测试
```

数据、模型权重、checkpoint、日志、完整预测和运行目录必须位于源码树之外，由 `config/paths.env` 指定。不要把个人服务器路径、凭据或私钥写入仓库。

## 检查与运行

本地静态检查：

```bash
python -m compileall -q src tools
python -m pytest -q tests
```

服务器同步和有界 smoke/训练命令见 [`docs/SYNC_AND_RUN.md`](docs/SYNC_AND_RUN.md)。共享远程发布时使用父仓库 `master` 的 `ocrmodel` subtree 推送远端 `main`；分支映射见 [`docs/BRANCH_AND_DATA_LAYOUT.md`](docs/BRANCH_AND_DATA_LAYOUT.md)。

## 历史方案

Fixed-Slot VLQA/VQLCA、Chunk/oracle-chunk、AncientDoc、BSCC、SOTA、M2/M3/M4/All、line-level 和双 GOT2 编排器不属于主线。它们保存在远程 `archive/legacy-vlqa-chunk-20260829` 分支，仅用于复现和结果溯源，不得被当前 runner、selection 或 test 读取。需要复现历史结果时，请单独 clone 该归档分支，不要把旧脚本复制回主线。

当前 PVLD 源码中保留的 VLQA/VQLCA 名称仅用于既有 checkpoint 加载兼容；主线实际使用 `visual_value_layout_routing` 和 causal PVLD 路径。
