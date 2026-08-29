# 当前训练入口

本目录的共享主线只有 GOT2 whole-page PVLD 的阶段化训练。正式输入是整页图像和 OCR prompt；bbox、方向和阅读顺序只作为训练期辅助监督或评测真值。

## P1 → P2 → P3

唯一的时间受限编排入口是 `run_time_constrained_pvld_baseline.sh`：

1. 从既有 P1 周期 checkpoint 中按 validation-only 规则选择 P1-best。
2. 从 P1-best 启动 P2 OCR＋layout 训练，再按 validation 选择 P2-best。
3. 从 P2-best 启动 P3 域适配，再按 validation 选择 P3-best。
4. 仅使用 P3 selection 之后锁定的 checkpoint 执行一次 test。

P1/P2/P3 必须使用 whole-page manifest；test 不得参与训练、选点、阈值或后处理。P2/P3 的 checkpoint provenance 写入 selection 文件和 summary。

```bash
bash tools/training/run_time_constrained_pvld_baseline.sh \
  --dataset-root "$GOT_LAYOUT_DATA" \
  --mthv2-root "$MTHV2_LAYOUT_ROOT" \
  --p1-run-root "$P1_CHECKPOINT_ROOT" \
  --run-prefix pvld_main_20260829_v1
```

入口默认只在启动瞬间查询命令允许范围内 `utilization.gpu < 50` 的卡；可用 `--gpu-ids` 显式限定。忙卡不等待、不抢占，已有 run 不覆盖。

## 直接使用阶段 runner

`run_variable_layout_a100.py` 负责 GPU 准入、模型加载、阶段步数、保存和 checkpoint 衔接。它支持 `p1`、`p2`、`p3` 阶段及有界 smoke；正式调用必须显式提供 train/validation manifest、source checkpoint 和新的 run ID。selection 使用 `tools/evaluation/select_layout_ablation_checkpoint.py`，locked test 使用 `tools/evaluation/evaluate_layout_ablation_test.py`。

## 训练口径

- `P1`：布局 query、causal PVLD 和辅助布局监督；OCR replay 仅按协议计入。
- `P2`：OCR 与布局联合训练，保持 whole-page 输入和视觉 Value 路由。
- `P3`：从 selected P2 checkpoint 进行域适配；不重新读取 test。
- `max_regions=512` 是变量长度生成的工程上限，不是 Fixed-Slot query 数。

完整数据划分和 MTHv2 用途见 [`docs/BRANCH_AND_DATA_LAYOUT.md`](../../docs/BRANCH_AND_DATA_LAYOUT.md)。服务器同步、环境和受限命令见 [`docs/SYNC_AND_RUN.md`](../../docs/SYNC_AND_RUN.md)。

## 历史入口

AncientDoc、BSCC、SOTA、Chunk、Fixed-Slot VLQA/VQLCA、M2/M3/M4/All、旧 layout runner 和历史 pilot 已从共享主线移出，保存在 `archive/legacy-vlqa-chunk-20260829`。主线不接受从归档分支复制脚本来启动正式实验；归档内容只用于复现和溯源。
