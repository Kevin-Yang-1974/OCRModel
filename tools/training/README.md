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

每个 checkpoint 保存前都会检查模型参数和 optimizer state 是否 finite，并写入 `checkpoint_health.json`。可在训练环境中对单个候选 checkpoint 做只读复核：

```bash
python tools/training/check_checkpoint_health.py /path/to/checkpoint-10000
```

命令返回 `status=nonfinite_weights` 或非零退出码时，该 checkpoint 不得用于 validation selection 或后续阶段初始化。

## BSCC P1 100k → 250k 与正式 P2

`run_bscc_p1_continue_p2_formal.sbatch` 是当前已登记的一体化 Slurm 入口。它使用 seed 42 锁定 400 页 validation，从既有 P1 `checkpoint-100000` 恢复 DeepSpeed optimizer/trainer state并累计训练到 250000，按 `100000/150000/200000/250000` 做 validation-only selection，然后直接启动 200000-step 正式 P2。P2 每 50000 steps 保存 checkpoint，并每 10000 steps写入并强制检查 `p2_health_checks.jsonl`。入口不运行 P2 selection、P3 或 test，不得重复提交同一 run。

## BSCC P1 50k 对照

BSCC 数据上的 50,000-step Legacy P1 对照同时提供两个平台入口，训练模式保持一致：

- A100：`run_bscc_p1_50000_a100.sh`（tmux，自动或显式物理 GPU）；
- BSCC：`run_bscc_p1_50000.sbatch`（Slurm，四卡）。

两者均使用 whole-page MTHv2 `train/validation/test` manifest、DeepSpeed ZeRO-2、bf16、`NCCL_P2P_DISABLE=1`、batch size 1、seed 42、64×64 layout memory 和同一 P1 冻结/学习率登记：Vary ViT/projector/layout 为 `1e-6/1e-5/1e-4`，Qwen、residual gate、lm head 冻结。BSCC 的 primary 已是 MTHv2 train，不重复启用同一 manifest 的 replay。P1 训练 50,000 steps，每 10,000 steps 保存；validation-only selection 严格只评估 `20,000/30,000/40,000/50,000`，不读取 test。该 BSCC run 是独立历史数据对照，不并入 S3/S4 主线结论。

## 训练口径

- `P1`：布局 query、causal PVLD 和辅助布局监督；OCR replay 仅按协议计入。
- `P2`：OCR 与布局联合训练，保持 whole-page 输入和视觉 Value 路由。
- `P3`：从 selected P2 checkpoint 进行域适配；不重新读取 test。
- `max_regions=512` 是变量长度生成的工程上限，不是 Fixed-Slot query 数。

完整数据划分和 MTHv2 用途见 [`docs/BRANCH_AND_DATA_LAYOUT.md`](../../docs/BRANCH_AND_DATA_LAYOUT.md)。服务器同步、环境和受限命令见 [`docs/SYNC_AND_RUN.md`](../../docs/SYNC_AND_RUN.md)。

## 历史入口

AncientDoc、BSCC、SOTA、Chunk、Fixed-Slot VLQA/VQLCA、M2/M3/M4/All、旧 layout runner 和历史 pilot 已从共享主线移出，保存在 `archive/legacy-vlqa-chunk-20260829`。主线不接受从归档分支复制脚本来启动正式实验；归档内容只用于复现和溯源。
