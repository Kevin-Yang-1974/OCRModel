# 主线同步与运行

本文只描述共享 `main` 的 GOT2 whole-page PVLD 主线。旧 VLQA/VQLCA、Chunk、AncientDoc、BSCC、SOTA、M2/M3/M4/All 和历史 pilot 入口位于 `archive/legacy-vlqa-chunk-20260829`，不在主线运行。

## 1. 同步源码

在 Windows Git 中，从 `ocrmodel` 目录执行：

```powershell
Set-Location 'D:\yangky\学推计划\ocrmodel'
.\tools\sync\sync_to_server.ps1 `
  -RemoteHost a100-yky `
  -RemoteRoot /data3/yky/yangky_ocr_models/ocrmodel
```

同步白名单只包含活动 `src`、`tools`、`config` 和必要参考文件；数据、权重、checkpoint、日志和本地环境留在源码树外。

## 2. 数据准备与审计

本地生成 S3/S4 whole-page 数据并对 train/validation/test 三份 manifest 一起审计：

```bash
python tools/preprocessing/prepare_diverse_synthetic_layout.py \
  --content-manifest <content.jsonl> \
  --content-root <content-root> \
  --output-root <dataset-root>
python tools/preprocessing/audit_synthetic_layout.py \
  --manifest <dataset-root>/train/manifest.jsonl \
  --manifest <dataset-root>/validation/manifest.jsonl \
  --manifest <dataset-root>/test/manifest.jsonl \
  --summary-json <dataset-root>/audit_summary.json
```

数据必须先按来源组划分，再生成页面；禁止同源页、近重复页或同一内容跨 split。正式时间受限运行会由 `prepare_time_constrained_validation.py` 从 S3/S4 validation 锁定 400 页（各 200 页）。

MTHv2 转换后保持官方 `train/validation/test` split：train 供 P1 replay/P3，validation 不训练，test 仅最终 locked test。`label_textline` 只是有序区域候选。

## 3. 当前 P1 → P2 → P3 入口

在 A100 项目环境中：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/training/run_time_constrained_pvld_baseline.sh \
  --dataset-root "$GOT_LAYOUT_DATA" \
  --mthv2-root "$MTHV2_LAYOUT_ROOT" \
  --p1-run-root "$P1_CHECKPOINT_ROOT" \
  --run-prefix pvld_main_20260829_v1
```

入口按顺序执行 P1 validation selection、P2、P2 validation selection、P3、P3 validation selection 和 selection-locked MTHv2 test。P2/P3 只能从 selection 文件中的 checkpoint 初始化；已有 run 或输出目录不会被覆盖。

GPU 默认只查询本次命令允许集合中瞬时 `utilization.gpu < 50` 的卡。需要固定物理卡时传 `--gpu-ids 0,1`；忙卡不等待、不抢占，查询失败或无合格卡时整体退出。

## 4. 双平台 BSCC P1 对照入口

BSCC 的 50,000-step P1 对照必须与 A100 使用相同训练模式和冻结策略。A100 使用 tmux 入口：

```bash
bash tools/training/run_bscc_p1_50000_a100.sh \
  --run-prefix bscc_p1_legacy_50000_a100_20260829_v1
```

BSCC 使用 Slurm 入口（提交一次，不重复启动）：

```bash
sbatch tools/training/run_bscc_p1_50000.sbatch
```

两份脚本均为 `bscc_pvld_p1_legacy`，P1 最大步数为 `50000`，保存间隔为 `10000`，validation 候选严格为 `20000,30000,40000,50000`，并记录 `test_used_for_selection=false`。A100 与 BSCC 只改变调度器和环境初始化，不改变 runner、DeepSpeed ZeRO-2、bf16、whole-page 输入、batch、seed、loss 或参数冻结/学习率。BSCC 默认工作区为 `$HOME/yangky_ocr_models_bscc_proto`；A100 默认读取 `/data3/yky/yangky_ocr_models` 下的 MTHv2 与 `/data4/hyf/backup` 原始 GOT2 权重。

## 5. 有界检查

```bash
bash tools/training/run_pvld_causal_cuda_smoke.sh <gpu-id> <new-run-id>
python -m compileall -q src tools
python -m pytest -q tests
```

smoke 只验证模型加载、前向/反向和 checkpoint 链路，不能作为正式性能结果。每次重试使用新的 run ID，并保留失败目录。

## 6. 结果回传

只回传 `metadata/status.txt`、完成标志和紧凑 `summary.json` 的必要字段；日志最多回传最后 20 行。不要输出完整训练日志、预测全集、私有路径、凭据或模型权重。

正式发布时从父仓库 `master` 执行 `git subtree split --prefix=ocrmodel`，将生成的 subtree 推送到共享远程 `main`；旧方案 subtree 推送到 `archive/legacy-vlqa-chunk-20260829`。分支、数据和 MTHv2 划分说明见 [`BRANCH_AND_DATA_LAYOUT.md`](BRANCH_AND_DATA_LAYOUT.md)。详细个人发布命令保存在被忽略的本地 `docs/PUBLISHING.md`。
