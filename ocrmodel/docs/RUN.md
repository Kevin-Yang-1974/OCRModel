# 本地检查与后续运行

## 当前可执行内容

建立隔离环境并运行单元测试：

```bash
cd "/mnt/d/yangky/学推计划-glm-ocr/ocrmodel"
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

manifest 为 JSONL，每行至少包含 `page_id`、`split`、`source_group` 和 `duplicate_group`。审计 split 隔离并生成确定性 128 页筛选列表：

```bash
layout-ocr-audit /external/path/manifest.jsonl --screen-pages 128 --seed 42
```

工具只输出一行 JSON。任一 `source_group` 或 `duplicate_group` 跨 train/validation/test 时非零退出。

## BSCC 稳定性确认

同步代码：

```bash
bash tools/sync/sync_to_bscc.sh
```

隔离环境、锁定权重和 R1/R2 协议已由既有 setup 完成。同步后用新的唯一 run ID 提交：

```bash
cd "$HOME/yangky_ocr_models_bscc_proto/glm_ocr_layout_ot/code/ocrmodel"
screen_job=$(GLM_OCR_SCREEN_ID=mechanism_stable_128_3seed_v1 sbatch --parsable \
  tools/bscc/run_mechanism_screen.sbatch)
printf '{"screen_job":"%s","run_id":"mechanism_stable_128_3seed_v1"}\n' "$screen_job"
```

`screen_job` 是 7 个单卡 array task：前 6 个覆盖 `attention`/`geometry`、固定辅助损失 `0.2` 和 seed `42/43/44`，最后一个运行零训练更新的 `content_only` eval-only 基线。每个训练组运行 1024 steps，使用同一 128 页 train 与 64 页 validation；64 页 test 不在本轮读取。生成上限固定为 768，不读取参考文本长度。

6 个训练 run 和基线全部完成后汇总 validation-only 结果：

```bash
python tools/summarize_confirmation.py \
  "$HOME/yangky_ocr_models_bscc_proto/glm_ocr_layout_ot/training_runs/mechanism_stable_128_3seed_v1"
```

训练产物按 `seed{42,43,44}/{attention,geometry}_aux0.2/` 保存，基线位于 `content_only_eval/`。汇总器只接受完整且 finite 的四个 checkpoint，按三种子平均 CER 选择模式与 step，并输出标准差、低频字符召回、生成触顶率、逐种子 geometry−attention 差值和稳定性验收结果。

## 首轮训练退化诊断：geometry / seed42 / 0-64-128

本轮只做快速诊断，不读取 test，也不改变既有稳定性确认结果。启动器将 step 0 作为未训练 adapter 的 identity validation，训练 128 steps，并在 64、128 保存 checkpoint 后分别 generation。每个点写入 `diagnostic_summary.json`，同时保留训练侧的 loss 分项、梯度范数、gate、residual、transport/query mass 和 dtype 记录。

本地同步后，使用新的唯一 run ID 提交单个作业：

```bash
bash tools/sync/sync_to_bscc.sh
diagnostic_job=$(GLM_OCR_DIAGNOSTIC_ID=geometry_seed42_steps128_v1 sbatch --parsable \
  tools/bscc/run_geometry_diagnostic.sbatch)
printf '{"diagnostic_job":"%s","run_id":"geometry_seed42_steps128_v1"}\n' "$diagnostic_job"
```

结果目录为 `training_runs/<run_id>/seed42/geometry_aux0.2/`；本轮只需回传其中的 `diagnostic_summary.json` 和最后的紧凑完成 JSON。

## 后续短程因果消融

同一诊断启动器支持一次只改变一个因素的 geometry/seed42/128-step 实验。默认仍是原始 mixed-BF16、fixed-order、完整辅助损失；FP32 首轮使用新的唯一 run ID：

```bash
GLM_OCR_DIAGNOSTIC_ID=geometry_seed42_steps128_fp32_v2 \
GLM_OCR_ADAPTER_PRECISION=fp32 \
sbatch --parsable tools/bscc/run_geometry_diagnostic.sbatch
```

loss profile 可选 `full`、`ocr_only`、`no_assignment`、`no_geometry`；query assignment 可选 `fixed_order`、`hungarian`。每次只设置一个变量，且不读取 test：

```bash
GLM_OCR_DIAGNOSTIC_ID=geometry_seed42_steps128_hungarian_v2 \
GLM_OCR_QUERY_ASSIGNMENT=hungarian \
sbatch --parsable tools/bscc/run_geometry_diagnostic.sbatch
```

完成后生成 identity-aware 紧凑汇总：

```bash
python tools/summarize_diagnostic.py \
  "$HOME/yangky_ocr_models_bscc_proto/glm_ocr_layout_ot/training_runs/<run_id>/seed42/geometry_aux0.2"
```

FP32 版本的 256-step 运行已完成，但它把学习率调度周期也延长到了 256 步，因此同一名义 step 的学习率与 128-step 运行不同，不能直接用来判断 128→256 的因果变化。下一轮固定训练总步数为 256，同时把学习率调度周期锁定为 128；128 步以后保持最小学习率继续观察：

```bash
GLM_OCR_DIAGNOSTIC_ID=geometry_seed42_steps256_fp32_lr128_v4 \
GLM_OCR_ADAPTER_PRECISION=fp32 \
GLM_OCR_MAX_STEPS=256 \
GLM_OCR_LR_SCHEDULE_STEPS=128 \
GLM_OCR_DIAGNOSTIC_STEPS=0,64,128,256 \
GLM_OCR_DIAGNOSTIC_STEPS_JSON='[0,64,128,256]' \
sbatch --parsable tools/bscc/run_geometry_diagnostic.sbatch
```

该运行仍只读取 train/validation，不读取 test；`metadata.json`、`diagnostic_summary.json` 和训练指标会记录独立的 `lr_schedule_steps`，用于确认 64/128 步轨迹与前一轮 128-step FP32 对照一致。

## 服务器边界

本地 A100 后续运行入口必须从环境变量读取外部资产，且输出位于 `/data3/yky/yangky_ocr_models/glm_ocr_layout_ot`。`/data4/hyf` 始终只读。每次命令只可查询明确允许的物理 GPU；任一目标卡的瞬时利用率达到 50% 时，在启动子任务前整体退出。BSCC 由 Slurm 分配 GPU，不在登录节点查询或抢占物理卡。

正式训练入口还必须自动执行 validation-only checkpoint selection 和 selection-locked test，且不允许 test 参与任何调参。
