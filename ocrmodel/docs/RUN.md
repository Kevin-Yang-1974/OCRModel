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

## 服务器边界

本地 A100 后续运行入口必须从环境变量读取外部资产，且输出位于 `/data3/yky/yangky_ocr_models/glm_ocr_layout_ot`。`/data4/hyf` 始终只读。每次命令只可查询明确允许的物理 GPU；任一目标卡的瞬时利用率达到 50% 时，在启动子任务前整体退出。BSCC 由 Slurm 分配 GPU，不在登录节点查询或抢占物理卡。

正式训练入口还必须自动执行 validation-only checkpoint selection 和 selection-locked test，且不允许 test 参与任何调参。
