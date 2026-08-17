# AncientDoc 同步与运行

正式协议固定使用书籍隔离数据集 `ancientdoc_layout_260707_group_isolated_seed20260815`。模型输入为原始整页图像和 `OCR: ` prompt；布局 metadata 不作为推理输入。

## 1. 本地同步

```powershell
Set-Location 'D:\yangky\学推计划\ocrmodel'
.\tools\sync\sync_to_server.ps1 `
  -RemoteHost a100-yky `
  -RemoteRoot /data3/yky/yangky_ocr_models/ocrmodel
```

## 2. 训练阶段边界

旧的“训练 C1/C4 后自动用 C4-final 启动 C5/C6”流程已经废弃。当前只有三个明确阶段：

1. `train-core`：训练 C1 和 C4，然后停止。
2. `select-c4`：只在 validation 评估 C4 周期 checkpoints，冻结 C4-best。
3. `train-replay`：从同一个 C4-best 独立训练 C5 和 C6。

`test` 不参与 C4 分支点选择。

## 3. Train Core

新实验使用：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env

bash tools/training/run_ancientdoc.sh train-core \
  --ancient-dataset-id ancientdoc_layout_260707_group_isolated_seed20260815 \
  --steps 12000 \
  --checkpoint-steps 2000 \
  --learning-rate 2e-5 \
  --gpu-id 1 \
  --run-prefix ancientdoc_12k_seed20260815
```

当前 `ancientdoc_12k_seed20260815` 是由旧入口启动的在途 suite：C1 已完成，C4 正在训练。为保护在途 C4，未重启该 suite；旧 C5 预期路径已放置 `BLOCKED_PENDING_C4_SELECTION`，使旧父入口在 C4 完成后安全退出而不启动 C5/C6。

## 4. Select C4

C4 完成后运行一次：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env

bash tools/training/run_ancientdoc.sh select-c4 \
  --c4-run "$GOT_TRAINING_RUNS/ancientdoc_12k_seed20260815_c4_vlqa_ocr_only" \
  --ancient-dataset-id ancientdoc_layout_260707_group_isolated_seed20260815 \
  --gpu-ids 0,1,2,3,4 \
  --output-dir "$GOT_EVALUATION_RUNS/ancientdoc_12k_c4_selection_seed20260815"
```

该入口固定使用：whole-page、`OCR: `、greedy、`max_new_tokens=2048`、`no_repeat_ngram_size=20`、batch 1、BF16 和同一图像处理器。它自动发现全部 `checkpoint-*`，校验 config、完整权重、trainer step 和哈希；若最终 `p2/model` 与 `checkpoint-12000` 字节一致，只评估一次。

选择规则不可修改：page CER 最低、去空白 page CER 最低、optimizer step 更早。输出包括 `selection.json`、`selected_checkpoint_metadata.json`、`report.md`、`queue.jsonl` 及每个候选的独立 metrics、predictions、`launcher.log`。

## 5. Train Replay

使用新的 run prefix，不能复用被阻断的旧 C5 路径：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env

bash tools/training/run_ancientdoc.sh train-replay \
  --c4-selection "$GOT_EVALUATION_RUNS/ancientdoc_12k_c4_selection_seed20260815/selection.json" \
  --ancient-dataset-id ancientdoc_layout_260707_group_isolated_seed20260815 \
  --steps 12000 \
  --checkpoint-steps 2000 \
  --learning-rate 2e-5 \
  --gpu-id 1 \
  --run-prefix ancientdoc_12k_c4best_seed20260815
```

入口从 `selection.json` 解析 selected model/step/CER/hash，不接受手工替换路径。C5/C6 从完全相同的 C4-best 独立加载，并分别创建全新的 optimizer 和 scheduler。训练结束后会再次比较两者的分支 provenance；不一致直接失败。

## 6. Final Validation Selection

C1、C5、C6 从各自周期 checkpoints 在 validation 选 best；C4 固定为上一步 C4-best：

```bash
bash tools/evaluation/run_ancientdoc.sh \
  --phase select \
  --c4-selection "$GOT_EVALUATION_RUNS/ancientdoc_12k_c4_selection_seed20260815/selection.json" \
  --c1-model "$GOT_TRAINING_RUNS/ancientdoc_12k_seed20260815_c1_got2_ocr_only/model" \
  --training-suite "$GOT_TRAINING_RUNS/ancientdoc_12k_c4best_seed20260815" \
  --gpu-ids 0,1,2,3,4 \
  --batch-size 1 \
  --run-prefix ancientdoc_12k_final_selection_seed20260815
```

评估入口会检查 C5/C6 metrics 中的 selected C4 step/path/hash 和 selection 路径完全一致，并拒绝 C4 报告 checkpoint 与 replay 分支起点不一致的 suite。

## 7. Frozen Test

```bash
bash tools/evaluation/run_ancientdoc.sh \
  --phase test \
  --c4-selection "$GOT_EVALUATION_RUNS/ancientdoc_12k_c4_selection_seed20260815/selection.json" \
  --selection "$GOT_EVALUATION_RUNS/ancientdoc_12k_final_selection_seed20260815/selection.json" \
  --suite-root "$GOT_EVALUATION_RUNS/ancientdoc_12k_final_selection_seed20260815" \
  --gpu-ids 0,1,2,3,4 \
  --batch-size 1 \
  --resume
```

只有冻结后的 C0、C1-best、C4-best、C5-best、C6-best 进入一次 test。

## 8. 紧凑回传

C4 selection：

```bash
jq -c '{status,purpose,c4_run_root,selection_split,test_used_for_selection,selected:{step:.selected.optimizer_step,model:.selected.model_path,metrics:.selected.validation_metrics},candidates:[.candidates[]|{step:.optimizer_step,model:.model_path,metrics:.validation_metrics}],excluded_duplicates}' \
  "$GOT_EVALUATION_RUNS/ancientdoc_12k_c4_selection_seed20260815/selection.json"
```

最终 test：

```bash
jq -c '{status,deltas,fairness,c4_branch_selection,replay_branch_consistency}' \
  "$GOT_EVALUATION_RUNS/ancientdoc_12k_final_selection_seed20260815/summary.json"
```

失败时只回传对应 `launcher.log` 最后 20 行，不输出完整 predictions 或训练日志。

## 9. GOT2 整页结构消融

本轮 A0–A5 使用统一入口，输入固定为 whole-page image 与 OCR prompt。训练用 `--gpu-id ID` 选择单张物理卡，或用 `--gpu-ids ID[,ID...]` 选择一次 DeepSpeed 数据并行任务使用的多张物理卡；任一指定卡忙碌时立即退出。validation/test 使用该组绑定列表中的第一张物理卡。validation 只选择 checkpoint，test 只加载 `selection.json` 中锁定的 checkpoint。A1–A4 的 P2 steps 和优化配置相同；A5 另外执行 P1，因此总 steps 与总页面曝光量更高，不能写成同总预算。

多个消融也可以各绑定一张卡并发训练。下面按列表顺序将 A2、A3、A4、A5 分别绑定到物理 GPU 1、2、3、4；这与单个实验的 `--gpu-ids` 多卡数据并行不同：

```bash
bash tools/training/run_layout_ablation_suite.sh \
  --dataset-id formal_pdf_short_seed20260812 \
  --ablations generic_adapter_projector,vlqa_ocr_only,vlqa_layout_direct,vlqa_layout_p1_p2 \
  --parallel-gpu-ids 1,2,3,4 \
  --p1-steps 4000 \
  --p2-steps 8000 \
  --checkpoint-steps 2000 \
  --seed 42 \
  --run-prefix layout_ablation_formal_v1
```

启动器在创建任何子任务前检查全部目标 GPU。任一卡忙碌即整体退出；通过预检后每组使用独立 run、GPU 锁和 launcher log。并发失败时不会终止其他已经启动的组，而是等待它们结束后汇总失败状态。

单组训练、选点和 Synthetic-ID test：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env

bash tools/training/run_layout_ablation_suite.sh \
  --dataset-id formal_pdf_short_seed20260812 \
  --ablations vlqa_layout_direct \
  --p2-steps 8000 \
  --checkpoint-steps 2000 \
  --gpu-ids 0,1 \
  --seed 42 \
  --run-prefix layout_ablation_formal_v1
```

顺序运行 A1–A5，并为 A5 指定 P1：

```bash
bash tools/training/run_layout_ablation_suite.sh \
  --dataset-id formal_pdf_short_seed20260812 \
  --ablations projector_only,generic_adapter_projector,vlqa_ocr_only,vlqa_layout_direct,vlqa_layout_p1_p2 \
  --p1-steps 4000 \
  --p2-steps 8000 \
  --checkpoint-steps 2000 \
  --gpu-ids 0,1 \
  --seed 42 \
  --run-prefix layout_ablation_formal_v1
```

额外 test 数据集通过可重复的 `--test-set Category:dataset-id` 指定，例如 `--test-set Synthetic-OOD:synthetic_ood_id --test-set Real-OOD:real_ood_id`。不同输入粒度不得加入同一结果比较。已有完成标志时默认退出；确认复用已完成训练、selection 或 test 时显式加 `--resume`。不完整目录不会被自动覆盖。

### 9.1 Selection 断点续跑

`--resume` 发现训练 run 已有 `LAYOUT_A100_FINISHED` 时不会重新训练。若某个 validation candidate 已经写出 `layout_validation_metrics.json`，selector 只在以下字段全部匹配时复用：`status=ok`、模型绝对路径、`model_kind`、validation manifest、`split=validation`、greedy decoding 参数、推理失败数，以及 `whole_page_image`＋`ocr_prompt` 输入协议。`layout_metadata_as_model_input` 必须为 `false`。不匹配时立即退出，不覆盖已有输出，也不把旧配置结果混入本轮选点。

evaluator 当前使用 `character_edits`、`reference_characters` 和 `page_exact_matches`；selector 会将其规范化为稳定的 `total_edit_distance`、`total_reference_characters` 和 `exact_matches`，并兼容历史字段名。该规范化不改变 page CER 主指标、去空白 page CER 次级指标和较早 optimizer step 的选点顺序。

已完成 `projector_only` 训练后，只继续 selection 和 selection-locked test：

```bash
bash tools/training/run_layout_ablation_suite.sh \
  --dataset-id formal_pdf_short_seed20260812 \
  --ablations projector_only \
  --p2-steps 8000 \
  --checkpoint-steps 2000 \
  --gpu-id 0 \
  --seed 42 \
  --run-prefix layout_ablation_formal_v1 \
  --resume
```

selection resume 的独立工程 smoke 只评估一个 checkpoint 的 1 页，并通过 evaluator 日志哈希确认第二次调用没有重新推理。它不用于选正式 checkpoint，也不产生性能结论：

```bash
bash tools/evaluation/run_layout_ablation_selection_smoke.sh \
  layout_ablation_formal_v1_projector_only_seed42 \
  formal_pdf_short_seed20260812 \
  2000
```

成功时最后一行事件为 `layout_ablation_selection_resume_smoke_completed`，且 `evaluator_log_unchanged=true`。正式 run 的 candidate 目录、`selection.json` 和 test 结果不得用该 1 页 smoke 替代。
