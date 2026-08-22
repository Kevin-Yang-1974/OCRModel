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

本轮 A0–A5 使用统一入口，输入固定为 whole-page image 与 OCR prompt。训练用 `--gpu-id ID` 选择单张物理卡，或用 `--gpu-ids ID[,ID...]` 选择一次 DeepSpeed 数据并行任务使用的多张物理卡；默认允许与其他进程共享瞬时 `utilization.gpu < 50` 的目标卡，达到或超过 50% 或查询失败时立即退出。阈值可用 `--gpu-utilization-limit` 调整。validation/test 使用该组绑定列表中的第一张物理卡。validation 只选择 checkpoint，test 只加载 `selection.json` 中锁定的 checkpoint。A1–A4 的 P2 steps 和优化配置相同；A5 另外执行 P1，因此总 steps 与总页面曝光量更高，不能写成同总预算。

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

启动器在创建任何子任务前检查全部目标 GPU 的瞬时利用率。低于共享阈值时允许继续并保留 GPU 锁、独立 run 和 launcher log；达到或超过阈值、查询失败或参数非法时整体退出。并发失败时不会终止其他已经启动的组，而是等待它们结束后汇总失败状态。

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

## 10. MTHv2 原始整页 VQLCA C1–C5

该入口固定读取 `/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1` 的 train/validation/test 原始整页 manifest，不读取 `mthv2_layout_column_chunks16_v1`。C1/C2 保持 projector/普通 adaptor 对照；C3/C4/C5 显式使用 `layout_writeback_mode=vqlca`。`max_regions=512` 只是覆盖当前最多约 407 个 ordered textline/region candidates 的 Fixed-Slot K512 容量设置，不是 PVLD。

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/training/run_mthv2_page_vqlca_ablation_tmux.sh \
  --session mthv2_page_vqlca_train_20260820 \
  --run-prefix mthv2_page_vqlca_ablation_20260820 \
  --gpu-id 0 \
  --ablations C1,C2,C3,C4,C5
```

启动前只查询实际指定 GPU 的瞬时利用率并要求 `<50%`。训练入口不自动启动 frozen test；训练结束后必须另行执行 validation-only checkpoint selection，再由锁定 selection 启动一次 test。

## 11. 多样化合成数据到 AncientDoc

新协议固定入口为 `tools/training/run_diverse_synthetic_ancientdoc.sh`。历史正式组只有 C0、C1、C4、C5、C6；不存在可直接复跑的 C2/C3。新 synthetic 数据不是 AncientDoc test 的替代品，AncientDoc 的 validation/test 仍使用书籍隔离整页数据。

先在本机用真实内容 manifest 和锁定字体生成数据。默认三 tier 合计 train/validation/test 为 `24000/3000/3000` 页；正式渲染前可把页数改小并加 `--plan-only`：

```powershell
Set-Location 'D:\yangky\学推计划\ocrmodel'
& .\.venv\Scripts\python.exe tools\preprocessing\prepare_diverse_synthetic_layout.py `
  --content-manifest D:\layout_source\content.jsonl `
  --content-root D:\layout_source `
  --output-root D:\layout_data\ancient_photo_diverse_v1_seed20260817 `
  --browser-channel msedge
```

该总入口在 Windows 当前 `.venv` 中分 split 调用现有生成器并复用系统 Edge。正式运行必须传入锁定字体文件及 SHA-256，并保留 `dataset_protocol.json`、三个 split 的 `dataset_meta.json` 和联合 `audit_summary.json`。A100 不生成页面。

代码仍使用第 1 节的白名单同步命令。数据目录单独上传到 `$GOT_LAYOUT_DATA/ancient_photo_diverse_v1_seed20260817`，不得放入源码树。确认服务器 `train/validation/test/manifest.jsonl` 及对应 `images/`、`html/` 完整后，按以下阶段运行。

1. 更长 synthetic A5 P1/P2，并只在 synthetic validation 选择 P2-best；该命令不运行 synthetic test：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/training/run_diverse_synthetic_ancientdoc.sh train-synthetic \
  --dataset-id ancient_photo_diverse_v1_seed20260817 \
  --run-prefix ancient_photo_diverse_v1 \
  --gpu-ids 0,1
```

默认 P1/P2 为 `12000/24000` optimizer steps，每 `2000` steps 保存。`selection.json` 的主指标为 synthetic validation page CER，去空白 CER 和较早 step 依次作为 tie-breaker；不是按 final step 或 test 选择。

2. 用被选中的 synthetic P2 checkpoint 启动 AncientDoc C1/C4：

```bash
bash tools/training/run_diverse_synthetic_ancientdoc.sh train-ancient-core \
  --synthetic-selection "$GOT_EVALUATION_RUNS/ancient_photo_diverse_v1_vlqa_layout_p1_p2_seed42_selection/selection.json" \
  --ancient-dataset-id ancientdoc_layout_260707_group_isolated_seed20260815 \
  --gpu-id 2 \
  --run-prefix ancient_photo_diverse_v1_ancient
```

入口通过 `--source-selection` 校验 selected model、config/weights SHA-256、validation-only 标志与 A5 ablation ID；不能手工替换为 synthetic final model。

3. 只在 AncientDoc validation 选择 C4-best：

```bash
bash tools/training/run_diverse_synthetic_ancientdoc.sh select-c4 \
  --c4-run "$GOT_TRAINING_RUNS/ancient_photo_diverse_v1_ancient_c4_vlqa_ocr_only" \
  --ancient-dataset-id ancientdoc_layout_260707_group_isolated_seed20260815 \
  --gpu-ids 0,1,2,3,4 \
  --output-dir "$GOT_EVALUATION_RUNS/ancient_photo_diverse_v1_ancient_c4_selection"
```

4. 从同一个 C4-best 独立训练 C5/C6。新协议默认 AncientDoc:synthetic 为 `7:1`，即请求 replay fraction 12.5%，低于旧实验的 25%：

```bash
bash tools/training/run_diverse_synthetic_ancientdoc.sh train-replay \
  --c4-selection "$GOT_EVALUATION_RUNS/ancient_photo_diverse_v1_ancient_c4_selection/selection.json" \
  --ancient-dataset-id ancientdoc_layout_260707_group_isolated_seed20260815 \
  --synthetic-dataset-id ancient_photo_diverse_v1_seed20260817 \
  --gpu-id 3 \
  --run-prefix ancient_photo_diverse_v1_ancient_replay
```

5. 选择 C1/C5/C6 checkpoints，C4 固定为第 3 步选择结果；只运行 validation：

```bash
bash tools/training/run_diverse_synthetic_ancientdoc.sh select-ancient \
  --c4-selection "$GOT_EVALUATION_RUNS/ancient_photo_diverse_v1_ancient_c4_selection/selection.json" \
  --c1-model "$GOT_TRAINING_RUNS/ancient_photo_diverse_v1_ancient_c1_got2_ocr_only/model" \
  --training-suite "$GOT_TRAINING_RUNS/ancient_photo_diverse_v1_ancient_replay" \
  --gpu-ids 0,1,2,3,4 \
  --run-prefix ancient_photo_diverse_v1_ancient_selection
```

6. 只有 validation selection 冻结后才执行一次正式 AncientDoc test：

```bash
bash tools/training/run_diverse_synthetic_ancientdoc.sh test-ancient \
  --c4-selection "$GOT_EVALUATION_RUNS/ancient_photo_diverse_v1_ancient_c4_selection/selection.json" \
  --selection "$GOT_EVALUATION_RUNS/ancient_photo_diverse_v1_ancient_selection/selection.json" \
  --suite-root "$GOT_EVALUATION_RUNS/ancient_photo_diverse_v1_ancient_selection" \
  --gpu-ids 0,1,2,3,4 \
  --resume
```

旧 AncientDoc test 已被查询过；新模型再次运行同一 test 属于重复 test 查询，不能再用其结果调 synthetic 分布、replay 比例或 checkpoint。后续超参数迭代只能看 validation，并应在新 seed 或新的 Real-OOD 集合上预注册复验。
### PVLD routing mode (2026-08-20)

Existing C1-C5 VQLCA runs retain their historical `layout_writeback_mode=vqlca`
identity and must not be relabeled as PVLD. A new PVLD run must explicitly set
`layout_writeback_mode=visual_value_layout_routing` and
`layout_writeback_source=layout_evidence`. This mode consumes high-resolution
Vary ViT features for `A=layout_evidence`, then applies factorized `V_i -> A -> V_i`
routing; the final OCR Value is always projected from `V_i` and the output
length remains `L_v`.

## 12. PVLD causal 修复 smoke 与 MTHv2 C3–C5 新流程（2026-08-22）

先使用第 1 节白名单同步工具；同步不包含数据、模型、checkpoint、run、缓存或日志。修复版有界 CUDA smoke 只需一次串行、无伪终端 SSH 调用：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/training/run_pvld_causal_cuda_smoke.sh \
  0 \
  pvld_causal_cuda_smoke_20260822_r1
```

入口只查询显式 GPU 0 的瞬时利用率，要求 `<50%`；使用新 evaluation run 目录，完整日志为 `smoke.log`，终端只打印紧凑 `summary.json`。检查项包括 causal self-attention、cross-attention、token head、coverage、bbox head 和 visual Value routing 的 finite gradient，0/1/多 REGION、FSM、两类 cap、REGION probability、bbox `[0,1]` 与 alpha=0 严格等价。它不加载 test，不启动正式训练。

smoke 成功后，本会话明确授权的新 C3–C5 流程为：

```bash
bash tools/training/run_mthv2_page_pvld_c3_c5_tmux.sh \
  --session mthv2_pvld_causal_c3_c5_20260822 \
  --run-prefix mthv2_pvld_causal_20260822_v1 \
  --gpu-ids 0,1,2,3,4
```

启动器固定读取原始整页 `mthv2_layout_page_v1`，不读取 oracle chunk。若可用卡数不少于三个，C3/C4/C5 各绑定一张；不足三个时，全部指定卡组成多卡作业并按 C3→C4→C5 串行。启动前查询命令指定的全部 GPU；任一卡 utilization 达到 50% 时，在创建任何控制任务前整体退出，不等待、不抢占、不停止已有进程。

C3/C4 为 P2 42000 steps；C5 为 P1 12000＋P2 30000 steps。checkpoint 间隔 2000。C5 P1 checkpoints 全部排队做自由生成 validation，预注册 ranking 为：停止错误总和、count MAE、region F1、matched/ordered bbox IoU、duplicate rate、count exact accuracy、较早 step；P2 只从 selected P1 初始化。训练完成后，各控制在 validation 选择 P2 checkpoint，并把 validation 默认未筛选阈值 0.0 写入 `selection.json`；随后 test 只加载锁定 checkpoint 与锁定阈值。test 结果不允许反向调整训练、P1 ranking 或 threshold。

该命令创建全新 run ID 和 tmux 会话，不使用 `--resume`，不覆盖旧 PVLD C1–C5、P1 checkpoints、selection 或 test。失败只查看对应 launcher/control 日志最后 20 行；不得直接输出完整 predictions、trainer state 或训练日志。
