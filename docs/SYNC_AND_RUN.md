# 同步与运行手册

> 适用范围：当前 A100 环境与本仓库的受限实验入口
>
> 命令基准：除非另行说明，均从仓库根目录执行

本手册不保存服务器地址、用户名或个人绝对路径。每位协作者在本机设置 SSH alias 和远端目录，并在服务器的未提交 `config/paths.env` 中配置模型、数据和 run 根目录。

## 1. 本地检查

Windows PowerShell：

```powershell
Set-Location '<path-to-ocrmodel>'
py -3 -m compileall -q src tools
py -3 -m pytest -q tests
```

语法检查不加载模型，也不占用 GPU。本地没有 PyTorch 时，依赖 PyTorch 的测试可以跳过，但必须记录跳过原因。

## 2. 同步活动代码

设置本机参数；尖括号内容必须替换：

```powershell
$RemoteHost = '<ssh-alias>'
$RemoteRoot = '/absolute/path/to/ocrmodel'
```

使用受限同步脚本：

```powershell
.\tools\sync\sync_to_server.ps1 `
  -RemoteHost $RemoteHost `
  -RemoteRoot $RemoteRoot
```

脚本只上传 Git 可见的 `src`、`tools`、`config` 和 `references` 文件，使用一个 `scp` 进程，不上传本机 `config/paths.env`、缓存、数据、模型、日志或结果，也不删除远端文件。详细范围见 [SYNC_SCOPE.md](../tools/sync/SYNC_SCOPE.md)。

已配置共享 Git 远程的协作者也可以在新的服务器目录直接 clone。父项目历史不属于共享仓库；不要把包含申报材料或个人记录的父分支推送到共享远程。

## 3. 初始化服务器路径

连接服务器后执行：

```bash
export OCRMODEL_ROOT='<absolute-path-to-ocrmodel>'
cd "$OCRMODEL_ROOT"
cp -n config/paths.env.example config/paths.env
```

编辑 `config/paths.env`，至少核对：

```bash
export OCR_WORKSPACE='<writable-workspace-outside-repository>'
export GOT_SOURCE_MODEL='<got2-model-directory>'
export GOT_LAYOUT_DATA='<whole-page-layout-dataset-root>'
export GOT_TRAINING_RUNS='<writable-training-run-root>'
```

然后加载配置并设置脚本权限：

```bash
source config/paths.env
find src tools references -type f -name '*.sh' -exec chmod 750 {} +
```

每个新 shell 都先执行：

```bash
export OCRMODEL_ROOT='<absolute-path-to-ocrmodel>'
cd "$OCRMODEL_ROOT"
source config/paths.env
```

`OCR_WORKSPACE`、数据、模型、环境、缓存和 run 目录必须位于源码树外。只读共享资产不得被安装或训练脚本修改。

## 4. 环境检查

先运行只读检查；结果只输出一行紧凑 JSON：

```bash
python3 tools/environment/check_server_envs.py
```

仅在 GOT 环境缺失、版本不符或 editable source 指向错误目录时执行：

```bash
bash tools/environment/setup_server_envs.sh got
python3 tools/environment/check_server_envs.py
```

AnandaSky 环境不是 VLQA 前置条件。只有运行 line-level AnandaSky 对照时才执行：

```bash
bash tools/environment/setup_server_envs.sh anandasky
bash tools/environment/install_anandasky_flash_attn.sh
python3 tools/environment/check_server_envs.py --include-anandasky
```

当前服务器 profile 固定使用物理 GPU 0。编排器发现 GPU 0 忙碌时以退出码 75 结束，不等待、不抢占，也不终止其他任务。

## 5. 已完成的 P1 两样本 overfit

当前代码已经打通整页 batch、VLQA CUDA forward/backward、P1→P2、checkpoint 保存和完整重载。`layout_overfit_20260812_002747` 已完成固定 P1、2 条记录、1000 steps 的实现诊断并通过；末 20 步 bbox L1、GIoU 和 mean IoU 分别为 `0.00531464`、`0.05508578` 和 `0.94497279`。该结果只证明两页实现可拟合，不是 validation 或性能结果。

该 run 已完成。不要因终端输出截断或网络波动重复启动同一 run；如需核对，只读取其 `summary.json` 的 `overfit_assessment` 紧凑字段。

## 6. 下一步：整页 prompt-only validation

validation 代码已在本地实现。运行时必须把 `--source-model` 指向包含 `use_vlqa=true` 和布局权重的 checkpoint；原始 GOT2 权重只能作为训练起点，不能直接作为 VLQA validation 模型。下面的两页 checkpoint 只用于验证加载、整页 generation 和指标链路，不构成正式性能结果：

首轮 run `layout_validate_20260812_012924` 被 evaluator 的 tokenizer 文件名预检错误拦截，未进入 checkpoint 加载或推理。当前代码已删除该白名单，改为直接使用 `AutoTokenizer.from_pretrained(..., local_files_only=True)` 验证 GOT/Qwen tokenizer；修复后的 `layout_validate_20260812_014816` 已在两页 `train` split 上完成链路验证。该 run 使用 `layout_overfit_20260812_002747/p1/model`，只证明 checkpoint 重载、整页 generation 和指标汇总，不是正式 validation 或性能结果。

```bash
dataset_root="$GOT_LAYOUT_DATA/vlqa_smoke_s0s2_seed20260810_v4"
validation_model="$GOT_TRAINING_RUNS/layout_overfit_20260812_002747/p1/model"
tokenizer_model="${GOT_TOKENIZER_MODEL:-$GOT_SOURCE_MODEL}"
bash "$OCRMODEL_ROOT/tools/environment/run_got2.sh" \
  "$OCRMODEL_ROOT/tools/training/run_layout_a100.py" \
  --dataset-root "$dataset_root" \
  --source-model "$validation_model" \
  --tokenizer-model "$tokenizer_model" \
  --layout-split train \
  --validation-max-records 2 \
  --mode validate
```

validation 不训练，不把 bbox、阅读顺序或书写方向传入模型。`layout_validate_20260812_014816` 已确认完成事件报告 `model_inputs=["whole_page_image","ocr_prompt"]` 和 `layout_metadata_as_model_input=false`。后续正式 split 仍只回传最后一条 `layout_a100_completed` JSON；生成的 `validation/layout_validation_metrics.json` 与 `validation/layout_validation_predictions.jsonl` 留在服务器 run 目录。

## 7. 新数据的单步 smoke

页面数据应先在本地按 [整页布局合成工具](../tools/preprocessing/README.md) 生成并审计，再由协作者显式上传到 `$GOT_LAYOUT_DATA/<dataset-id>`。数据不通过代码同步脚本传输。

首次使用新数据时运行：

```bash
dataset_root="$GOT_LAYOUT_DATA/<dataset-id>"
bash "$OCRMODEL_ROOT/tools/environment/run_got2.sh" \
  "$OCRMODEL_ROOT/tools/training/run_layout_a100.py" \
  --dataset-root "$dataset_root" \
  --mode smoke
```

`smoke` 固定为 P1、P2 各 1 条记录和 1 个 optimizer step。主要输出为：

```text
<run_root>/metadata/status.txt
<run_root>/metadata/audit_summary.json
<run_root>/p1/train.log
<run_root>/p1/model/layout_training_metrics.json
<run_root>/p1/metadata/checkpoint_verification.json
<run_root>/p2/...
<run_root>/summary.json
<run_root>/LAYOUT_A100_FINISHED
```

`pilot` 没有默认步数，并要求显式 `--allow-unvalidated-pilot`。在真实 validation 和统一实验协议锁定前，不启动 pilot。

## 8. 正式 P1/P2 训练入口

`overfit`、`smoke` 和 `pilot` 只是工程诊断，不是正式训练预算。正式数据必须是已经按来源组隔离并通过同一次 train/validation/test 联合审计的 HTML 整页 manifest；训练、验证和推理都只读取整页图像与 `OCR: ` prompt，bbox、顺序和方向只进入训练标签或离线指标。

正式 P1 使用原始 GOT2 权重，布局损失预训练 queries 和辅助头，OCR 标签全部 mask，写回门保持为 0。P1 只用于检验布局泛化，不能作为 OCR 性能比较 checkpoint。P1 held-out validation 通过后，再以同一 P1 checkpoint 启动 P2 `joint-train`；P2 才打开 OCR 损失和零门控写回，并可进入 baseline/VLQA OCR 对照。

### 8.1 本地同步

以下命令只同步活动源码、工具、配置模板和参考脚本，不上传数据、模型、checkpoint 或日志：

```powershell
Set-Location 'D:\yangky\学推计划\ocrmodel'
.\tools\sync\sync_to_server.ps1 `
  -RemoteHost a100-yky `
  -RemoteRoot /data3/yky/yangky_ocr_models/ocrmodel
```

正式数据不通过 `sync_to_server.ps1` 上传。若本地已生成正式数据目录，例如 `D:\yangky\datasets\got_layout_pages\formal_pdf_short_seed20260812`，使用独立数据上传入口：

```powershell
Set-Location 'D:\yangky\学推计划\ocrmodel'
.\tools\sync\upload_layout_dataset.ps1 `
  -RemoteHost a100-yky `
  -LocalDatasetRoot 'D:\yangky\datasets\got_layout_pages\formal_pdf_short_seed20260812' `
  -RemoteLayoutDataRoot /data3/yky/yangky_ocr_models/training_data/got_layout_pages
```

服务器的 `config/paths.env` 中应把 `GOT_LAYOUT_DATA` 指向同一个远端数据根，例如：

```bash
export GOT_LAYOUT_DATA=/data3/yky/yangky_ocr_models/training_data/got_layout_pages
```

训练前在服务器做只读挂载检查：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/training/check_layout_dataset_mount.sh formal_pdf_short_seed20260812
```

该检查只确认 `train/`、`validation/`、`test/`、`split_audit.json` 和首条图片可读，不启动训练。输出应为一行 `layout_dataset_mount_ok` JSON。

### 8.2 P1 正式预训练

将 `<P1_STEPS>` 固定为预注册的训练预算。此命令只运行一次 P1，并在结束后自动对 held-out validation 做 prompt-only 评测；test 只参与泄漏审计，不会被提前评测：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/training/run_formal_layout_p1p2.sh \
  pretrain formal_pdf_short_seed20260812 \
  --p1-steps <P1_STEPS>
```

P1 完成事件中的 `validation` 只表示 held-out 布局/生成链路和该阶段指标已跑完，不代表 OCR 已改善。正式记录应保留最后一条 `layout_a100_completed` JSON，以及 run 目录中的 `summary.json`、`metadata/audit_summary.json` 和 validation 指标文件。

### 8.3 P2 正式联合训练

只有 P1 的 held-out validation、审计和 checkpoint reload 均通过后才运行 P2。`<P1_RUN_ID>` 必须指向 P1 run 的 `p1/model`；编排器会再次检查 `config.use_vlqa=true`、完整布局状态和 `layout_stage=p1`，拒绝原始 GOT2 或部分布局 checkpoint：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/training/run_formal_layout_p1p2.sh \
  joint-train formal_pdf_short_seed20260812 \
  --p1-model "$GOT_TRAINING_RUNS/<P1_RUN_ID>/p1/model" \
  --p2-steps <P2_STEPS>
```

P2 的 validation 自动使用 P2 checkpoint，并要求 `layout_stage=p2`。P2 运行成功后才可以把该 checkpoint 用于 OCR 性能对照；P1 checkpoint 的 OCR 数字不应写入正式结果表。

`formal_pdf_short_seed20260812` 的首轮正式合成训练已完成：P1 run `layout_pretrain_20260813_003142` 为 `1000` steps，P2 run `layout_joint-train_20260813_012356` 为 `2000` steps。后续 test 对照 run `got2_vlqa_compare_20260813_020251` 显示，P2 VLQA 相对原始 GOT2 的页面 CER delta 为 `-0.087047`，去空白 CER delta 为 `-0.058194`，页面 exact match rate delta 为 `+0.19`。该结果只代表合成 test 同协议 OCR 对照优于原始 GOT2；布局完整 F1 和有序槽位 bbox IoU 仍低，且尚未完成消融，不能直接归因到 VLQA 或外推真实跨域。

长程训练也已完成：P1 run `layout_pretrain_4000_20260813` 为 `4000` steps，P2 run `layout_joint-train_8000_20260813` 为 `8000` steps。P2 8000 在 validation 上明显优于 P2 2000：页面 CER `0.139487`，布局完整 F1 `0.762183`，有序槽位 bbox mean IoU `0.652210`。P2 8000 的 test 对照 `got2_vlqa_compare_p2_8000_20260813` 已完成，相对原始 GOT2 的页面 CER delta 为 `-0.260725`，布局完整 F1 为 `0.787056`，有序槽位 bbox mean IoU 为 `0.647827`。

## 9. 原始 GOT2 与 VLQA 对照测试

`tools/evaluation/compare_got2_vlqa.py` 在同一 test manifest 上串行启动两个独立 evaluator：baseline 必须是未包含 VLQA 的原始 GOT2，VLQA 必须是 `layout_stage=p2` 的完整 checkpoint。两个模型共享整页图像、`OCR: ` prompt、tokenizer、贪心解码、`max_new_tokens`、`no_repeat_ngram_size`、图像预处理和 test split；bbox、阅读顺序和方向不作为输入。正式 test 必须同时传 train/validation manifest 做一次跨 split 审计，不能使用 `--skip-audit` 或 `--allow-non-test-split`。

短入口固定使用 `$GOT_LAYOUT_DATA/<dataset-id>/train|validation|test`，并自动把 train/validation/test manifest 纳入同一次审计：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/evaluation/run_formal_layout_comparison.sh \
  formal_pdf_short_seed20260812 \
  --vlqa-model "$GOT_TRAINING_RUNS/layout_joint-train_20260813_012356/p2/model"
```

P2 8000 的正式 test 对照命令如下；该 run 已完成，除非明确说明复跑原因，不要重复查询 test：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/evaluation/run_formal_layout_comparison.sh \
  formal_pdf_short_seed20260812 \
  --vlqa-model "$GOT_TRAINING_RUNS/layout_joint-train_8000_20260813/p2/model" \
  --run-id got2_vlqa_compare_p2_8000_20260813
```

如需限制调试页数，可追加 `--max-records <N>`，但正式 test 结果不要使用该限制。

`got2_vlqa_compare_20260813_020251` 已完成 P2 2000 正式 test 对照，`got2_vlqa_compare_p2_8000_20260813` 已完成 P2 8000 正式 test 对照。后续若复跑，必须使用新的 run 目录，并说明复跑原因，避免把多次 test 查询当作调参反馈。

终端只回传最后一条 `layout_comparison_completed` JSON。完整 OCR/layout 预测、审计和日志留在 comparison run 目录。汇总中的 CER、去空白 CER、exact-match 差值是 `VLQA - baseline`；VLQA 的布局指标和可选解释 forward 时间单独报告。由于 evaluator 会先做一次布局 forward 再做 OCR generation，`ocr_generation_seconds` 才是两模型可直接比较的 OCR 推理时间，`total_inference_seconds` 包含额外解释 forward，不能混作纯 OCR 延迟。

## 10. test 错误分析入口

`tools/evaluation/analyze_layout_comparison_errors.py` 是离线、只读、CPU 分析脚本，不加载模型、不占用 GPU。它读取 comparison run 下的 baseline/VLQA predictions 和 test manifest，输出每页 CSV、分组 CSV、Markdown 摘要和紧凑 JSON。默认按 tier、区域数量、方向组合、文本长度和 layout failure type 分组，用于回答：

1. VLQA 的 OCR 收益主要来自哪些页面；
2. layout F1 低主要来自漏检、多检、bbox/slot 低 IoU，还是混合问题。

服务器上运行：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
python3 tools/evaluation/analyze_layout_comparison_errors.py \
  --comparison-root /data3/yky/yangky_ocr_models/evaluation_runs/GOT/got2_vlqa_compare_p2_8000_20260813 \
  --manifest "$GOT_LAYOUT_DATA/formal_pdf_short_seed20260812/test/manifest.jsonl"
```

正常结束只回传最后一条 `layout_comparison_error_analysis_completed` JSON。详细文件默认写入：

```text
<comparison_root>/analysis/error_analysis_summary.json
<comparison_root>/analysis/page_error_analysis.csv
<comparison_root>/analysis/group_error_analysis.csv
<comparison_root>/analysis/error_analysis.md
```

进一步判断 `miss_and_extra` 是否主要由 object threshold 导致时，运行 threshold sweep。该脚本同样只读、不加载模型、不占 GPU：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
python3 tools/evaluation/analyze_layout_threshold_sweep.py \
  --comparison-root /data3/yky/yangky_ocr_models/evaluation_runs/GOT/got2_vlqa_compare_p2_8000_20260813 \
  --manifest "$GOT_LAYOUT_DATA/formal_pdf_short_seed20260812/test/manifest.jsonl"
```

默认评估 object threshold `0.1` 到 `0.9`。正常结束只回传最后一条 `layout_threshold_sweep_completed` JSON。详细文件默认写入：

```text
<comparison_root>/analysis/threshold_sweep/threshold_sweep_summary.json
<comparison_root>/analysis/threshold_sweep/threshold_sweep.csv
<comparison_root>/analysis/threshold_sweep/threshold_sweep.md
```

若 threshold sweep 不能显著提升 F1，再检查 query 槽位是否错配。slot alignment 脚本会比较“阅读顺序第 k 个真值区域与 query k 的 IoU”和“该真值区域与所有 query 的最高 IoU”：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
python3 tools/evaluation/analyze_layout_slot_alignment.py \
  --comparison-root /data3/yky/yangky_ocr_models/evaluation_runs/GOT/got2_vlqa_compare_p2_8000_20260813 \
  --manifest "$GOT_LAYOUT_DATA/formal_pdf_short_seed20260812/test/manifest.jsonl"
```

正常结束只回传最后一条 `layout_slot_alignment_completed` JSON。若 `best_hit_rate` 明显高于 `ordered_hit_rate`，或 `slot_misaligned_hit_rate` 较高，说明模型能用某些 query 找到区域，但 query index 与阅读顺序槽位错配。

P2 8000 的三项离线诊断已完成：错误分析显示 OCR 改善/持平/变差为 `264/16/20`，exact match 从 `11` 增至 `75` 页，layout failure 为 `layout_ok=124`、`miss_and_extra=169`；threshold sweep 最佳 F1 为 `0.789143`，仅略高于默认 `0.787056`；slot alignment 的 ordered hit rate 为 `0.769977`，best hit rate 为 `0.805936`，slot-misaligned hit rate 为 `0.035959`，best query offset 以 `0=1494` 为主。当前不需要重复运行上述三个诊断，除非 comparison 输出丢失或换了 checkpoint。

三项诊断完成后，可用 bundle 汇总入口把已有 JSON 压成一条紧凑结论和一份 Markdown，不重新推理、不读取大 prediction：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
python3 tools/evaluation/summarize_layout_analysis_bundle.py \
  --comparison-root /data3/yky/yangky_ocr_models/evaluation_runs/GOT/got2_vlqa_compare_p2_8000_20260813
```

正常结束只回传最后一条 `layout_analysis_bundle_completed` JSON。详细文件默认写入：

```text
<comparison_root>/analysis/analysis_bundle/analysis_bundle_summary.json
<comparison_root>/analysis/analysis_bundle/analysis_bundle.md
```

该入口会同时汇总 OCR delta、layout failure type、threshold 最优点相对默认 F1 的增量、slot alignment gap、主要 query offset、最差 group 和最差 page，用于决定是否继续看 hard pages、做消融或改结构。

若 offset 计数提示存在固定偏移，例如大量 best query offset 为 `+1`，再运行 fixed slot-offset sweep：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
python3 tools/evaluation/analyze_layout_slot_offset_sweep.py \
  --comparison-root /data3/yky/yangky_ocr_models/evaluation_runs/GOT/<comparison_run_id> \
  --manifest "$GOT_LAYOUT_DATA/formal_pdf_short_seed20260812/test/manifest.jsonl"
```

正常结束只回传最后一条 `layout_slot_offset_sweep_completed` JSON。若 `offset=+1` 的 hit rate 或 mean IoU 明显优于 `offset=0`，优先检查训练目标分配、背景/空槽处理和 query 0 是否被系统性占用。

该脚本同时输出每页最佳固定 offset oracle：

```text
<comparison_root>/analysis/slot_offset_sweep/slot_offset_page_oracle.csv
```

如果全局 `+1` 提升有限，但 page oracle 提升明显，说明错配更可能是页面级动态 offset，不是简单 off-by-one。

## 11. P2 loss-supervision 消融入口

当前已可重复运行的消融是 loss-supervision 消融，不是等参数结构消融。入口为 `tools/training/run_formal_layout_ablation.sh`，从同一个 P1 checkpoint 启动单个 P2 formal joint-train，并固定 train/validation/test 三 split 审计。

可用 preset：

| preset | 作用 |
|---|---|
| `full` | 完整 P2 VLQA，作为复跑对照 |
| `no-direction` | 关闭 direction loss |
| `no-bbox` | 关闭 bbox L1/GIoU loss |
| `object-only` | 只保留 object loss 和 OCR loss |
| `ocr-only-adapter` | 关闭全部 layout losses，只让 adapter 经 OCR loss 学习 |

示例只启动一个 ablation；不要一次性在终端里连续启动多个长任务：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/training/run_formal_layout_ablation.sh \
  no-direction \
  formal_pdf_short_seed20260812 \
  --p1-model "$GOT_TRAINING_RUNS/layout_pretrain_4000_20260813/p1/model" \
  --p2-steps 8000 \
  --run-id layout_ablate_no_direction_8000_20260813
```

每个 ablation 完成后，使用同一 comparison 和离线 analysis 链路：

```bash
bash tools/evaluation/run_formal_layout_comparison.sh \
  formal_pdf_short_seed20260812 \
  --vlqa-model "$GOT_TRAINING_RUNS/<ablation_run_id>/p2/model" \
  --run-id got2_vlqa_compare_<ablation_run_id>
```

然后依次运行 error analysis、threshold sweep、slot alignment 和 analysis bundle。回传只取最后一条 JSON。等参数普通 adaptor、无布局监督 query 结构对照和旧 oracle/pseudo-layout region-token adapter 仍需要单独结构实现或明确已有开关，不能用 `ocr-only-adapter` 冒充。

也可以直接使用 post-analysis 短入口，在一个命令中串行完成 comparison、三项离线诊断和 bundle 汇总：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/evaluation/run_formal_layout_post_analysis.sh \
  formal_pdf_short_seed20260812 \
  --vlqa-model "$GOT_TRAINING_RUNS/layout_ablate_no_direction_8000_20260813/p2/model" \
  --run-id got2_vlqa_compare_ablate_no_direction_8000_20260813
```

该脚本会拒绝已存在的 comparison run 目录，避免无意中重复 test 查询或覆盖结果。正常结束时只回传最后一条 `layout_analysis_bundle_completed` JSON。

## 12. line-level 兼容诊断

以下入口只检查既有单行/单列数据链路，不属于正式整页协议：

```bash
cd "$GOT_PROJECT_ROOT"
"$OCRMODEL_ROOT/tools/environment/run_got2.sh" scripts/preflight_linelevel_dataset.py \
  --source-model "$GOT_SOURCE_MODEL" \
  --annotations "$GOT_LINELEVEL_DATA/annotations.json" \
  --image-root "$GOT_LINELEVEL_DATA" \
  --model-max-length 1024

GOT_RUN_ID="linelevel_smoke_$(date +%Y%m%d_%H%M%S)" \
  bash scripts/run_linelevel_smoke.sh
```

单步 line-level smoke 只验证加载、反向传播、保存和重载，不验证布局 queries，也不能与页面 CER 直接比较。

## 13. AncientDoc 历史兼容入口

AncientDoc 旧 split 存在书籍级重叠。这些命令只用于复现历史页面兼容链路，不属于小样本或 VLQA 正式实验。

数据审计：

```bash
mkdir -p "$(dirname "$ANCIENTDOC_AUDIT")"
cd "$GOT_PROJECT_ROOT"
"$OCRMODEL_ROOT/tools/environment/run_got2.sh" scripts/audit_ancientdoc_dataset.py \
  --data-root "$ANCIENTDOC_ROOT" \
  --source-model "$GOT_SOURCE_MODEL" \
  --output "$ANCIENTDOC_AUDIT"
```

历史 split5 评估：

```bash
cd "$OCRMODEL_ROOT"
GOT_EVAL_MODEL="$GOT_SOURCE_MODEL" \
  bash tools/evaluation/run_legacy_ancientdoc_eval.sh
```

该入口复用 `references/legacy-ancientdoc-eval/GOT/eval/myeval.py` 的历史解码参数，并将预测、日志和指标写入 `$GOT_EVALUATION_RUNS`。输出字段 `metrics_page_macro_legacy_editops` 只代表该兼容口径。

## 14. 回传与协作

不要复制完整 `train.log`、`trainer_state.json` 或 `predictions.json`。布局 run 正常结束时优先回传终端最后一条完成 JSON。需要补充核对时只选择紧凑字段：

```bash
jq -c '{status,run_id,mode,overfit_assessment,p1:{status:.p1.status,global_step:.p1.metrics.global_step,train_loss:.p1.metrics.train_loss,diagnostics:.p1.metrics.diagnostics}}' \
  <run_root>/summary.json
```

错误只回传最后 20 行相关日志。SSH 断开、终端输出截断或网络波动不构成重复启动理由；先检查同一 `run_id` 的 `metadata/status.txt`、完成标志和 `summary.json`。
