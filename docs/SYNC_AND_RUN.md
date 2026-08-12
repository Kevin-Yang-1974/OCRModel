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

`overfit`、`smoke` 和 `pilot` 只是工程诊断，不是正式训练预算。正式入口已经实现，但截至本手册更新时尚未在 A100 上运行正式大规模实验。正式数据必须是已经按来源组隔离并通过同一次 train/validation/test 联合审计的 HTML 整页 manifest；训练、验证和推理都只读取整页图像与 `OCR: ` prompt，bbox、顺序和方向只进入训练标签或离线指标。

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

## 9. 原始 GOT2 与 VLQA 对照测试

`tools/evaluation/compare_got2_vlqa.py` 在同一 test manifest 上串行启动两个独立 evaluator：baseline 必须是未包含 VLQA 的原始 GOT2，VLQA 必须是 `layout_stage=p2` 的完整 checkpoint。两个模型共享整页图像、`OCR: ` prompt、tokenizer、贪心解码、`max_new_tokens`、`no_repeat_ngram_size`、图像预处理和 test split；bbox、阅读顺序和方向不作为输入。正式 test 必须同时传 train/validation manifest 做一次跨 split 审计，不能使用 `--skip-audit` 或 `--allow-non-test-split`。

```bash
bash /data3/yky/yangky_ocr_models/ocrmodel/tools/environment/run_got2.sh \
  /data3/yky/yangky_ocr_models/ocrmodel/tools/evaluation/compare_got2_vlqa.py \
  --baseline-model /data3/yky/yangky_ocr_models/models/GOT-OCR2_0 \
  --vlqa-model /data3/yky/yangky_ocr_models/training_runs/GOT/<P2_RUN_ID>/p2/model \
  --tokenizer-model /data3/yky/yangky_ocr_models/models/GOT-OCR2_0 \
  --train-manifest /data3/yky/yangky_ocr_models/training_data/got_layout_pages/<TRAIN_DATASET_ID>/manifest.jsonl \
  --validation-manifest /data3/yky/yangky_ocr_models/training_data/got_layout_pages/<VALIDATION_DATASET_ID>/manifest.jsonl \
  --layout-manifest /data3/yky/yangky_ocr_models/training_data/got_layout_pages/<TEST_DATASET_ID>/manifest.jsonl \
  --layout-split test
```

终端只回传最后一条 `layout_comparison_completed` JSON。完整 OCR/layout 预测、审计和日志留在 comparison run 目录。汇总中的 CER、去空白 CER、exact-match 差值是 `VLQA - baseline`；VLQA 的布局指标和可选解释 forward 时间单独报告。由于 evaluator 会先做一次布局 forward 再做 OCR generation，`ocr_generation_seconds` 才是两模型可直接比较的 OCR 推理时间，`total_inference_seconds` 包含额外解释 forward，不能混作纯 OCR 延迟。

## 10. line-level 兼容诊断

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

## 11. AncientDoc 历史兼容入口

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

## 12. 回传与协作

不要复制完整 `train.log`、`trainer_state.json` 或 `predictions.json`。布局 run 正常结束时优先回传终端最后一条完成 JSON。需要补充核对时只选择紧凑字段：

```bash
jq -c '{status,run_id,mode,overfit_assessment,p1:{status:.p1.status,global_step:.p1.metrics.global_step,train_loss:.p1.metrics.train_loss,diagnostics:.p1.metrics.diagnostics}}' \
  <run_root>/summary.json
```

错误只回传最后 20 行相关日志。SSH 断开、终端输出截断或网络波动不构成重复启动理由；先检查同一 `run_id` 的 `metadata/status.txt`、完成标志和 `summary.json`。
