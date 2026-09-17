# BSCC 外部 SOTA zero-shot 运行手册（敦煌 + 地方志 Q32）

本文件用于在 BSCC 上评测三个官方外部对照模型：PaddleOCR-VL-1.6、MinerU2.5-Pro-2605-1.2B 和 OpenDoc-0.1B。当前协议不再使用 MTHv2。

## 数据与评测协议

本次数据由敦煌数据集和地方志数据集合并，便携包为：

`dunhuang_local_gazetteer_q32_v1_portable.zip`

便携包中的 `manifests/` 和 `images/` 是当前正式入口；`manifests_original/` 只保留来源追溯信息，不作为推理输入。

| split | 页数 | 敦煌 | 地方志 |
|---|---:|---:|---:|
| train | 240 | 73 | 167 |
| validation | 80 | 20 | 60 |
| test | 59 | 19 | 40 |

`source_group_overlap=0`，seed 为 42，最大区域数为 32。三个模型均只读取整页图像和固定 OCR prompt，不读取 bbox、阅读顺序、书写方向或 token-region 对应信息。prompt 固定为：

`Read all text on this page in reading order. Return plain text only. Do not describe the image.`

zero-shot 没有训练 checkpoint，因此没有 validation selection。test 只在显式设置 `ALLOW_SOTA_TEST=1` 后读取，并在每个 protocol/summary 中记录 `test_used_for_selection=false`。

三个模型统一使用确定性生成，最大输出长度为 `1536` tokens，与当前 GLM-OCR 分支正式 evaluator（`run_glmocr_a100_decoder_lora.sh` / `run_glmocr_mthv2_ddp.sh`）的 `max_eval_new_tokens` 一致；该值写入每个 shard 的 `protocol.json` 和 `run_summary.json`。

统一指标包括：micro page CER、去空白 micro CER、macro page CER、mean normalized edit distance、平均编辑距离、exact match、失败页、状态计数、平均/中位/P95 延迟、吞吐、峰值显存、输出长度，以及按 `dunhuang` / `local_gazetteer` 和参考文本长度分层的同组指标。CER 保留真实编辑距离结果，不截断到 1。

## BSCC 路径

假设：`WS="$HOME/yangky_ocr_models_bscc_proto"`，`C="$WS/glm_ocr_layout_ot/code/ocrmodel"`。

- 代码：`$C`
- 数据：`$WS/datasets/dunhuang_local_gazetteer_q32_v1_portable`
- 权重：
  - `$WS/models/sota/paddleocr_vl_1_6/c5630abae1d940eafe0697512a0325494b02ab42`
  - `$WS/models/sota/mineru2_5_pro/bff20d4ae2bf202df9f45284b4d43681555a97ed`
  - `$WS/models/sota/opendoc_onnx_20260912`
- 环境：`$WS/glm_ocr_layout_ot/envs/sota-transformers`、`$WS/glm_ocr_layout_ot/envs/sota-opendoc`
- 结果：`$WS/evaluation_runs/SOTA/<run_id>/<model>/test/`

## 运行步骤

### 1. 同步代码

```bash
cd /mnt/d/yangky/学推计划-glm-ocr/ocrmodel
bash tools/sync/sync_to_bscc.sh
```

### 2. 部署权重和环境

登录节点上执行已有入口；完整权重和环境只部署一次，完整文件不会被覆盖：

```bash
WS="$HOME/yangky_ocr_models_bscc_proto"
bash "$WS/glm_ocr_layout_ot/code/ocrmodel/tools/bscc/deploy_sota_models_bscc.sh"
bash "$WS/glm_ocr_layout_ot/code/ocrmodel/tools/bscc/setup_sota_envs.sh"
"$WS/glm_ocr_layout_ot/envs/sota-transformers/bin/python" -m pip install rapidfuzz
```

OpenDoc 还需要官方 OpenOCR 目录和 ONNX 缓存软链接，沿用原入口的布局：

```bash
[ -e "$HOME/openocr" ] || ln -s "$HOME/OpenOCR-main" "$HOME/openocr"
mkdir -p "$WS/models/sota/.cache/openocr"
[ -e "$WS/models/sota/.cache/openocr/unirec_0_1b_onnx" ] || \
  ln -s "$WS/models/sota/opendoc_onnx_20260912" "$WS/models/sota/.cache/openocr/unirec_0_1b_onnx"
```

### 3. test zero-shot

正式 test 使用 `manifests/test.jsonl`，每个模型可使用多个 Slurm 分片：

```bash
cd "$WS/glm_ocr_layout_ot/code/ocrmodel"
ALLOW_SOTA_TEST=1 bash tools/bscc/run_sota_zeroshot_submit.sh paddleocr_vl_1_6 test 4 <run_id>
ALLOW_SOTA_TEST=1 bash tools/bscc/run_sota_zeroshot_submit.sh mineru2_5_pro test 4 <run_id>
ALLOW_SOTA_TEST=1 bash tools/bscc/run_sota_zeroshot_submit.sh opendoc_0_1b test 4 <run_id>
```

BSCC 作业上限或节点资源不足时，按顺序逐个提交 array；不要重复提交已有 run 或已有 shard。脚本默认排除已知 CUDA 驱动异常节点 `paraai-n32-h-01-agent-4`。

### 4. 合并并计算指标

每个模型的 array 完成后执行：

```bash
SOTA_FINALIZE_PYTHON="$WS/glm_ocr_layout_ot/envs/sota-transformers/bin/python" \
  bash tools/bscc/finalize_sota_zeroshot.sh <model> test <run_id>
```

入口会严格校验 test manifest 的 59 页均有且仅有一条预测，生成：

- `predictions.jsonl`：按 manifest 顺序合并的逐页原始输出和规范化文本；
- `unified_metrics.json`：总集、两个域和文本长度分层的详细指标。

## 实现边界

- `tools/sota/dataset.py` 是当前合并数据集的 manifest 入口；旧 `mthv2.py` 仅作为历史兼容文件，不被当前 BSCC SOTA 入口引用。
- 模型仍使用官方 checkpoint 和固定 prompt；本次只替换数据协议，不进行微调、checkpoint 选择或 test 后处理调参。
- 代码、数据、模型和结果分开存放；test 结果完成后应保留 run 目录和 protocol，便于复核 manifest SHA-256、模型 revision 与运行环境。
