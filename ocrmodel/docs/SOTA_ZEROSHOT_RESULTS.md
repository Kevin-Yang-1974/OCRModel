# 外部 SOTA zero-shot 数据记录（MTHv2）

本文件是外部 SOTA 零样本结果的**数据记录**，机器可读版本见
`docs/data/sota_zeroshot_results.json`。记录区分 A100 历史结果、BSCC 重跑结果和
尚未完成的项目；未完成前不写成已验证结论。

## 协议

- 数据：MTHv2 official split，train 2159 / validation 240 / test 800 页。
- 输入：`whole_page_image + ocr_prompt`；prompt 固定：
  `Read all text on this page in reading order. Return plain text only. Do not describe the image.`
- 推理不使用 bbox、阅读顺序、书写方向或 token-region 对应。
- 外部官方 checkpoint 属于 zero-shot；没有训练 checkpoint，因此没有 validation
  selection，`test_used_for_selection=false`。
- 指标由 `tools/sota/summarize_metrics.py` 计算：micro page CER、去空白 CER、
  macro page CER、mean NED、平均编辑距离、exact match、失败页、延迟。CER>1 是实际
  编辑距离结果，不截断为 1。

## 结果总表

### A100 历史运行（归档 tools/sota，2026-08-24）

test 800 页。

| 模型 | 参数 | micro CER | 去空白 CER | macro CER | mean NED | 平均编辑距离 | 失败页 | 延迟 | pages/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM-OCR | 1,107,405,824 | 0.339441 | 0.273526 | 0.315755 | 0.288678 | 111.736 | 0/800 | 7.136 s | 0.1401 |
| PaddleOCR-VL-1.6 | 958,588,736 | 2.484101 | 2.125195 | 2.857077 | 0.999832 | 817.710 | 0/800 | 16.826 s | 0.0594 |
| MinerU2.5-Pro | ~1.2B | blocked | — | — | — | — | — | — | — |
| OpenDoc-0.1B | ~0.1B | 未完成 | — | — | — | — | — | — | — |

证据（服务器 `/data3`）：

```text
/data3/yky/yangky_ocr_models/evaluation_runs/SOTA/sota_formal_20260823_v13/glm_ocr/test/unified_metrics.json
/data3/yky/yangky_ocr_models/evaluation_runs/SOTA/sota_paddle_formal_20260824_v1/paddleocr_vl_1_6/test/unified_metrics.json
```

### BSCC 重跑（run `sota_zeroshot_20260912_v1`，2026-09-12）

| 模型 | split | 页数 | 参数 | micro CER | 去空白 CER | macro CER | mean NED | 平均编辑距离 | 失败页 | 延迟 | 峰值显存 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PaddleOCR-VL-1.6 | validation | 240 | 905,601,648 | 0.934449 | 0.864979 | 0.889020 | 0.379808 | 319.854 | 0/240 | 41.86 s | 1996.5 MiB |
| PaddleOCR-VL-1.6 | test | 800 | 905,601,648 | 1.142799 | 1.074565 | 1.018510 | 0.377411 | 376.184 | 0/800 | 40.39 s | 1997.1 MiB |
| MinerU2.5-Pro-2605-1.2B | test | 800 | 1,156,026,624 | 1.033758 | 0.695229 | 1.092458 | 0.295153 | 340.290 | 0/800 | 51.01 s | 2612.9 MiB |
| OpenDoc-0.1B | test | 800 | ~0.1B | 1.206067 | 1.168359 | 1.341855 | 0.966948 | 397.010 | 0/800 | 109.64 s | N/A |

状态说明：

- PaddleOCR-VL-1.6 / MinerU2.5-Pro / OpenDoc-0.1B 的 test 均为 `800/800 status=ok`，已完成。
- MinerU2.5-Pro 的 validation（240 页）为优先跑 test 被主动取消，无 validation 指标。
- 一次 PaddleOCR test 分片曾落在坏节点 `paraai-n32-h-01-agent-4`（驱动 11.6）而
  blocked 100 页，已用 `--exclude` 重跑该分片，最终 0 失败。

证据（BSCC，路径相对
`$HOME/yangky_ocr_models_bscc_proto/evaluation_runs/SOTA/sota_zeroshot_20260912_v1/`）：

```text
paddleocr_vl_1_6/validation/unified_metrics.json
paddleocr_vl_1_6/test/unified_metrics.json
mineru2_5_pro/test/unified_metrics.json
opendoc_0_1b/test/unified_metrics.json   # 完成后生成
```

## 观察

- GLM-OCR 是 A100 历史运行中 MTHv2 test 上最好的外部模型（micro CER 0.339）；
  BSCC 未重跑 GLM-OCR，沿用该历史记录并单独标注来源。
- BSCC 上 PaddleOCR-VL-1.6 与 MinerU2.5-Pro 的 micro CER 均 >1（插入/内容不匹配
  多），去空白后 MinerU2.5-Pro（0.695）明显好于 PaddleOCR-VL-1.6（1.075）。
- A100 与 BSCC 的 PaddleOCR-VL-1.6 结果训练条件相同（官方 checkpoint、zero-shot），
  但运行环境/prompt 生成配置不同，两个数字不得混成同一排名，需分别标注来源。

## 复现入口

```bash
# 权重（BSCC 登录节点，ModelScope）
bash tools/bscc/deploy_sota_models_bscc.sh
# 环境
bash tools/bscc/setup_sota_envs.sh
# 推理（validation / test 分片）
ALLOW_SOTA_TEST=1 bash tools/bscc/run_sota_zeroshot_submit.sh <model> test 8 <run_id>
# 合并 + 统一指标
bash tools/bscc/finalize_sota_zeroshot.sh <model> test <run_id>
```
