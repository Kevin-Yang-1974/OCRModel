# MTHv2 统一内部基线与外部 SOTA 对比协议

> **命名边界**：当前论文方法为 LAVP（Layout-Aware Visual Prompting），当前工程子模块为 causal PVLD。Fixed-Slot VLQA/VQLCA 是历史方案，只作为内部 baseline；LAVP、PVLD、VLQA 和 VQLCA 不得混作同一模型名。

## 当前阶段边界

本协议用于建立可重复的 MTHv2 whole-page 对比工程。当前阶段只允许官方模型权重部署、validation 1–2 页 zero-shot smoke 和最多 1 optimizer step 的 fine-tune smoke；不启动正式长程微调、正式 validation selection、MTHv2 test 或 frozen test。所有正式 test 必须在 validation selection、prompt、阈值和后处理锁定后，由用户另行授权启动。

MTHv2 数据根固定为 `/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1`，官方划分为 train 2159 页、validation 240 页、test 800 页。正式输入是 `whole_page_image + ocr_prompt`；不使用 `mthv2_layout_column_chunks16_v1`、oracle chunk、人工 bbox、direction 或 reading_order 作为模型输入。`label_textline` 只能称为 ordered textline/region candidate，不是严格 column ground truth。

## 参数量口径

当前 GOT2 + LAVP/PVLD 完整模型参数量为 `564,759,576`（约 `564.76M/0.565B`），其中当前训练只更新 `5,280,536`（约 `5.28M`）参数：`mm_projector_vary=1,049,600`，LAVP/PVLD layout branch=4,230,936。外部模型比较使用完整推理模型参数量；训练成本单独报告可训练参数量，不能用 5.28M 冒充模型总参数。

## 内部基线

| ID | 模型 | 训练/推理状态 | 关键约束 |
|---|---|---|---|
| B0 | 官方 GOT2 | MTHv2 zero-shot | 只作原始参考 |
| B1 | 原始 GOT2 OCR-only | 同数据适配 | 尽量统一起点、seed、batch、optimizer 和页面曝光量 |
| B2 | GOT2 + 等参数量普通 adaptor | 同数据适配 | 目标可训练参数约 5.28M |
| B3 | Fixed-Slot VLQA-K32 | 同数据适配 | 明确记录 `>32` 区域容量限制 |
| B4 | LAVP/PVLD C3 | 已有 run 只读复用 | causal PVLD；不重复启动 |
| B5 | LAVP/PVLD C4 | 已有 run 只读复用 | causal PVLD；直接 P2 布局联合训练 |
| B6 | LAVP/PVLD C5 | 已有 run 只读复用 | causal PVLD；validation-selected P1 后进入 P2，额外 P1 预算单列 |

B4–B6 复用时记录原 run ID、selection 路径和权重哈希，不复制或覆盖服务器产物。

## 外部模型注册

| 模型 | 官方 checkpoint | 固定 revision | 许可证 | 参数量口径 | 微调状态 |
|---|---|---|---|---:|---|
| PaddleOCR-VL-1.6 | `PaddlePaddle/PaddleOCR-VL-1.6` | `c5630abae1d940eafe0697512a0325494b02ab42` | Apache-2.0 | 约 0.9B | 官方入口待确认，当前只部署/zero-shot |
| MinerU2.5-Pro | `opendatalab/MinerU2.5-Pro-2605-1.2B` | `bff20d4ae2bf202df9f45284b4d43681555a97ed` | Apache-2.0 | 约 1.2B | 未确认公开微调入口 |
| GLM-OCR | `zai-org/GLM-OCR` | `ca5d8b3e287e52589e37c28385d9655ee4372f9d` | MIT | 约 0.9B | 官方 LLaMA-Factory 入口可用，当前只做 1-step smoke |
| OpenDoc-0.1B | `topdu/unirec-0.1b` | `a377e00d62c01b6544603e2a90f2cffe2a0388e1` | Apache-2.0 | 约 0.1B | OpenDoc 文档只确认推理/下载，微调入口未确认 |

OpenDoc-0.1B 是由 OpenOCR 的 PP-DocLayoutV2 + UniRec-0.1B 组成的系统，不能猜测不存在的 `topdu/OpenDoc-0.1B` checkpoint。实际参数量最终以部署 checkpoint `config.json` 和运行时统计为准。

## 统一输出与指标

每个页面写入一条 JSONL，字段为：

```json
{"page_id":"...","image":"...","model":"...","raw_output":{},"normalized_text":"...","status":"ok|failed|blocked","runtime":{},"layout":null}
```

`raw_output` 必须保留；`normalized_text` 由确定性提取器生成，不访问参考答案。Markdown/HTML/结构化结果不允许只保留清洗文本。推理失败页也必须进入 JSONL。没有可确定对应到 MTHv2 textline/region candidate 的 bbox 时，布局字段写 `N/A`/`null`，不能写成 0。

OCR 指标统一报告 micro page CER、去空白 page CER、macro page CER、normalized edit distance、exact match、平均编辑距离、失败率、pages/s、延迟和峰值显存。布局指标只对确实输出且粒度可比的模型报告。

Zero-shot、MTHv2-adapted 和 future fine-tuned 结果分成三张表，不能混成同一训练条件排名。OmniDocBench 榜单只用于选择 SOTA 对照和记录公共 benchmark 背景；其指标不能复制为 MTHv2 指标，MTHv2 数字必须由本项目实际运行得到。

## 选择与 test lock

`tools/sota/select_validation.py` 只读取 validation prediction，固定记录 `selection_split=validation`、`test_used_for_selection=false`、prompt/postprocess/threshold locked。`run_selection_locked_test.py` 必须读取该 selection；当前入口始终拒绝执行 MTHv2 test，待用户明确授权后才解除锁定。禁止用 test 调整 prompt、阈值、后处理或 checkpoint。

## 运行目录和许可

模型权重、独立环境、数据和日志位于源码树外：

```text
/data3/yky/yangky_ocr_models/models/sota/<model>/<revision>
/data3/yky/yangky_ocr_models/envs/sota/<model>
/data3/yky/yangky_ocr_models/evaluation_runs/SOTA/<run-id>
```

下载脚本不覆盖已有目录；gated、许可、依赖或官方训练入口缺失时写结构化 blocker。当前协议不保存 token、密码或私钥。

## 2026-08-23 部署与 bounded smoke 记录

官方权重已下载到以下源码树外目录，并使用独立 run 保存日志和预测：

```text
/data3/yky/yangky_ocr_models/models/sota/paddleocr_vl_1_6/c5630abae1d940eafe0697512a0325494b02ab42
/data3/yky/yangky_ocr_models/models/sota/mineru2_5_pro/bff20d4ae2bf202df9f45284b4d43681555a97ed
/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d
/data3/yky/yangky_ocr_models/models/sota/opendoc_0_1b/a377e00d62c01b6544603e2a90f2cffe2a0388e1
```

部署 run 为 `sota_deploy_20260823_v3`。服务器缺少 `python3-venv/ensurepip`，四个独立环境均记录为 `blocked_missing_python_venv`；没有安装系统包，也没有修改共享 GOT2 环境。由于 `/data3` 仅剩约 88G，未继续无控制地安装大型 CUDA 依赖。

bounded smoke 结果如下，均只读取 validation 的 1 页，`test_used=false`：

| 模型 | zero-shot | 1-step fine-tune smoke | 具体状态 |
|---|---|---|---|
| GLM-OCR | 成功，run `sota_zero_shot_20260823_v12_glm_ocr` | CPU 成功，run `sota_finetune_smoke_20260823_v3_glm_ocr_cpu` | runtime 参数 `1,107,405,824`；zero-shot 延迟约 8.49 s、峰值显存约 3216 MiB；CPU 单步 loss `16.3594`，loss/gradient finite，checkpoint save/reload 成功 |
| PaddleOCR-VL-1.6 | 受控失败 | 受控失败 | 官方模型加载在独立 Transformers 环境中均为 `KeyError: 'default'`，未伪造预测或训练成功 |
| MinerU2.5-Pro | blocked | 未执行 | 当前 adapter 要求 MinerU 官方 CLI/API provider runtime，尚未确认可直接复用的公开本地推理契约 |
| OpenDoc-0.1B | 受控失败，run `sota_zero_shot_20260823_v5_opendoc_0_1b` | 未执行 | OpenOCR CLI 已加载，但运行时缺少 `paddlex` |

GLM GPU 1-step 曾在显式 GPU 3 上因完整模型 AdamW 显存不足退出；没有停止或干预其他进程。CPU smoke 的成功只证明官方 checkpoint 的单步 forward/backward、梯度和重载链路，不是 MTHv2 训练或性能结果。失败 run 全部保留，后续应先解决官方依赖/空间 blocker，再由用户授权正式训练和评测。

## 2026-08-23 正式入口启动记录

用户已明确授权启动正式 SOTA 训练与测评。正式编排器已同步并分别挂载到新 tmux 会话，始终只准入 GPU `0,1,3,4`，显式排除 GPU `2`。由于官方依赖 blocker，以下新 run 均在训练/推理阶段受控结束，未产生可报告的 MTHv2 性能：

```text
sota_formal_20260823_v2  session sota_mthv2_formal_20260823
sota_formal_20260823_v3  session sota_mthv2_formal_20260823_v2
sota_formal_20260823_v4  session sota_mthv2_formal_20260823_v3
sota_formal_20260823_v5  session sota_mthv2_formal_20260823_v4
sota_formal_20260823_v6  session sota_mthv2_formal_20260823_v5
sota_formal_20260823_v7  session sota_mthv2_formal_20260823_v6
```

`v2–v5` 分别暴露了默认 `python` 缺失、源码 `PYTHONPATH`、split image root 和独立 site-packages 注入问题；这些 run 均保留且没有覆盖。`v6` 已完成四个模型各 240 页 validation prediction、validation-only `selection.json` 和各 800 页 selection-locked test blocked prediction。`v7` 在注入已安装 site-packages 后仍受官方依赖错误阻断：PaddleOCR/GLM 报 PyTorch C 扩展加载失败，OpenDoc 报 `openocr executable not found`，MinerU 仍要求未确认的 provider runtime。因而当前没有正式训练 loss、checkpoint selection 或 SOTA 指标；必须先修复官方环境，再重新使用全新的 run ID。
## OpenDoc ONNX provider update (2026-08-24)

OpenDoc-0.1B now has a verified official OpenOCR ONNX path. The isolated
runtime uses the official `topdu/unirec_0_1b_onnx` assets downloaded through
ModelScope into `/data3/yky/yangky_ocr_models/models/sota/opendoc_onnx_cache_20260824`.
The bounded whole-page smoke completed with `status=ok`, provider
`official_openocr_onnx`, `layout_detection=false`, and approximately 13.93 s
per page. ONNX Runtime attempted CUDA but the server lacks cuDNN 9, so it
fell back to CPU; this is recorded as a runtime condition, not a model
substitution. The full run is mounted as
`sota_opendoc_formal_20260824_v1` and writes validation-only selection before
the selection-locked test stage.

OpenDoc has no verified official same-model MTHv2 fine-tuning entry in the
deployed OpenOCR/UniRec package. Its run records
`official_finetuning_unavailable`; the project does not claim a training
result or implement an unverified pseudo-trainer.

## 2026-08-24 已完成结果（暂不含 OpenDoc）

以下结果均使用 MTHv2 官方原始整页 test 800 页。外部模型使用官方 checkpoint，属于 zero-shot 表；内部模型使用 MTHv2 同数据训练，属于内部基线表。两类训练条件不同，不合并成同一公平排名。所有 test 都在 validation selection 和 prompt/postprocess/threshold 锁定后执行，`test_used_for_selection=false`。

### 外部官方 checkpoint zero-shot

| 模型 | 实际参数 | micro page CER | 去空白 CER | macro page CER | mean NED | 平均编辑距离 | 失败页 | 平均延迟 | pages/s | 布局指标 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GLM-OCR | 1,107,405,824 | 0.339441 | 0.273526 | 0.315755 | 0.288678 | 111.736 | 0/800 | 7.136 s | 0.1401 | N/A |
| PaddleOCR-VL-1.6 | 958,588,736 | 2.484101 | 2.125195 | 2.857077 | 0.999832 | 817.710 | 0/800 | 16.826 s | 0.0594 | N/A |
| MinerU2.5-Pro | 约 1.2B | N/A | N/A | N/A | N/A | N/A | blocked | N/A | N/A | N/A |

GLM-OCR 与 PaddleOCR-VL 均为 800 页有效 schema、0 inference failure、0 exact match。PaddleOCR-VL 的 CER 大于 1 是实际编辑距离结果，不应截断为 1；它说明当前固定 whole-page prompt/generation 配置在 MTHv2 上产生了大量插入或内容不匹配。GLM 的 prediction runtime 将 `peak_memory_mib` 写为 0，因此该字段视为未可靠采集，不作为显存比较。MinerU 的 test summary 为 `status=blocked`，原因仍是没有验证可用的官方本地 provider runtime，不能把 blocked prediction 当成空文本计算 CER。

统一指标文件为：

```text
/data3/yky/yangky_ocr_models/evaluation_runs/SOTA/sota_paddle_formal_20260824_v1/paddleocr_vl_1_6/test/unified_metrics.json
/data3/yky/yangky_ocr_models/evaluation_runs/SOTA/sota_formal_20260823_v13/glm_ocr/test/unified_metrics.json
```

独立 bounded 1-step 工程 smoke 均成功，但不是正式 MTHv2 微调：Paddle run `sota_paddle_train_smoke_20260824_v9` 的 loss 为 `11.313699`；GLM run `sota_finetune_smoke_20260823_v3_glm_ocr_cpu` 的 loss 为 `16.359404`。两者 loss/gradient finite、checkpoint save/reload 成功、`test_used=false`。正式 inference run 内附带的训练 smoke 失败记录继续保留，不覆盖上述独立成功 run。

### 内部 MTHv2 同数据 whole-page C1-C5（旧 PVLD 生成器）

| 控制 | test page CER | 去空白 CER | 平均编辑距离 | complete layout F1 | ordered bbox IoU | matched bbox IoU |
|---|---:|---:|---:|---:|---:|---:|
| C1 projector-only | 0.842911 | 0.825618 | 277.468 | N/A | N/A | N/A |
| C2 generic adaptor | 0.850016 | 0.847992 | 279.806 | N/A | N/A | N/A |
| C3 PVLD OCR-only | 0.889030 | 0.874585 | 292.649 | 0.000075 | 0.002170 | 0.583255 |
| C4 PVLD direct P2 | 0.885286 | 0.878747 | 291.416 | 0.106258 | 0.232918 | 0.687763 |
| C5 PVLD P1->P2 | 0.835286 | 0.819812 | 274.958 | 0.136239 | 0.256520 | 0.707138 |

该表对应 2026-08-22 已完成的旧 whole-page C1-C5 selection-locked test，exact match 均为 0/800。C5 在这组内部控制中 OCR 和布局最好，但只比 C1 降低 `0.007625` CER，且多出 P1 12,000 steps；单 seed 结果不足以声称结构已验证。该轮 C3-C5 使用后来确认存在缺陷的累计均值自由生成器，因此不能作为 causal decoder 修复版的性能结论。

causal 修复后 run `mthv2_pvld_causal_20260822_v1` 当前只有 C3/C4 分别完成 42,000-step P2 训练，尚无 validation selection 或 test；C5 的 P1 validation 在 step 4,000 选中 checkpoint，但 P2 因 `PVLD C5 P2 must initialize from its validation-eligible P1 model` 契约检查失败。该新 run 当前只有训练/故障结论，不能加入上述结果表。

## OpenDoc serialization 修复与正式重跑（2026-08-24）

旧 run `sota_opendoc_formal_20260824_v1` 错误地把 `blocks[*].img` NumPy 像素数组序列化到 raw output，同时文本标准化未识别官方 `recognition_results[*].text`。因此旧 validation/test prediction 分别膨胀到 `32,045,280,763` 和 `44,584,796,160` bytes，且其 normalized text 不可用于 CER。经用户明确授权，已永久删除这两个无效 JSONL，保留约 `1.6MB` 的日志、protocol、selection 和 finished metadata 作为失败审计；删除内容本机不可恢复。

修复后 raw output 保留官方有意义字段，但从 `blocks` 中排除内部 `img`；文本只从 `recognition_results[*].text` 等确定性官方字段提取。专用入口 `tools/sota/run_opendoc_formal_tmux.sh` 显式使用官方 OpenOCR ONNX CPU provider，不查询 GPU，不使用总启动器的 `cuda:4` 路径；validation 必须为 240 条且全部 `status=ok` 后才允许 selection，test 必须为 800 条且全部成功，最后自动计算统一指标。任何失败都会写 `failed.json`，不静默进入下一阶段。

新 tmux `sota_opendoc_formal_20260824_v2`、run `sota_opendoc_formal_20260824_v2` 已启动。真实首批验证为 provider `official_openocr_onnx`、device `cpu`、2 页共 `13,528` bytes；首条 `normalized_text` 长度 `511`，`blocks` keys 仅为 `box/label/merge_aligns/score`，不含 `img`，单页延迟约 `13.26s`。这只证明修复后的正式链路正在正常运行，完整 240/800 页和最终 CER 尚未完成。
