# A100 布局训练编排器

`run_layout_a100.py` 是 GOT2＋VLQA 合成预训练的服务器侧编排入口。它不生成页面，而是读取本机已经生成并整体上传的 `manifest.jsonl`、`images/` 和 `html/`。

默认 `--mode smoke` 依次执行：

1. 校验 GOT Python、PyTorch、Transformers、DeepSpeed 版本及活动源码位置；
2. 检查物理 GPU 0，忙碌时以退出码 75 结束；
3. 复审 manifest、图片哈希、DOM、字体证据和跨 split 泄漏；
4. 在 GPU 0 上以放大视觉输入运行独立 VLQA 小张量 forward/backward，并拒绝异常初始 logit 尺度；
5. P1 使用原始 GOT2 权重，确认 checkpoint 没有 VLQA 键后显式完整初始化适配器，再执行 1 条记录、1 个 optimizer step；
6. 验证 P1 的 VLQA 权重、有限性和严格为 0 的 `residual_gate`；
7. P2 必须从 P1 模型目录加载完整 VLQA 状态，部分或不兼容状态会在训练前被拒绝，再执行 1 条记录、1 个 optimizer step；
8. 验证 P2 的 VLQA、`mm_projector_vary`、训练指标和完整模型重载。

完整 Trainer 输出只写入 run 目录，终端只显示紧凑状态。参数化同步与 A100 启动命令见 `../../docs/SYNC_AND_RUN.md`。

## 服务器数据入口

正式整页数据不随代码同步。先在本地用 `tools/sync/upload_layout_dataset.ps1` 上传整个数据目录，再在服务器用 `check_layout_dataset_mount.sh` 做只读检查：

```bash
bash tools/training/check_layout_dataset_mount.sh formal_pdf_short_seed20260812
```

该检查只确认 `$GOT_LAYOUT_DATA/<dataset-id>` 下存在 `train/manifest.jsonl`、`validation/manifest.jsonl`、`test/manifest.jsonl`、`split_audit.json` 和三份 split 的图片目录，并输出一行紧凑 JSON；不会启动训练。

A100 run `layout_pilot_20260811_023528` 已打通上述工程链路，但其 2 页、每阶段 500 epoch 的配置没有 validation，P1 也未拟合到接近零，不能作为有效 pilot 或性能结果。

A100 run `layout_overfit_20260811_110317` 已执行首次受控诊断并返回 fail。第 1 步 object/direction logits 已约为 1680，bbox 同时饱和到约 0/1，说明主要异常在优化前的 VLQA 初始化或前向尺度，而不是训练后期逐渐发散。当前修复在加载原始 GOT2 后显式完整初始化 VLQA，严格区分无布局权重、完整布局权重和部分布局权重，并在辅助头前增加最终 LayerNorm 与小尺度 bbox 输出层初始化。

A100 run `layout_overfit_20260811_113817` 已确认初始化修复有效：首步 object/direction/bbox raw logits 分别为 0.1592、0.3984 和 0.0010。200 步后 object 与 direction 已通过阈值；bbox L1 从 0.9761 降至尾段均值 0.1159，bbox mean IoU 从 0.0726 升至 0.3496，说明 bbox 正在学习但现有步数不足以完成严格实现检查。

`layout_overfit_20260812_002747` 已完成固定 P1、2 条记录、1000 optimizer steps 的实现诊断并返回 `overfit_assessment.status=pass`。末 20 步均值为 object loss `0.00201755`、bbox L1 `0.00531464`、bbox GIoU `0.05508578`、direction loss `0.00154159`、object/direction accuracy `1.0`、bbox mean IoU `0.94497279`。`status=pass` 只代表两样本实现检查通过，不代表泛化性能。

当前 validation 入口为 `--mode validate`。它只执行环境/manifest/component preflight 和 prompt-only 整页评测，不启动 DeepSpeed 训练；必须通过 `--source-model` 指向包含 VLQA 权重的 checkpoint，并可用 `--tokenizer-model` 指向原始 GOT tokenizer。评测输出统一写入 run 目录，布局 metadata 只用于离线指标。

首轮 run `layout_validate_20260812_012924` 在模型加载前被 evaluator 的 tokenizer 文件名白名单错误拦截。当前 loader 已改为直接让 Transformers 离线验证本地 GOT/Qwen tokenizer；修复后的 `layout_validate_20260812_014816` 已在两页 `train` split 上通过 prompt-only checkpoint 重载和指标链路。该结果只属于同页 P1 overfit 诊断。

`pilot` 没有默认训练步数，必须同时提供 `--allow-unvalidated-pilot`、`--p1-max-steps` 和 `--p2-max-steps`。在真实 validation 和统一实验协议锁定前，不应启动 pilot。

## 正式训练模式

`run_layout_a100.py` 的正式模式与开发诊断严格分开：

| 模式 | 起点 | 训练阶段 | 自动评测 | 用途 |
|---|---|---|---|---|
| `pretrain` | 无 `use_vlqa` 的原始 GOT2 | 仅 P1 | held-out validation | HTML 整页布局 queries 预训练；不用于 OCR 性能结论 |
| `joint-train` | 完整 `layout_stage=p1` VLQA checkpoint | 仅 P2 | held-out validation | 打开 OCR 损失和写回门，得到可参与 OCR 对照的 checkpoint |

两种正式模式都要求显式训练步数、`--validation-manifest` 和 `--test-manifest`。启动前三份 manifest 会被一次性联合审计，validation/test 必须非空且 split 名不同；test 只用于泄漏审计，不会被训练入口自动评测。训练 manifest 必须直接位于 `--dataset-root` 下，validation/test 图像根默认取各自 manifest 的父目录。

P1 的 `ocr_loss_weight=0`、OCR token 全部 mask、`residual_gate=0`；P2 的 OCR loss 必须为正，并从 P1 checkpoint 加载完整 VLQA 状态。编排器拒绝带 VLQA 的 P1 起点、原始 GOT2 的 P2 起点、部分布局状态和错误的 `layout_stage`。P1 held-out validation 通过后才启动 P2；P2 checkpoint 才能用于原始 GOT2 baseline 对照。

服务器侧短入口为 `run_formal_layout_p1p2.sh`，它固定使用 `$GOT_LAYOUT_DATA/<dataset-id>/train`、`validation` 和 `test` 三个子目录，并显式传入各 split 的 image root：

```bash
bash tools/training/run_formal_layout_p1p2.sh pretrain formal_pdf_short_seed20260812 --p1-steps <P1_STEPS>
bash tools/training/run_formal_layout_p1p2.sh joint-train formal_pdf_short_seed20260812 --p1-model "$GOT_TRAINING_RUNS/<P1_RUN_ID>/p1/model" --p2-steps <P2_STEPS>
```

尖括号中的训练步数必须替换为预注册预算。P2 只能在 P1 held-out validation、审计和 checkpoint reload 均通过后运行。

`formal_pdf_short_seed20260812` 的首轮正式合成训练已在 A100 上完成：P1 run `layout_pretrain_20260813_003142` 为 `1000` steps，P2 run `layout_joint-train_20260813_012356` 为 `2000` steps，二者均完成 300 页 validation。P2 的合成 validation OCR CER 相比 P1 明显下降，但布局完整 F1 和有序槽位 bbox IoU 未改善。

首轮 test 对照 `got2_vlqa_compare_20260813_020251` 已完成：P2 VLQA 相对原始 GOT2 的页面 CER delta 为 `-0.087047`，去空白 CER delta 为 `-0.058194`，页面 exact match rate delta 为 `+0.19`。该结果说明合成 test 同协议 OCR 对照优于原始 GOT2，但尚不能直接归因到 VLQA，也不是真实跨域泛化性能。

长程训练 `layout_pretrain_4000_20260813` 与 `layout_joint-train_8000_20260813` 已完成。P2 8000 validation 页面 CER 为 `0.139487`，布局完整 F1 为 `0.762183`，有序槽位 bbox mean IoU 为 `0.652210`，相对 P2 2000 显著继续收敛。P2 8000 test comparison `got2_vlqa_compare_p2_8000_20260813` 也已完成：相对原始 GOT2 的页面 CER delta 为 `-0.260725`，布局完整 F1 为 `0.787056`，有序槽位 bbox mean IoU 为 `0.647827`。后续合成数据主 checkpoint 应暂以 P2 8000 为准，但仍需消融和真实跨域验证。

P2 完成后，可使用 `tools/evaluation/run_formal_layout_comparison.sh` 在同一 test split 上比较原始 GOT2 与 P2 VLQA。当前合成数据主 checkpoint 使用 P2 8000：

```bash
bash tools/evaluation/run_formal_layout_comparison.sh \
  formal_pdf_short_seed20260812 \
  --vlqa-model "$GOT_TRAINING_RUNS/layout_joint-train_8000_20260813/p2/model" \
  --run-id got2_vlqa_compare_p2_8000_20260813
```

comparison 完成后，使用离线 CPU 脚本做可重复错误分析：

```bash
python3 tools/evaluation/analyze_layout_comparison_errors.py \
  --comparison-root "$GOT_EVALUATION_RUNS/got2_vlqa_compare_p2_8000_20260813" \
  --manifest "$GOT_LAYOUT_DATA/formal_pdf_short_seed20260812/test/manifest.jsonl"
```

该脚本不加载模型、不占用 GPU，默认输出每页 CSV、分组 CSV、Markdown 和紧凑 JSON，按 tier、区域数、方向、文本长度和 layout failure type 汇总 OCR delta 与布局失败类型。

`got2_vlqa_compare_p2_8000_20260813/analysis` 已完成 P2 8000 离线错误分析：300 页中 VLQA 有 `264` 页 OCR 改善、`16` 页持平、`20` 页变差，exact match 增加 `64` 页；layout failure type 中 `layout_ok=124`、`miss_and_extra=169`。相对 P2 2000 的 `layout_ok=3` 和 `miss_and_extra=285`，长程训练已显著减少布局失败。

object threshold 先用离线 sweep 检查，不重新推理：

```bash
python3 tools/evaluation/analyze_layout_threshold_sweep.py \
  --comparison-root "$GOT_EVALUATION_RUNS/got2_vlqa_compare_p2_8000_20260813" \
  --manifest "$GOT_LAYOUT_DATA/formal_pdf_short_seed20260812/test/manifest.jsonl"
```

P2 8000 的最佳 threshold 为 `object_threshold=0.3`，complete region F1 为 `0.789143`，只比默认 `0.787056` 略高；预测区域数 `1748` 与真值区域数 `1752` 基本匹配，因此剩余误差不主要来自阈值或区域数量偏差。

若阈值不能显著提升 F1，再运行 slot alignment 诊断：

```bash
python3 tools/evaluation/analyze_layout_slot_alignment.py \
  --comparison-root "$GOT_EVALUATION_RUNS/got2_vlqa_compare_p2_8000_20260813" \
  --manifest "$GOT_LAYOUT_DATA/formal_pdf_short_seed20260812/test/manifest.jsonl"
```

P2 8000 的 slot alignment 已显示 ordered hit rate `0.769977`、best hit rate `0.805936`、slot-misaligned hit rate `0.035959`，best query offset 以 `0=1494` 为主，说明 P2 2000 阶段明显的槽位错配在 P2 8000 已基本收敛。

三项诊断完成后，可用 bundle 入口汇总现有 analysis JSON：

```bash
python3 tools/evaluation/summarize_layout_analysis_bundle.py \
  --comparison-root "$GOT_EVALUATION_RUNS/got2_vlqa_compare_p2_8000_20260813"
```

该入口输出 `analysis/analysis_bundle/analysis_bundle_summary.json` 和 `analysis_bundle.md`，并在最后一条 `layout_analysis_bundle_completed` JSON 中返回 OCR、threshold、slot alignment 和剩余错误判断，便于低噪声回传。

## P2 loss-supervision 消融入口

`tools/training/run_formal_layout_ablation.sh` 用于从同一个 P1 checkpoint 启动单个 P2 loss-supervision 消融。它只封装现有 `train_GOT_layout.py` 已支持的 loss weight，不实现等参数普通 adaptor 或无 VLQA 结构对照。

可用 preset：

| preset | 含义 |
|---|---|
| `full` | 完整 P2 VLQA：object + bbox L1/GIoU + direction + OCR |
| `no-direction` | 去除方向监督，保留 object + bbox + OCR |
| `no-bbox` | 去除 bbox L1/GIoU 监督，保留 object + direction + OCR |
| `object-only` | 只保留 object 监督 + OCR，去除 bbox 和 direction |
| `ocr-only-adapter` | VLQA adapter 只通过 OCR 训练，关闭全部 layout losses |

示例：

```bash
bash tools/training/run_formal_layout_ablation.sh \
  no-direction \
  formal_pdf_short_seed20260812 \
  --p1-model "$GOT_TRAINING_RUNS/layout_pretrain_4000_20260813/p1/model" \
  --p2-steps 8000 \
  --run-id layout_ablate_no_direction_8000_20260813
```

每个 ablation 结束后仍使用同一个 formal comparison 入口做 test 对照，再运行 error analysis、threshold sweep、slot alignment 和 analysis bundle。不能只用 validation loss 判断消融结果。

如果希望减少手工步骤，可用 post-analysis 入口串行完成 comparison 和三项离线诊断：

```bash
bash tools/evaluation/run_formal_layout_post_analysis.sh \
  formal_pdf_short_seed20260812 \
  --vlqa-model "$GOT_TRAINING_RUNS/layout_ablate_no_direction_8000_20260813/p2/model" \
  --run-id got2_vlqa_compare_ablate_no_direction_8000_20260813
```

该脚本最后只输出 `layout_analysis_bundle_completed` JSON；若指定的 comparison run 目录已存在会直接退出，避免重复 test 查询。

等参数量视觉 adaptor、无布局监督 queries 或旧 oracle/pseudo-layout region-token adapter 属于结构消融，需要单独实现或确认已有结构开关后再跑；不能用 `ocr-only-adapter` 冒充等参数结构对照。

如果新 run 的 slot alignment 显示固定偏移，再运行 fixed offset sweep：

```bash
python3 tools/evaluation/analyze_layout_slot_offset_sweep.py \
  --comparison-root "$GOT_EVALUATION_RUNS/<comparison_run_id>" \
  --manifest "$GOT_LAYOUT_DATA/formal_pdf_short_seed20260812/test/manifest.jsonl"
```

该脚本同时输出 `slot_offset_page_oracle.csv`，用于判断是否存在页面级动态 offset。

对应命令、数据隔离要求和回传格式见 [SYNC_AND_RUN.md](../../docs/SYNC_AND_RUN.md)。
