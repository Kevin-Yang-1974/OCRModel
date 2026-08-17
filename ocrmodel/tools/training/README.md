# AncientDoc 训练协议

## 用户入口

```bash
bash tools/training/run_ancientdoc.sh prepare
bash tools/training/run_ancientdoc.sh train-core
bash tools/training/run_ancientdoc.sh select-c4
bash tools/training/run_ancientdoc.sh train-replay --c4-selection <selection.json>
```

旧 `run_ancientdoc.sh train` 已禁用，因为它会默认使用 C4-final 启动 replay。

## 分支定义

| 模型 | 起点 | 训练内容 |
|---|---|---|
| C0 | 原始 GOT2 | zero-shot reference，不训练 |
| C1 | 原始 GOT2 | AncientDoc `L_ocr` |
| C4 | synthetic P2 | AncientDoc `L_ocr`；周期 checkpoints 在 validation 选 C4-best |
| C5 | C4-best | AncientDoc＋synthetic OCR replay，3:1；从 C4-best 独立启动 |
| C6 | 与 C5 完全相同的 C4-best | AncientDoc＋带布局监督 synthetic replay，3:1；不经过 C5 |

默认冻结 `vision_tower_high`。C1 训练 decoder＋projector；C4/C5/C6 训练 decoder＋VLQA adapter＋projector。默认 AdamW、`2e-5`、cosine、warmup `0.03`、weight decay `0.01`、12000 optimizer steps、每 2000 steps 保存。

## C4 分支选择

先选择 C4-best 是创建下游 replay 分支的必要 validation，不是提前查看 test。固定排序键为 page CER、去空白 page CER、较早 step。C4 selection 保存 checkpoint/config/weights/trainer state/evaluator/manifest 哈希；周期 checkpoint 缺少根目录 `layout_training_metrics.json` 时，必须通过 `selection.json` 证明其父 C4 run、step 和权重，禁止静默回退 C4-final。

C5/C6 metrics、summary 和 suite JSONL 均记录 selected C4 step、model path、validation CER、selection 路径及 checkpoint hash；两者 optimizer/scheduler 从零开始。

## 公平性

- C5-C4：从同一 C4-best 分支点加入 OCR replay 的变化。
- C6-C4：从同一 C4-best 分支点加入带布局监督 replay 的整体变化。
- C6-C5：相同起点、数据源和采样比例下，比较 synthetic layout supervision。
- C1-C4：仍不是同参数量、同上游训练历史的严格结构对照，只能称为完整适配路线对照。

相同步数不等于相同训练预算。正式报告继续记录可训练参数、optimizer steps、样本/token 暴露、初始 checkpoint 和上游历史。完整命令见 `docs/SYNC_AND_RUN.md`。

统一结构消融入口为 `run_layout_ablation_suite.sh`。它支持 `got2_zero_shot`、`projector_only`、`generic_adapter_projector`、`vlqa_ocr_only`、`vlqa_layout_direct` 和 `vlqa_layout_p1_p2`，按组执行训练、validation checkpoint 选点和 selection-locked test。训练可用 `--gpu-id 2` 选择单张物理卡，或用 `--gpu-ids 0,2,3` 在一次 DeepSpeed 任务中进行数据并行多卡训练。`--parallel-gpu-ids 1,2,3,4` 则按 `--ablations` 的顺序把不同消融分别绑定到一张物理卡并发执行，例如 A2、A3、A4、A5 分别使用 GPU 1、2、3、4；每组后续 validation/test 继续使用自己的绑定卡。三种 GPU 模式互斥，所有目标卡必须在任何子任务启动前同时空闲。未指定 `--resume` 时，已有 run 或 test 会退出，避免重复正式评测。

selection 的 `--resume` 只复用通过模型路径、manifest、split、解码参数和 whole-page prompt-only 输入协议校验的既有 candidate summary；不一致时拒绝继续。指标兼容层把 evaluator 的 `character_edits`、`reference_characters`、`page_exact_matches` 规范化为 selection/test 的稳定总量字段。用 `tools/evaluation/run_layout_ablation_selection_smoke.sh <training-run-id> <dataset-id> <checkpoint-step>` 做 1 页断点续跑工程验证；该结果不能代替正式 validation 选点。
