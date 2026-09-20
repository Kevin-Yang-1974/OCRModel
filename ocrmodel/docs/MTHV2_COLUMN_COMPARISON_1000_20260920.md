# MTHv2 GT 列级样本对照结果（1000 samples）

## 口径

- 数据：MTHv2 `test` split 中固定均匀抽取的 1000 个 GT 裁剪纵向列／line images；原 test manifest 共 25,262 个样本。
- 参考文本总长度：10,294 字符。
- 三个新评估均使用同一份 `manifest_subset_1000.jsonl`、greedy decoding 和 `max_new_tokens=256`。
- 这里的“列级”是根据 MTHv2 标注裁剪得到的纵向列图；它评估的是识别能力，不评估整页分列能力。MTHv2 的 `label_textline` 本身是有序区域候选，不能把本结果写成整页 column detection 结果。
- GOT 和 GLM-OCR 本次均为原始权重 zero-shot，没有在 MTHv2 上训练。AnandaSky baseline 也是原始 HF 权重；改进方法沿用前一轮已经完成的 MTHv2 GT-line 训练和 validation-only 选点。
- 四组的 `test_used_for_selection=false`。

## 结果

| 方法 | MTHv2 训练 | 字符错误 | 插入 | 删除 | 替换 | CER ↓ | 整列完全正确率 ↑ | K≤1 recall | K≤3 recall | K≤5 recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GOT-OCR2.0 original | 否，zero-shot | 2258 | 78 | 451 | 1729 | 0.219351 | 0.164 | 0.120000 | 0.210526 | 0.206897 |
| GLM-OCR original | 否，zero-shot | 1093 | 54 | 58 | 981 | 0.106178 | 0.408 | 0.280000 | 0.456140 | 0.505747 |
| AnandaSky baseline | 否，原始 HF 权重 | 760 | 27 | 10 | 723 | 0.073829 | 0.560 | 0.560000 | 0.649123 | 0.655172 |
| AnandaSky + 改进方法 | 是，GT-line layout 3000 steps + semantic 2000 steps | 755 | 26 | 10 | 719 | **0.073344** | 0.560 | 0.560000 | 0.649123 | 0.655172 |

## 结论

1. 在这份 GT 裁剪列图上，原始 GLM-OCR 已明显优于 GOT：CER 从 `0.219351` 降到 `0.106178`，相对降低约 `51.6%`。
2. AnandaSky 原始权重进一步低于 GLM-OCR，CER 为 `0.073829`；因此该列级结果不能解释为“改进方法单独带来的收益”，AnandaSky 本身就是更强的识别基线。
3. 当前改进方法相对 AnandaSky baseline 只减少 5 个字符错误，CER 绝对下降 `0.000486`、相对下降约 `0.66%`；整列完全正确率和低频字符 recall 没有变化。当前证据更适合表述为“小幅改善”，不能写成显著结构收益。
4. 改进方法的 checkpoint-2000 是在 validation CER 上从 semantic-1000/2000 中选出的，test 子集只用于最终锁定评估，没有参与选点。

## 改进方法的实际训练内容

- layout 阶段：冻结 AnandaSky 基础模型，只训练 layout adapter 的 query、geometry／box、order、direction、validity 等分支，3000 steps。
- semantic 阶段：冻结基础模型和 layout 分支，只训练 `sem_adapter` 两层 MLP 与 `content_gate`，2000 steps。
- 选点：validation CER 选择 step-2000；对应 validation CER 为 `0.08490566037735849`。

## 复现实验产物

- evaluator：`tools/evaluation/evaluate_mthv2_columns.py`
- BSCC launcher：`tools/bscc/run_mthv2_column_compare.sbatch`
- GOT summary：`$HOME/yangky_ocr_models_bscc_proto/runs/mthv2_column_compare_1000_260920_v3/formal/got/summary.json`
- GLM summary：`$HOME/yangky_ocr_models_bscc_proto/runs/mthv2_column_compare_1000_260920_v3/formal/glm/summary.json`
- Ananda locked-test summaries：`$HOME/yangky_ocr_models_bscc_proto/anandasky_line_layout/runs/anandasky_mthv2_gtline_locked_test_1000_260920_v2/locked_test/{baseline,method}/summary.json`
