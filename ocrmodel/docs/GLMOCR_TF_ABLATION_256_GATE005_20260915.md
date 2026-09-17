# GLM-OCR teacher-forcing 三模式消融（2026-09-15）

## 目的

在敦煌／地方志 q32 whole-page 数据上，使用纯 teacher-forcing 目标，直接比较 `content_only`、`attention` 和 `geometry`。本轮用于诊断布局残差是否导致自由生成退化，不追求复现历史 validation 数值。

## 固定协议

| 项目 | 配置 |
| --- | --- |
| 数据 | `dunhuang_local_gazetteer_q32_v1`；相同 train／validation／test manifest |
| 模型起点 | GLM-OCR revision `ca5d8b3e287e52589e37c28385d9655ee4372f9d`；三组均从基础权重重新开始 |
| 输入 | whole-page image + OCR prompt；32 layout queries |
| 训练目标 | `L_official + auxiliary_weight * L_layout`；不启用 free-generation、scheduled sampling、loop escape 或 continuation head |
| 训练 | 五卡同步 DDP；seed `42`；256 optimizer steps；LR horizon `256`；warmup `216`；max grad norm `1.0` |
| 学习率 | adapter `2.5e-5`；decoder LoRA `5e-6`；LoRA rank／alpha `8/8`；dropout `0` |
| gate | `initial_residual_scale=0.005`；effective residual cap `0.03` |
| 生成 | plain；最多 `1536` new tokens；fast processor；Hungarian matching；full layout loss |

`content_only` 按官方 baseline 关闭 layout loss（`auxiliary_weight=0`），其 gate 参数只写入共同元数据，实际融合路径仍为 identity；`attention` 和 `geometry` 使用 `auxiliary_weight=0.4`。

## 执行顺序

1. 先完成三个 mode 各自 8-step smoke，全部通过后才开始正式训练。
2. 三组正式训练串行使用同五张 A100，分别只保存最终 `checkpoint-256`，训练协议不读取 test。
3. 不做 validation 选点；训练结束后将 `checkpoint-256` 固定为 direct-test checkpoint，三组各完整执行一次 held-out test。
4. 结果只作为 direct-test diagnostic，不替代 validation-selection → locked-test 的正式最终结果。

## Run IDs

| mode | run ID | test |
| --- | --- | --- |
| `content_only` | `glmocr_q32_tf_content_only_gate005_256_a100_260915_v1` | 待完成 |
| `attention` | `glmocr_q32_tf_attention_gate005_256_a100_260915_v1` | 待完成 |
| `geometry` | `glmocr_q32_tf_geometry_gate005_256_a100_260915_v1` | 待完成 |

统一启动器：`tools/training/run_dunhuang_local_q32_tf_ablation_a100.sh`。

## 指标

Test 完成后记录 micro CER、去空白 CER、macro CER、mean NED、exact match、平均编辑距离、插入／删除／替换、EOS 命中率、触顶率、循环页率和 repeated-trigram 诊断，并保留每组的 `summary.json` 与 `locked_test_summary.json`。
