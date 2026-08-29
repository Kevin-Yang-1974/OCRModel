# LAVP/PVLD Experiment Configuration Register

更新日期：2026-08-27

本文登记当前两条完全隔离的实验线：正在生成数据后的正式 P1/P2/P3 主线，以及仅用于工程和方向性验证的历史合成数据 P1/P2 pilot。本文是运行配置与结果解释的本地事实记录；不以 pilot 结果替代正式实验，不把任何 pilot checkpoint、selection 或 test 结论迁入正式主线。

## 1. 共用模型与输入契约

| 项目 | 固定配置 |
|---|---|
| OCR 主输入 | GOT2 原生 `whole-page image + OCR prompt` |
| Qwen OCR 视觉输入 | 原始 GOT2 256 个 16×16 visual tokens；不拼接 64×64 memory |
| 布局 memory | `layout_memory_resolution=64`：Vary ViT neck 后 `[B,256,64,64]`，仅进入 layout queries、PVLD decoder、coverage 和布局辅助头 |
| 不作为推理输入的字段 | bbox、writing direction、reading order；它们只用于训练监督、离线解释和评测真值 |
| 最大布局容量 | `max_regions=512`、`max_layout_records=512`、`max_layout_tokens=2048` |
| 随机种子 | `42` |
| GPU 准入 | 自动使用启动瞬间所有 `utilization.gpu < 50%` 的物理卡；达到阈值的卡不等待、不抢占；显式列表只查询该列表 |
| 分布式方式 | 原生 DeepSpeed ZeRO-2 默认；可显式切换 DDP。训练入口写入实际物理 GPU、准入模式和观测利用率 |

`64×64` memory 的已完成 A100 单步有界测量为：16×16 layout memory 约 `10487.59 MiB`、`0.443 steps/s`、vision gradient norm `0.01961`；64×64 layout memory 约 `10485.46 MiB`、`0.454 steps/s`、vision gradient norm `0.02281`。该测量只验证工程路径，不构成 OCR 或布局效果结论。

## 2. 参数组与损失

| 参数组 | P1 | P2 | P3 |
|---|---:|---:|---:|
| Vary ViT | `1e-6` | `5e-7` | `2e-7` |
| `mm_projector_vary` | `1e-5` | `5e-6` | `2e-6` |
| layout queries / PVLD decoder / layout heads | `1e-4` | `5e-5` | `1e-5` |
| Qwen decoder | frozen | `1e-6` | `5e-7` |
| residual gate | frozen at zero | `1e-5` | `1e-6` |
| OCR lm head | frozen，除非 GOT2 tied-head 加载路径要求同步更新 | frozen | frozen |

P1 解冻 vision encoder、projector 和 layout branch；Qwen、lm head 和 residual gate 冻结。P2/P3 解冻 vision、projector、layout、Qwen 和 residual gate，并以不同学习率优化。训练脚本在构造 `GOTTrainer` 前将这些 six-group rates 从 `LayoutTrainingArguments` 显式传给 `TrainingArguments`，以避免所有 optimizer group 回退到通用 `--learning_rate`。

| 阶段 | 损失 |
|---|---|
| P1 | `L_P1 = L_layout(primary + replay) + 0.25 * L_ocr(replay)` |
| P2 / P3 | `L_total = L_ocr + 1.0 * L_layout` |

`L_layout` 包含 layout token causal loss、REGION/EOS boundary、bbox、direction 和 count；coverage 只在 M2 启用；未经验证的 duplicate loss 不进入主损失。P1 replay 固定为 MTHv2 train 原始 whole-page GOT2 OCR 路径，`primary:replay=7:1`、`replay_ocr_loss_weight=0.25`。

训练指标必须记录 vision/projector gradient norm、parameter update norm、vision feature drift、replay supervised token count/replay OCR loss、residual gate、CER、layout F1、bbox IoU、EOS/count、吞吐和峰值显存。

## 3. 正式主线

| 项目 | 注册值 |
|---|---|
| 编排器 | `tools/training/run_lavp_p1_p3_formal_tmux.sh` |
| session / run prefix | `lavp_p1_p3_formal_20260827_v5` |
| 合成数据根 | `/data3/yky/yangky_ocr_models/training_data/got_layout_pages/ancient_photo_diverse_formal_s3s4_dense_20260827_v4` |
| 真实 replay / P3 数据根 | `/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1` |
| P1 / P2 / P3 steps | `12,000 / 30,000 / 8,000` |
| checkpoint interval | `2,000` |
| checkpoint retention | `16` |
| 正式数据硬门槛 | rendered train pages `>=10,000`；train `>32` regions 页面比例 `>=0.50`；`dataset_protocol.status=ready`；全量 audit `status=ok` |
| region exposure | 强制报告的覆盖/预算指标，不再作为一百万的启动阻断条件 |

正式链严格为：

```text
P1
-> P1 validation-only layout selection
-> P2 initialized from P1-selected checkpoint
-> P2 validation-only OCR selection
-> P3 initialized from P2-selected checkpoint
-> P3 validation-only OCR selection
-> selection-locked P2 Synthetic-ID test and P3 Real-OOD test
```

P3 selection 完成前不得运行任何正式 P2/P3 test。test 不得进入训练、选点、阈值、prompt、后处理或结构判断。正式编排器按 `legacy`、`m2`、`m3`、`m4`、`all` 顺序创建完全隔离的 `P1 -> P2 -> P3` run 根；每组 P1 均使用同一 Legacy P1 配置，但保留自己的 P1 validation-selected checkpoint。P2 接收该组完整的 M2/M3/M4 开关；P3 仅沿用 spatial memory 和 gradient-scale 配置，predicted-layout routing 保持 P2-only，避免把尚未重新验证的 routing 训练机制外推至域适配阶段。五组共享数据、steps、batch、seed、解冻范围、学习率、checkpoint interval 和 selection 规则，且不得交叉读取 checkpoint。M4 validation 额外要求 normal predicted-layout routing、`alpha=0` 和 shuffled evidence 三个条件。

对 M4/All，P2 validation selection 后、P3 前必须用该 selection 的 SHA-256 锁定 checkpoint 运行这三个 control。每项只读取 validation，且明确记录 `test_manifest_read=false`；normal page CER 必须严格低于两个消融 control，否则该组中止在 P2，不创建 P3 或 test。这个 gate 是 routing 因果性证据要求，不得由训练 loss、P1 指标或 test 结果替代。

正式主线当前只允许进行 S3/S4 数据生成及一次性全量 manifest/near-duplicate/source-leakage audit，不得以定时等待或轮询替代完成信号。三份 manifest 和全量审计通过后，先提交协议审计；只有用户随后明确确认并以新调用显式传入 `--confirm-formal-training`，才可创建正式 P1/P2/P3。任何 pilot run 均不得被主线读取。

## 4. 历史合成数据 Pilot

### 4.1 目的与隔离边界

| 项目 | 注册值 |
|---|---|
| 启动器 | `tools/training/run_historical_s3s4_p1_p2_pilot_tmux.sh` |
| 历史数据 | `ancient_photo_diverse_formal_s3s4_20260826_v1` |
| 用途 | 工程与方向性验证；不是正式主线数据、泛化结论或最终 test |
| 阶段 | P1 -> P1 validation-only selection -> P2 -> P2 validation-only selection |
| 禁止项 | 不运行 P3；不运行 test；不复用正式 v4/v5 数据、run、checkpoint、selection 或日志 |
| P1 / P2 steps | `2,000 / 5,000` |
| checkpoint interval | `1,000` |
| P1 replay | MTHv2 train；`7:1`；`0.25` |
| layout memory | `64`；Qwen 仍只读 256 个 16×16 OCR visual tokens |

pilot 使用独立的 `${GOT_TRAINING_RUNS}/<run_prefix>_p1`、`${GOT_TRAINING_RUNS}/<run_prefix>_p2` 和 `${GOT_TRAINING_RUNS}/<run_prefix>_pilot_logs`。启动器在创建训练前检查 train/validation/test manifest、source model、MTHv2 replay manifest、输出根不存在和 GPU 准入。历史数据本身只读；脚本没有删除、覆盖或恢复已有 run 的路径。

### 4.2 锁定 validation

pilot validation 来自历史数据 validation split，不读取 test。锁定算法按 `tier × region-count bucket × direction signature` 分层：每个 stratum 内按 `page_id` 排序，再以 stable round-robin 选取 256 页。锁定文件和元数据位于独立 pilot 根：

```text
validation/pilot_validation_256.jsonl
validation/pilot_validation_256.lock.json
```

`pilot_validation_256.lock.json` 必须含 `status=locked`、`selection_split=validation`、`page_count=256`、清单 SHA-256、tier/bucket/direction counts 和 `test_used_for_selection=false`。当前 v2 锁定清单 hash 为：

```text
678b76954b2077b59aade98e35896a6efa77952b6dd30810a8fac7daf6459823
```

锁定 v2 的计数为 S3 `105` 页、S4 `151` 页；region buckets 为 `1-8:81`、`9-16:60`、`17-32:50`、`33-64:31`、`65-128:25`、`>128:9`。该清单包含水平、垂直和混合方向 strata。

### 4.3 Pilot run 记录

| run prefix | 状态 | 说明 |
|---|---|---|
| `lavp_historical_s3s4_pilot_20260827_v1` | failed, preserved | 启动后远端 `bitsandbytes` 未找到 CUDA 11.8 `libcusparse.so.11`；没有产生 P1/P2 结果 |
| `lavp_historical_s3s4_pilot_20260827_v2` | stopped, preserved, diagnostic invalid | 经 `run_got2.sh` 修复 CUDA library path 后进入 P1；随后发现 optimizer group 实际全部使用通用 `1e-4`，不符合已注册的 six-group LR。经授权已停止 tmux session；目录、日志和锁定清单均保留，且不得用于 pilot 结论 |
| `lavp_historical_s3s4_pilot_20260827_v3` | failed, preserved | 分组学习率已正确为 vision/projector/layout=`1e-6/1e-5/1e-4`，但多卡 DDP 初始化因冻结参数的私有 DDP ignore 注册出现 rank 参数不一致，在首步前失败；不得用于结论 |
| `lavp_historical_s3s4_pilot_20260827_v4` | stopped, preserved | 移除私有 DDP ignore 后已通过 DDP 模型构造并持续满卡计算，但在有界启动窗口内未进入 `first_forward_start`；为避免无进展占卡，已停止并保留状态、日志与 validation lock |
| `lavp_historical_s3s4_pilot_20260827_v5` | stopped, preserved | `torchrun + Trainer --deepspeed` 在 optimizer/engine 构造后同样未进入首个 forward；已停止。runner 随后改为仓库其他 ZeRO-2 作业使用的原生 `deepspeed --num_gpus` launcher；v6 将独立验证该修复 |
| `lavp_historical_s3s4_pilot_20260827_v6` | stopped, preserved | 原生 DeepSpeed 已正确建立五个 rank，但首次 forward 前仍停滞；既有 A100 多卡脚本默认启用 `NCCL_P2P_DISABLE=1` 以绕过已知拓扑问题，v7 将以该配置独立验证 |
| `lavp_historical_s3s4_pilot_20260827_v7` | stopped, preserved | 原生 DeepSpeed 加 `NCCL_P2P_DISABLE=1` 后已真实运行超过 60 steps，证明多卡前向/反向可行；但当时 replay 指标仅反映 rank 0 本地 batch，不能证明全局 replay，因此停止并保留。v8 加入跨 rank replay audit |
| `lavp_historical_s3s4_pilot_20260827_v8` | P1 running, engineering-valid | 原生 DeepSpeed、`NCCL_P2P_DISABLE=1`、64×64 layout memory、五卡自动准入。已连续超过 50 steps，无 OOM/NCCL 错误；跨 rank audit 已观察到 replay global samples `1–3`、replay OCR tokens `255–1247`、replay-only OCR loss 为有限正值；vision gradient 和 vision/projector/layout update norm 均为有限正值，P1 residual gate 固定为 `0`。仅证明工程训练路径和方向性诊断可运行，不是正式数据结果 |

`lavp_p1_p3_formal_20260828_v9` 的 Legacy P1 训练已完成 12000 steps，随后进入全量 4000 页 P1 validation selection。按 2026-08-29 时间受限任务要求，该 tmux session 已停止并保留全部产物。现有周期 checkpoint 为 `2000/4000/6000/8000/10000/12000`；注册的新 P1 候选 `6000/9000/12000` 中，`checkpoint-9000` 不存在，因此新的 400 页验证入口当前不会启动 P2。该缺口必须由用户确认候选变更解决，不重训 P1，也不以相邻 checkpoint 代替。

v3-v8 都不是对 v2 的恢复，均从原始 GOT2 重新开始 P1，并各自生成独立写入其 run 根的 validation lock。任何 pilot 的 P2 只能由同一 run prefix 的 P1 validation-selected checkpoint 初始化；当前若 v8 完成 P1，只有 v8 的 selected checkpoint 可以初始化 v8 的 P2。

## 5. 当前禁止与报告口径

### 5.1 BSCC 历史 M2-M4 结果归档（2026-08-27）

这一组使用 BSCC MTHv2 whole-page 数据 `train/validation/test=2159/240/800`，不是本文件第 3 节正式 S3/S4 主线。以下数值来自只读的远端 `selection.json` 和 selection-locked test `summary.json`；不得用于当前高难度合成数据主线的 checkpoint、阈值或结论。

| 条件 | P2 选点 | validation page CER | locked test page CER | locked test whitespace CER | locked test complete F1 | count MAE | matched bbox IoU | duplicate rate | 协议状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M2 `mthv2_pvld_m2_spatial_20260826_v1` | 7,500 | 0.932514 | 0.970107 | 0.952731 | 0.231005 | 6.7375 | 0.630271 | 0.183975 | validation-only selection；800 页 selection-locked test；0 inference failures |
| M3 `mthv2_pvld_m3_gradscale_20260826_v1` | 7,500 | 0.997383 | 0.956669 | 0.937613 | 0.249716 | 6.6675 | 0.652170 | 0.183459 | validation-only selection；800 页 selection-locked test；0 inference failures |
| M4 `mthv2_pvld_m4_predrouting_20260826_v1` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | P2 first forward failed; no P2 selection, routing controls, or test |

M2 P1 从 validation 选中的 step 2,000 初始化，M3 P1 从 validation 选中的 step 3,000 初始化；两者 P2 的 selection 文件均明确记录 `selection_split=validation` 与 `test_used_for_selection=false`。M4 的失败为 routed linear input `Float` 与权重 `BFloat16` 不匹配（所有 rank 同类失败），不是负面性能观察，也不触发 validation 或 test。

重要配置偏差：BSCC M3 的实际保存配置同时为 `pvld_use_spatial_memory=true`、`pvld_shared_gradient_scale_p2=pvld_record_gradient_scale_p2=0.25`。它检验的是 spatial-memory 基础上的梯度缩放，而非本文件预注册矩阵中 memory 关闭的纯 M3；因此这批 M2/M3 仅支持受限的条件比较，不能替代 Legacy/M2/M3/M4/All 完整消融，也不能与不同硬件/代码快照的 A100 M1 结果直接作因果排名。

- 不把 v1/v2/v3 pilot 的 checkpoint、selection、指标或阈值用于正式 v5。
- 不把 pilot validation 的 CER、layout F1、IoU 或吞吐描述为正式实验、最终 test 或跨域泛化结论。
- 不运行 test，直到对应正式 validation-only selection 完成并由 selection 文件锁定；pilot 永远不运行 test。
- 不删除、覆盖或重启已存在的 v1/v2 pilot 目录、历史 run、正式 run、checkpoint、日志或 test 产物。
- 正式数据审计只接受 S3/S4 v4 的 train/validation/test 三份完整 manifest；历史 v1 不可替代。

## 6. 实现与验证索引

| 事项 | 文件 |
|---|---|
| 训练参数、视觉梯度与诊断 | `src/GOT-OCR-2.0/scripts/train_GOT_layout.py` |
| 参数组 optimizer | `src/GOT-OCR-2.0/GOT/train/trainer_vit_fixlr.py` |
| GPU 准入、P1 selection、阶段 runner | `tools/training/run_variable_layout_a100.py` |
| 正式主线编排 | `tools/training/run_lavp_p1_p3_formal_tmux.sh` |
| 历史 pilot 编排 | `tools/training/run_historical_s3s4_p1_p2_pilot_tmux.sh` |
| selection evaluator | `tools/evaluation/select_layout_ablation_checkpoint.py` |
| pilot 静态契约 | `tests/test_historical_pilot_contract.py` |
