# GLMOCR 全量 MTHv2 五卡同步训练

## 结论口径

确定性 A100 架构对照的 10/10 任务、63 个 checkpoint finite，`geometry@768` 三 seed 的 validation CER 均值为 `0.189777`，选中点最大 residual norm 为 `0.00815`。这支持“工程与优化意义上的初步稳定”，但不等于泛化稳定：该结果只使用 128 页 train、64 页 validation，test 未进入正式流程，且未选中的后程点出现过 `0.01142` 的轻微超阈值。

全量 MTHv2 后续诊断已经说明：`glmocr_mthv2_full_ddp_v1` 的学习率在 step 432 后长期停在 `5e-6`，因此停止作学习率诊断；gate warm-start 诊断未形成正式训练依据；`no_assignment` 使辅助项下降但没有带来 OCR loss 下降；旧 validity/no-object head 后，`p_valid` 主要表现为整体偏置上升，valid/no-object 分离和 gated fusion 均未达标。因此当前不能把旧 validity gating 写成已验证架构收益，也不能据此直接进入全量三 seed 正式训练。

本协议仍作为全量 MTHv2 的数据、DDP 和 selection/test 边界说明。MTHv2 官方 split 不提供完整书手、版本或馆藏元数据，结果不得表述为严格跨书手泛化验证。

## 固定数据与配置

A100 远端官方 manifest 为：

| split | 页数 |
|---|---:|
| train | 2159 |
| validation | 240 |
| test | 800 |

预检同时记录每页相对图像路径及 SHA-256、manifest SHA-256、最大区域数 `407` 和最大页面文本长度 `1310`；正式入口固定 `num_queries=512`，不再以 32 queries 过滤页面。

训练固定为 `geometry`、Hungarian assignment、`full` layout loss、auxiliary weight `0.2`、冻结 BF16 backbone、FP32 adapter/transport/loss、fast processor + torchvision、math SDP、residual cap `±0.03`。学习率为 `5e-5 → 5e-6`，weight decay `0.01`，max grad norm `1.0`。

每卡 micro-batch 为 `1`，gradient accumulation 为 `1`，五卡同步 DDP 的 global batch 为 `5`。每 epoch 为 `ceil(2159/5)=432` 次更新，总计 `3456` steps（8 epochs）；warmup `216` steps，LR schedule horizon 改为完整的 `3456` steps；在 `432, 864, ..., 3456` 保存 checkpoint 并执行 validation。生成上限为 `1536` tokens，seed 为 `42, 43, 44`。

首个全量 run `glmocr_mthv2_full_ddp_v1` 的诊断显示，原 `lr_schedule_steps=432` 使学习率在第 432 步后长期固定为 `5e-6`；step 1040 时仍为该终值，训练 loss 进入平台期。因此后续 run 保留已验证的峰值 `5e-5`，仅将调度 horizon 延长到完整训练步数，并使用新的 run ID，保留 v1 作为诊断对照。

由于 v1 从早期 step 起 OCR loss 就没有明显下降，另行进行 gate warm-start 诊断：保持 seed42、geometry、Hungarian、full layout loss、`auxiliary_weight=0.2`、`max_steps=432` 和原 `lr_schedule_steps=432` 不变，仅将 `initial_residual_scale` 设为 `0.01`（上限仍为 `0.03`）。该诊断用于区分零门控初始化造成的 OCR 梯度通路抑制与学习率调度问题，不参与正式三 seed 结果或 test 选点。该诊断随后按用户要求停止，不能作为“gate warm-start 已解决”的证据。

## 近期全量 MTHv2 诊断记录

下表只记录 train/validation 侧诊断；这些 run 均不执行 selection，test 不得参与训练或选点。

| run_id | 关键配置 | 终态/进度 | 主要观察 | 结论 |
| --- | --- | --- | --- | --- |
| `glmocr_mthv2_full_ddp_v1` | geometry、full、512 queries、五卡、global batch 5 | stopped_for_lr_diagnosis；指标到 step 1072 | step 432 后 LR 固定为 `5e-6`，训练进入平台期 | 保留为调度诊断对照，不作正式结果 |
| `glmocr_mthv2_full_ddp_v2` | gate 诊断前置 run | stopped_for_gate_diagnostic；指标到 step 64 | OCR/auxiliary 仍有明显页面级波动 | 不作正式结果 |
| `glmocr_mthv2_gatewarm_diag_v1` | 初始 residual scale `0.01`、432 steps | stopped_by_user；记录到 step 432 | step 432 OCR `1.1234`、total `1.8647`，但未完成 validation 证据链 | 仅作 gate warm-start 诊断 |
| `glmocr_mthv2_no_assignment_256_v4_r2` | no_assignment、五卡、effective global batch 20、256 steps | 指标到 step 256，随后 stopped_by_user | OCR loss 前/后 64 步中位数约 `1.272→1.366`，无下降趋势；assignment 已从 total 中移除 | assignment 不是唯一直接根因；不作 selection |
| `glmocr_mthv2_validity_no_assignment_256_v1` | no_assignment_validity、validity head、五卡、effective global batch 20、256 steps | 指标到 step 256；完整 validation 按用户要求停止；状态为 `stopped_by_user`、`validation_stop_only=true` | OCR 中位数约 `1.3856→1.4068`；后程 valid/no-object `p_valid` 差值约 `0.0101`；gated invalid fusion mass 约 `0.946` | 代码路径可运行，但 validity gating 尚未证明有效 |
| `glmocr_mthv2_validity_assignment_256_v1` | validity_assignment、raw-mass gate、detached transport evidence、Hungarian region support、五卡、seed42、256 steps；全量 train＋32 页 validation 子集 | wrapper 在训练启动前因同步后的执行位问题退出；未产生训练指标 | 不作结果；已用新 run ID 重试 | 不读取 test |
| `glmocr_mthv2_validity_assignment_256_v2` | validity_assignment、raw-mass gate、detached transport evidence、Hungarian region support、五卡、seed42、256 steps；全量 train＋32 页 validation 子集 | `a100-yky` smoke 通过；bounded 机制验证运行中 | 待验证 step128/256 的 p gap、AUROC/AP、invalid context share、OCR/CER | 未通过 query-level 阈值前不扩展 seed43/44，不执行 selection-locked test |

前一版 `glmocr_mthv2_no_assignment_256_v4` 因 `test_manifest_read=true` 的协议字段错误标为 failed；其产物保留，后续使用 `v4_r2` 修正字段，不覆盖旧 run。

Validity 分支的实现位置为 `adapter.py` 的 `validity_head`、detached transport evidence、legacy/raw-mass gated transport，`losses.py` 的 object/cardinality/ranking/assignment 损失，以及 `train_screen.py` 的 query-level 诊断指标。新 profile 仍是待验证候选，默认架构和正式三 seed 协议暂不切换。

## 入口与数据边界

- `tools/audit_mthv2_manifest.py`：联合审计 train/validation/test、相对图像路径、页数、区域数、文本长度和 manifest hash，并生成 protocol JSON。
- `tools/training/run_glmocr_mthv2_ddp.sh`：真正的五进程 `torchrun` DDP；每个 rank 处理一个 whole-page，DDP 只包装 adapter；不足整除时按固定规则补齐，不丢页。
- `tools/training/run_glmocr_mthv2_validity_assignment_256.sh`：运行 seed42、256-step、前 64 steps gate freeze 的 validity 机制验证；训练使用全量 train，默认自动生成 32 页 validation 子集和独立 protocol，显式 `without-test`、`skip-selection`。
- `tools/smoke_glmocr_ddp.py`：8-step 五卡 smoke，使用最高区域数页面覆盖 512-query、全 rank forward/backward、finite 检查和 checkpoint reload。
- `tools/summarize_mthv2_full.py`：三 seed 共同 validation-only 选点，输出根目录 `selection.json`；test 不参与选点。
- `tools/evaluate_glmocr_locked_test.py`：读取共同 `selection.json` 后独立评估 800 页 test，写入每个 seed 的 `locked-test/locked_test_summary.json`。
- `tools/training/run_glmocr_mthv2_locked_test.sh`：上述 locked-test 的 A100 包装入口。

旧的 `run_glmocr_a100_5gpu.sh` 仍是独立单卡架构对照入口，不得当作本协议的联合训练入口。

## 执行顺序

1. 同步活动 `src`、`tools`、`config`，并运行 `run_glmocr_mthv2_ddp.sh --smoke --foreground`。smoke 失败时保留失败目录，使用新 run ID 重试。
2. 使用基础 GLM-OCR 权重运行 `tools/training/run_glmocr_mthv2_validity_assignment_256.sh`；入口默认在 32 页 validation 子集上做 bounded validation，不得从 `glmocr_mthv2_validity_no_assignment_256_v1` 的退化 validity checkpoint 续训。
3. 在 validity/query target 诊断通过前，不启动正式三 seed 全量训练；任何新诊断都使用新的 run ID，并保留失败或停止产物。
4. 只有候选架构在 seed42 的 bounded validation 机制检查通过后，才依次启动 seed `42`、`43`、`44` 的正式五卡 DDP run。
5. 三个正式 run 均 complete 后，运行 `summarize_mthv2_full.py`，按三 seed 平均 validation CER 锁定一个共同 step。
6. 使用共同 step 分别运行三次 `run_glmocr_mthv2_locked_test.sh`。test 只在此阶段读取一次，不参与训练、早停、阈值或后处理调整。

稳定性单独判定：三 run 无 CUDA/OOM/NaN/Inf/Traceback；共同选中点 residual norm `≤0.01`；generation limit rate `<0.10`；且不存在所有 seed 同时发生的后程灾难性 CER 退化。性能收益与稳定性分开报告。

## 历史：BSCC 四卡 decoder-LoRA 变体

为验证 decoder LoRA 学习率 `1e-5`，新增 BSCC 四卡变体。该变体沿用本页的 whole-page、512-query、geometry、Hungarian、FP32 adapter 和 fast processor 配置，但使用四卡同步 DDP（global batch 4）、20,000 steps、完整 20,000-step LR horizon，并从基础 GLM-OCR revision 重新初始化，不续接旧 32-query checkpoint。本次 official-layout run 的训练目标固定为 `L_official + 0.2 L_layout`，关闭 `L_natural_loop`、scheduled sampling、loop escape 和 continuation head；`generation_mode=loop_recovery` 仅保留给验证/推理解码，不能代替训练信号。

公共训练代码仍保留 natural-loop 参数和旧诊断模式，以便复现历史证据，但当前 BSCC 入口不启用这些分支；本轮结果只按上述 plain training objective 解释。

训练入口 `tools/bscc/run_glmocr_mthv2_decoder_lora_4gpu.sbatch` 只审计 train/validation 并保存 step `5000/10000/15000/20000`，使用 `--defer-validation` 避免训练进程串行选点。`tools/bscc/run_glmocr_mthv2_parallel_validation_4gpu.sbatch` 将四个 checkpoint 分配到四张 GPU，按 validation CER 选择一个 step；`tools/bscc/run_glmocr_mthv2_locked_test_4gpu.sbatch` 在 selection 完成后才生成 test protocol 并执行四路分片 test。该变体的 test 仍为 selection-locked，且 `test_used_for_selection=false`。

四卡变体原计划用于高 decoder-LoRA 学习率探索，但在未分配节点、未产生训练产物前切换到 A100 五卡入口；对应 BSCC pending job 不作为结果。此前的训练期循环损失 run 已停止并排除，上一轮 synthetic loop-escape smoke 不进入当前实验结果。

## 当前：A100 五卡 decoder-LoRA plain training

当前正式 run 为 `glmocr_mthv2_decoder_lora_lr1e5_20k_5gpu_a100_official_layout_260912_v1`，使用 `tools/training/run_glmocr_a100_decoder_lora.sh`。该入口与 BSCC 版本共享相同的训练代码、whole-page/512-query/geometry/Hungarian/full layout loss、FP32 adapter、fast processor、decoder LoRA `rank/alpha/dropout=8/8/0`、decoder LoRA learning rate `1e-5` 及 plain objective `L_official + 0.2 L_layout`；差异是 A100 使用五卡同步 DDP（global batch 5），BSCC 计划使用四卡同步 DDP（global batch 4）。

A100 launcher 先执行 bounded smoke，再执行 20,000-step deferred training，固定在 step `5000/10000/15000/20000` 保存 checkpoint 并进行 validation-only selection，最后才执行 selection-locked test。训练和 validation 均不读取 test；`generation_mode=loop_recovery` 仅是验证/测试解码配置，不会加入训练损失。当前 run 已完成启动前 GPU 准入并进入 smoke，后续阶段以 `pipeline_status.json`、metrics 和 checkpoint 健康状态为准。
