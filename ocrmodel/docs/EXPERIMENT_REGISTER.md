# 实验登记表

## 固定协议

| 字段 | 当前口径 |
| --- | --- |
| 主任务 | 小样本多场景通用符号识别 |
| 输入 | 整页图像＋prompt |
| 布局真值 | 仅训练辅助监督与评测，不进入推理 |
| R1 | 领域级少样本，限制标注页预算 |
| R2 | 稀有符号级 K-shot |
| 机制筛选 | 128 页，seed 42 |
| 隔离单元 | 书手、版本、馆藏或符号类型＋近重复组 |
| 选点 | validation only |
| 测试 | selection-locked test |
| 对照 | content-only、attention、geometry、layout_ot |
| 全量 MTHv2 协议 | train 2159 / validation 240 / test 800 页；whole-page；512 queries；五卡同步 DDP |
| 当前 validity 候选 | `validity_assignment`：Hungarian＋detached raw-transport region support；object BCE＋cardinality＋ranking；`log T + log p_valid` assignment；raw-mass gate |

## 后续统一默认超参数

参数优选后的正式 run 统一使用：主 adapter LR `2.5e-5`、decoder LoRA LR `5e-6`、`auxiliary_weight=0.4`、warmup `216`、max grad norm `1.0`，训练目标为 `L_official + 0.4 L_layout`。natural-loop、scheduled sampling、loop escape 和 continuation head 均关闭。A100-yky 与 BSCC 分别使用独立的新 run ID；本轮 gate=`0.005` 跨域微调是按用户指定的较小 LR `1e-5`／`2e-6` 执行的例外；下面已完成 run 的旧参数和结果不作回写。

## 运行记录

| ID | 配置 | 数据指纹 | checkpoint 起点 | 状态 | validation | test | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1` | Q32 layout-only；`history_box_equalized_v2`（box `820`、assignment `1`、order/direction `0.5`）；seed42；五卡；3000-step continuation | MTHv2 sparse24 train/validation/test manifest | `glmocr_mthv2_sparse24_q32_layout_boxeq58_3000_a100_260916_v1/seed42/checkpoint-3000` | complete；checkpoint finite | step `3000`；IoU `0.456072`；MAE `0.029717` | step `3000`；IoU `0.446998`；MAE `0.030560`；CER `0.393093` | box 权重按训练历史调至与 assignment 同量级；作为后续敦煌／地方志 Q32 微调的共同起点；`test_used_for_selection=false` |
| `glmocr_dunhuang_local_q32_gate005_attn_geom_260916_v1` | `attention`／`geometry`；Q32；`history_box_equalized_v2`；残差 gate `0.005`；256 steps；五卡；seed42 | `dunhuang_local_gazetteer_q32_v1_portable` | 上一行 run 的 `seed42/checkpoint-3000`；同时加载 `adapter.safetensors` 与 `decoder_lora.safetensors` | complete；两臂均完成 smoke、训练、validation-only selection 和 locked test | 两臂均只评估 step `256`；validation 选点未读取 test | attention：IoU `0.271496`、CER `0.173582`；geometry：IoU `0.286134`、CER `0.173371` | 主／decoder LR `1e-5`／`2e-6`；attention 复用 geometry adapter（仅放宽 mode 字段）；两臂 EOS `1.0`、触顶／循环均为 `0`；`test_used_for_selection=false` |
| `glmocr_dunhuang_local_q32_boxeq820_gate005_lr1e5_content_only_256_frommthv2_260916_v1` | `content_only`；Q32；同一 `history_box_equalized_v2`；残差 gate `0.005`；256 steps；五卡；seed42 | `dunhuang_local_gazetteer_q32_v1_portable` | 同一 source `seed42/checkpoint-3000`；同时加载 `adapter.safetensors` 与 `decoder_lora.safetensors` | complete；smoke、训练、validation-only selection 和 locked test 均完成 | step `256`；CER `0.209867`；layout IoU `0.309260`；MAE `0.061531` | 59 页；IoU `0.263915`；MAE `0.068268`；CER `0.173582` | 主／decoder LR `1e-5`／`2e-6`；`auxiliary_weight=0`，融合保持 identity；EOS `1.0`、触顶／循环均为 `0`；`test_used_for_selection=false` |
| `mechanism_screen_128_seed42_20260907_v1` | `configs/mechanism_screen_128.toml` | BSCC setup 时生成并锁定 | GLM-OCR `ca5d8b3` | setup `1480858`、screen `1480859_[0-3]` 均完成；`selection.json` 已生成 | geometry；validation CER `0.155984` | 不运行 | 256 steps 机制筛选：content_only `0.171386`、attention `0.156192`、geometry `0.155984`、layout_ot `0.162466`；R2 与布局指标见各 run `summary.json`；首次 setup `1480589` 因计算节点无外网失败，失败记录保留 |
| `mechanism_confirm_128_3seed_v1` | 同一锁定协议；attention/geometry × auxiliary off/on | 同上 | 同上 | 12 个 run 已完成 | 两种 `aux0.2` 模式的三种子均在 step 256 最优；attention 平均 CER `0.2251`，geometry `0.2045` | 未读取 | step 1024 平均 CER 分别退化至 `1.3745`、`1.2408`；gate 绝对值增至约 `0.085–0.089`，生成触顶率增至约 `35%–47%`。训练 loss 下降且未出现 NaN，当前判断为残差扰动累积与解码漂移 |
| `mechanism_stable_128_3seed_v1` | `configs/mechanism_stable_128.toml`；attention/geometry × `auxiliary_weight=0.2` × 三种子，另含 content-only eval-only 基线 | 复用同一锁定协议 | 同上 | Slurm array `1481332` 的 7 个 task 均完成，退出码 `0:0`；`selection.json` 已生成 | validation-only 选择 geometry、step `256`；平均 CER `0.206043`，标准差 `0.000024`；content-only 基线 CER `0.208411` | 未读取；稳定性验收失败，不进入正式 test | 64-step warmup、峰值 `5e-5`、cosine 至 `0.1×`；有效 residual scale 限制 `±0.03`。step 1024 两种模式 CER 均回退约 `0.185`，生成触顶率 `0.109375` 超过 `0.10`；attention/seed44/step768 CER `0.520120` 超过 `0.5`；`test_used_for_selection=false`，`eligible_for_formal_test=false` |
| `architecture_1024_hungarian_fp32_lr128_det_procfast_a100_5gpu_v1` | geometry/attention/layout_ot × `42/43/44`；A100 五卡入口；1024 steps | 固定小样本协议 | GLM-OCR 固定 revision | 10/10 task complete；63 个 checkpoint finite | validation-only 选择 geometry@768，三 seed 平均 CER `0.189777`；选中点 residual 最大 `0.00815`，未选后程点最高 `0.01142` | 未读取 test | 作为确定性小样本架构对照，不等同全量 MTHv2 泛化 |
| `glmocr_mthv2_full_ddp_v1` | geometry、full、512 queries、五卡同步 DDP、global batch 5 | 全量 MTHv2 manifest | GLM-OCR 固定 revision | stopped_for_lr_diagnosis；指标到 step 1072 | step 432 后 LR 固定为 `5e-6`，OCR loss 进入平台期 | 不读取 | 保留为调度诊断对照，不选点 |
| `glmocr_mthv2_full_ddp_v2` | gate 诊断前置 run | 全量 MTHv2 manifest | 同上 | stopped_for_gate_diagnostic；指标到 step 64 | 未形成正式 validation 证据 | 不读取 | 不选点 |
| `glmocr_mthv2_gatewarm_diag_v1` | initial residual scale `0.01`、432 steps、seed42 | 全量 MTHv2 manifest | 同上 | stopped_by_user；记录到 step 432 | step 432 OCR `1.1234`、total `1.8647`；未完成 validation 证据链 | 不读取 | 仅作 gate warm-start 诊断 |
| `glmocr_mthv2_no_assignment_256_v4_r2` | `no_assignment`；五卡；effective global batch 20；256 steps；无 selection | 全量 MTHv2 manifest | 同上 | 训练指标到 step 256，随后 stopped_by_user | OCR loss 前/后 64 步中位数约 `1.272→1.366`；auxiliary 下降但 OCR 未改善 | 不读取 | assignment 从 total 移除后仍未解决 OCR 平台 |
| `glmocr_mthv2_validity_no_assignment_256_v1` | `no_assignment_validity`；validity head；五卡；effective global batch 20；256 steps；无 selection | 全量 MTHv2 manifest | 同上 | 指标到 step 256；完整 validation 按用户要求停止；状态 `stopped_by_user`、`validation_stop_only=true` | OCR 中位数约 `1.3856→1.4068`；后程 valid/no-object `p_valid` 差值约 `0.0101`；gated invalid fusion mass 约 `0.946` | 不读取 | 实现可运行但未证明有效；不扩展 seed |
| `glmocr_mthv2_validity_assignment_256_v1` | `validity_assignment`；raw-mass gating；detached transport evidence；Hungarian region support；五卡；seed42；256 steps；全量 train＋32 页 validation 子集；无 selection | 全量 MTHv2 train manifest；validation 为确定性小子集 | GLM-OCR 固定 revision；从基础权重重新开始 | wrapper 在训练启动前因同步后的执行位问题退出，未产生训练指标；产物保留 | 无训练结果；已用新 run ID 重试 | 不读取 | 不作结果；修复 wrapper 后使用 v2 重试 |
| `glmocr_mthv2_validity_assignment_256_v2` | `validity_assignment`；raw-mass gating；detached transport evidence；Hungarian region support；五卡；seed42；256 steps；全量 train＋32 页 validation 子集；无 selection | 全量 MTHv2 train manifest；validation 为确定性小子集 | GLM-OCR 固定 revision；从基础权重重新开始 | `a100-yky` smoke 通过；bounded 256-step 机制验证运行中 | 待验证：step128/256 query-level p gap、AUROC/AP、invalid context share、OCR/CER | 不读取 | 未通过 query-level 机制阈值前不扩展 seed43/44，不执行 selection-locked test |
| `glmocr_natural_loop_A1_1024_260911_v1` | natural predicted-loop loss；`λ=0.05`；plain generation；无硬 EOS/循环 guard；LoRA；1024 steps；no-validation | 全量 MTHv2 train manifest；固定 800 页 test manifest 仅在训练完成后读取 | GLM-OCR 固定 revision；基础权重；seed42 | complete；step1024 checkpoint finite；fixed-final selection；训练不做 validation | 无 validation；固定 final step1024 | 已完成 direct test：CER `0.874635`；test 未参与选点 | 训练 mean official base loss `1.409982`、mean total `2.422198`；natural-loop active ratio `0`；test 结果见 `GLMOCR-B-260911-002.md` |
| `glmocr_plain_baseline_1024_260911_v1` | plain baseline；natural loop disabled；plain generation；无硬 EOS/循环 guard；LoRA；1024 steps；no-validation | 全量 MTHv2 train manifest；固定 800 页 test manifest | GLM-OCR 固定 revision；基础权重；seed42 | complete；step1024 checkpoint finite；fixed-final selection；训练不做 validation | 无 validation；固定 final step1024 | 五卡 direct test 已完成；CER `0.874635`；循环页率 `0.195000`；长度上限率 `0.207500` | 训练 mean official base loss `1.409982`、mean total `2.422198`；与 A1 的 checkpoint 和 test 指标完全一致 |
| `glmocr_mthv2_attribution_128_retry_v1` | seed42 三组容量归因：A `no-op`；B 固定 gate `adapter-only`；C 固定 gate＋decoder-LoRA（rank 8、alpha 8、dropout 0）；五卡 DDP；128 steps；32 页 validation；无 selection | 同一 MTHv2 train manifest；同一确定性 validation32 manifest（32 页） | GLM-OCR 固定 revision；三组均从同一基础权重起点；B/C 训练预算一致 | bundle `complete`；A/B 复用已完成 run，C 使用新 run ID 重试完成；所有 checkpoint finite，residual relative norm A/B/C=`0/0.001505/0.001438` | validation CER A/B/C=`0.787648/0.786034/0.794012`；B−A=`−0.001613`，C−B=`+0.007978`；A/B/C teacher-forced OCR loss=`1.666193/1.664493/1.663769`。C 的 layout box MAE=`0.077862`、validity AUROC=`0.953366`，但 invalid gated context share=`0.798970`、p gap=`0.019232`，尚未达到 query 消除阈值；B validity AUROC=`0.494084`、invalid share=`0.924753` | 不读取；`test_manifest_read=false`、`test_used_for_selection=false` | A=`glmocr_mthv2_attribution_128_v1_A_noop`，B=`glmocr_mthv2_attribution_128_v1_B_adapter_only`，C=`glmocr_mthv2_attribution_128_retry_v1_C_decoder_lora`；原 C run 因 NCCL watchdog 超时保留，retry 将 DDP timeout 提高到 3600s；容量归因结论：LoRA 明显改善布局/validity，但在 128 steps 下未转化为 OCR CER，反而较 B 回退 |

## 2026-09-15/16：box 权重等化续训与 gate=0.005 跨域微调

本轮先在 MTHv2 sparse24 上从既有 `boxeq58` checkpoint 接续 3000 步，把 layout loss 中 box 项提高到 `820.0`，再将该 run 的 seed42 Q32 `adapter.safetensors` 和 `decoder_lora.safetensors` 一起迁移到敦煌／地方志合并数据。后续微调严格只跑 `attention` 与 `geometry` 两臂、各 256 步；训练结束后先用唯一 checkpoint 做 validation-only selection，再做 selection-locked test。

| 阶段 | run ID | validation | locked test | 关键状态 |
| --- | --- | --- | --- | --- |
| MTHv2 Q32 source | `glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1` | IoU `0.456072`；MAE `0.029717` | IoU `0.446998`；MAE `0.030560`；CER `0.393093` | step `3000` complete；训练、选点和 test 均 finite；test 未参与选点 |
| 敦煌／地方志 attention | `glmocr_dunhuang_local_q32_boxeq820_gate005_lr1e5_attention_256_frommthv2_260916_v1` | step `256` selected | IoU `0.271496`；MAE `0.053813`；CER `0.173582`；I/D/S `482/487/1495` | gate `0.005`；EOS `1.0`；触顶、repeated-cycle、loop 均 `0` |
| 敦煌／地方志 geometry | `glmocr_dunhuang_local_q32_boxeq820_gate005_lr1e5_geometry_256_frommthv2_260916_v1` | step `256` selected | IoU `0.286134`；MAE `0.053027`；CER `0.173371`；I/D/S `483/486/1492` | gate `0.005`；EOS `1.0`；触顶、repeated-cycle、loop 均 `0` |
| 敦煌／地方志 content-only | `glmocr_dunhuang_local_q32_boxeq820_gate005_lr1e5_content_only_256_frommthv2_260916_v1` | step `256` selected | IoU `0.263915`；MAE `0.068268`；CER `0.173582`；I/D/S `482/486/1496` | `auxiliary_weight=0`；融合 identity；EOS `1.0`；触顶、repeated-cycle、loop 均 `0` |

相同起点和预算下，content-only 的 CER `0.173582` 与 attention 完全一致，但 layout test IoU/MAE 为 `0.263915/0.068268`，弱于 attention 的 `0.271496/0.053813`；geometry 仍以 `0.286134/0.053027` 的 IoU/MAE 和 `0.173371` CER 最优。该 content-only 结果支持布局融合对布局指标的增益方向，但当前只有一个 seed、一个 256-step checkpoint，仍属于诊断性证据，不替代多 seed 或长程正式比较。

远端证据根目录为 `/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/training_runs/`；编排汇总为 `/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/runs/glmocr_dunhuang_local_q32_gate005_attn_geom_260916_v1.summary.json`。两个微调 run 均记录 `test_manifest_read=false`、`test_used_for_selection=false`，且各自保存 `checkpoint-256`、`selection.json` 和 `locked-test/locked_test_summary.json`。

## 历史记录：2026-09-10/11 循环生成与 Teacher Forcing 审计

本节仅封存已完成的 natural-loop/Teacher Forcing 审计，不属于当前 BSCC 方案。其历史记录统一使用 seed `42`、同一分层 64 页 validation、最多 256 steps、五卡 DDP、整页输入、test 锁定；所有正式记录均为 `test_manifest_read=false`、`test_used_for_selection=false`。旧 A1 审计是已有 checkpoint 的 eval-only，不是当前 BSCC 训练。

| ID | 配置与状态 | validation / 工程结果 | test | 备注 |
| --- | --- | --- | --- | --- |
| `GLMOCR-B-260910-003` | 历史 fast-screen A0/A1；各 256 steps；旧日志中的生成评估 | A0 step128/256 CER=`0.775010/0.945414`；A1 step128/256=`0.394836/0.448864`；TF loss=`1.736313/1.698475` | 不读取 | 原始记录见 `docs/实验日志/GLMOCR/训练退化诊断/GLMOCR-B-260910-003.md`；A1 使用旧硬 EOS guard。该日志与后续 repeat-audit 是不同评估输出，不能混为一条曲线 |
| `glmocr_a1_repeat_audit_260910_v3` | 旧 A1 checkpoint eval-only；64 页；旧 AdaptiveCycle guard | guard ON：step128/256 CER=`0.394836/0.413352`；guard OFF：`0.493811/0.546165`。ON 的删除数=`3961/3802`，OFF 的长度上限页率=`0.15625/0.21875` | 不读取 | ON 使用 `cycle_penalty=2.0`、`force_eos_steps=16`，属于强制截断；旧标签无显式 EOS（`eos_label_count=0`） |
| `glmocr_loop_escape_smoke_260910_v1` | loop-escape 首次 bounded smoke；启动前协议检查失败 | 未训练 | 不读取 | 旧 launcher 误判为可能读取 test，保留失败产物，不作性能结论 |
| `glmocr_loop_escape_smoke_260910_v2` | 修正 no-test 协议后启动 | 输入追加 EOS 后首次 forward 触发 `mm_token_type_ids` 长度不一致，未形成结果 | 不读取 | 修复 `append_eos_label_token` 并保留失败 run |
| `glmocr_loop_escape_smoke_260910_v3` | 8 steps；五卡；loop escape、continuation head、EOS label | complete；最大区域 407；reload finite；mean loop escape/margin/continue/head=`5.56189/1.43736/0.000885/0.55095` | 不读取 | 证明新损失、EOS 追加、checkpoint 重载可运行 |
| `glmocr_loop_escape_audit_260910_v1` | 首次完整审计尝试 | stopped；脚本错误包含 validation-0，违反“直接训练、不做 validation-0”的当前要求 | 不读取 | 协议无效，不作性能结论 |
| `glmocr_loop_escape_audit_260910_v2_A0` | A0：纯 teacher forcing；256 steps；`force_eos_steps=0` | complete；step128/256 CER=`0.778612/0.749493`；TF loss=`1.815761/1.779871`；cycle page=`0.109375/0.109375`；limit=`0.15625/0.140625`；EOS=`0.84375/0.859375` | 不读取 | 选择 step256；mean OCR/token-weighted=`1.516484/1.596768`；loop/mixed/head loss 均为 0 |
| `glmocr_loop_escape_audit_260910_v2_A1` | A1 首次训练；loop escape＋continuation head＋scheduled sampling | stopped at step1；DDP 各 rank 条件分支导致第二次 forward 不一致，30 分钟无有效进展 | 不读取 | 保留为工程失败，不作性能结论 |
| `glmocr_loop_escape_smoke_260911_v4` | 修复 DDP：启用 loop escape 时所有 rank 同步执行第二次 forward；8 steps | complete；五卡、reload finite、最大区域 407 | 不读取 | 为 A1 重试提供工程验证 |
| `glmocr_loop_escape_audit_260910_v3_A1` | A1 重试；同一 64 页子集；256 steps；其余与 A0 一致 | complete；step128/256 CER=`0.686485/0.863078`；TF loss=`1.813698/1.776209`；cycle page=`0.09375/0.15625`；limit=`0.125/0.171875`；EOS=`0.875/0.828125`；loop escape success=`0` | 不读取 | validation 选择 step128；mean OCR/token-weighted/mixed/loop/head=`1.515252/1.595696/2.564753/4.515455/0.416838`；step256 出现后程退化 |

### A0/A1 与旧 A1 的可比指标

| 实验 / checkpoint | CER | TF OCR loss | 插入/删除/替换 | 循环页率 | 长度上限页率 | EOS 正常结束率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧 A1 guard ON / 128 | 0.394836 | 1.736313 | 532 / 3961 / 3290 | 0.046875 | 0 | 1.000000 |
| 旧 A1 guard ON / 256 | 0.413352 | 1.698475 | 809 / 3802 / 3537 | 0.046875 | 0.015625 | 0.984375 |
| A0 / 128 | 0.778612 | 1.815761 | 8287 / 1395 / 5666 | 0.109375 | 0.156250 | 0.843750 |
| A0 / 256 | 0.749493 | 1.779871 | 7933 / 1527 / 5314 | 0.109375 | 0.140625 | 0.859375 |
| A1 / 128 | 0.686485 | 1.813698 | 6524 / 1543 / 5465 | 0.093750 | 0.125000 | 0.875000 |
| A1 / 256 | 0.863078 | 1.776209 | 9727 / 1500 / 5786 | 0.156250 | 0.171875 | 0.828125 |

## 2026-09-11：natural predicted-loop 1024-step direct-test 记录

本条记录覆盖两个从同一基础权重、同一 seed `42`、同一全量 MTHv2 train manifest 重新开始的 1024-step run。两者均不做 validation，step `1024` 作为 fixed-final checkpoint；文档先于 baseline test 更新，随后完成了五卡分片 direct test。该 no-validation 直测是用户明确授权的工程对照，不应与 validation-selected formal test 混称。

| 项目 | A1 | plain baseline |
| --- | --- | --- |
| run ID | `glmocr_natural_loop_A1_1024_260911_v1` | `glmocr_plain_baseline_1024_260911_v1` |
| 训练目标 | natural predicted-loop loss，权重 `0.05` | `outputs.loss` 官方兼容基础 OCR loss，natural loop 关闭 |
| 生成 | `plain`；cycle penalty `0`；forced EOS `0` | 同左 |
| 训练 | LoRA；1024 steps；五卡 DDP；global batch `5`；warmup `102`；LR horizon `1024` | 同左 |
| 训练状态 | complete；checkpoint-1024 finite | complete；checkpoint-1024 finite |
| validation | 不读取；fixed-final step1024 | 不读取；fixed-final step1024 |
| 训练 mean official base loss | `1.4099817903` | `1.4099817903` |
| 训练 mean total loss | `2.4221982436` | `2.4221982436` |
| token-weighted OCR loss | `1.4721145695` | `1.4721145695` |
| natural-loop active ratio | `0` | `0`（功能关闭） |

### A1 已完成 direct test

| 指标 | A1 step1024 |
| --- | ---: |
| test pages | `800` |
| CER | `0.8746345057` |
| 插入 / 删除 / 替换 | `149592 / 12942 / 67794` |
| 循环页率 | `0.195000` |
| repeated-cycle 页率 | `0.167500` |
| 长度上限率 | `0.207500` |
| EOS 命中率 | `0.792500` |

A1 与 baseline 均使用 fixed-final step1024，`test_used_for_selection=false`；两次 direct test 都不是 validation 选点结果。baseline 使用 5 个互斥 shard（GPU `0,1,2,3,4`）并在合并后得到单一 summary。

### A1 与 plain baseline direct test 对照

| 指标 | A1 step1024 | plain baseline step1024 |
| --- | ---: | ---: |
| test pages | `800` | `800` |
| CER | `0.8746345057` | `0.8746345057` |
| 插入 / 删除 / 替换 | `149592 / 12942 / 67794` | `149592 / 12942 / 67794` |
| 循环页率 | `0.195000` | `0.195000` |
| repeated-cycle 页率 | `0.167500` | `0.167500` |
| 长度上限率 | `0.207500` | `0.207500` |
| EOS 命中率 | `0.792500` | `0.792500` |
| 平均新生成 token 数 | `594.62125` | `594.62125` |

两组 test 指标逐项完全一致。A1 与 baseline 的 `adapter.safetensors`、`decoder_lora.safetensors` 和 `adapter_config.json` SHA-256 也完全一致，说明 natural-loop loss 在该训练协议中没有产生任何参数更新差异。

### 配置、数据与证据指纹

- 分支：`glm-ocr-layout-ot`；代码提交：`9c7e578`（五卡 locked-test 分片入口）。
- 模型：GLM-OCR revision `ca5d8b3e287e52589e37c28385d9655ee4372f9d`。
- train manifest SHA-256：`047f2101254d5cdcdd889a3840cbc0eae25e994ca2ffd333f3a53b7d0f803d3f`。
- test manifest SHA-256：`2904bdaf155a4d1b162d4e4f5fc378cc2263f7d9ea9990c1e3e1020775911962`。
- full protocol SHA-256：`e713c821afd97c6050b8f6d37553d3e783ed754c6b619379c532b88d957e0f93`。
- 64 页 validation manifest 未读取；其 SHA-256 为 `8fb3478d621f46c1d6f9dc7f5c400626af46dc068c6fc6b4ce2a519a3ccbba38`，仅作为启动参数保留。
- A1 训练证据：`/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/training_runs/glmocr_natural_loop_A1_1024_260911_v1/seed42/summary.json`；A1 test：同目录 `locked-test/locked_test_summary.json`。
- baseline 训练证据：`/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/training_runs/glmocr_plain_baseline_1024_260911_v1/seed42/summary.json`；baseline test：同目录 `locked-test/locked_test_summary.json`。

### 当前解释边界

A1 与 baseline 的 1024-step 训练汇总完全一致，A1 的 natural-loop active ratio 为 `0`，两组最终 adapter/checkpoint 文件逐字节一致，且 800 页 plain direct test 指标逐项一致。因此历史审计没有证据表明该惩罚改变了模型参数、降低了循环率或学会了自然脱环；该方向不再继续，当前 BSCC 返回 plain training objective。

### 下降原因与边界判断

1. **首先不是同一种解码协议。** 旧 A1 的低 CER 依赖 `force_eos_steps=16`：检测到循环后允许最多继续 16 token，仍未脱离就直接输出 EOS。它把循环截断为删除错误；旧 guard ON 的删除数达到 `3961/3802`，但插入数只有 `532/809`。新 A0/A1 将该硬 EOS 关闭，目标是“脱离循环后继续生成直到真实 EOS”，所以会暴露原模型的插入和循环错误。旧的 `0.39–0.41` 不能当作同协议下的识别能力上限。
2. **新实验从基础权重重新训练，旧结果是已有 fast-screen checkpoint 的 eval-only。** 旧审计读取 `glmocr_fast_screen_260910_r3_A1` 的 step128/256；A0/A1 是新的 256-step 训练。即使只看 teacher-forced loss，旧值为 `1.7363/1.6985`，新 A0/A1 为约 `1.81/1.78`，说明 checkpoint 学到的状态本身就不同，下降不只是推理器造成。
3. **监督分布发生了有意但实质的变化。** 旧标签没有显式 EOS，目标 token 数为 `23228`；新标签每页补真实 EOS，`eos_label_count=64`、目标 token 数为 `23292`。这对学习正确停止是必要修正，但会改变最后位置监督与优化轨迹，不能把两组 loss 当成完全同分布。
4. **A1 在后半程改变了梯度预算。** A1 的第二次 loop-corrupted-prefix forward、scheduled sampling、continuation head 和 loop escape margin 都产生非零梯度；平均 `mixed_prefix_loss=2.5648`、`loop_escape=4.5155`、`head=0.4168`。step128 的 CER 比 A0 好，但 step256 反而升到 `0.8631`，同时循环页率和插入数上升，表明当前权重/调度在后程过强或不稳定，而不是 teacher-forced CER 下降就等于自由生成变好。
5. **当前“loop escape success”尚未证明模型真的学会续写。** A1 两个 checkpoint 的该指标均为 `0`；step128 的相对改善主要来自插入数下降和循环/触顶率暂时下降，不能解释为已经学会“停循环后输出后文”。
6. **下降主要由密集页和自由生成长尾放大。** 新协议不再用硬 EOS 掩盖长尾，密集页面更容易累积区域/文本错误；A1 step256 的密集页 CER 已约 `1.3318`，而旧 guard ON 密集页约 `0.6269`。因此 64 页宏平均的恶化不是单一 batch loss 上升，而是 dense-page free-run 长尾失稳。

结论：这次“相较上一个实验都下降”同时包含**协议不可比**（硬 EOS 截断被移除）和**真实训练退化**（新 checkpoint 的 TF loss 更高、A1 后程过拟合/梯度竞争）。该结论仅用于封存历史 natural-loop 诊断；当前不再继续 natural-loop/loop-escape 路线，后续 A100-yky 与 BSCC 新 run 统一使用参数优选后的 `L_official + 0.4 L_layout` plain objective。旧低 CER 也不能作为当前协议下的性能承诺。

## 2026-09-13/18：iou_consistent 布局精修、tf 三臂与 sem_adapter 阶段二

补登 2026-09-13 至 09-18 共 25 个 run。除注明外均为 A100 五卡同步 DDP、global batch 5、seed 42、`test_manifest_read=false`。

### A. 2026-09-13/14：敦煌地方志长程微调与跨域 warm-start

| run ID | 配置 | validation | locked test | 关键状态 |
| --- | --- | --- | --- | --- |
| `glmocr_dunhuang_local_gazetteer_q32_geometry_2k_a100_260913_v4` | geometry；full layout loss；`auxiliary_weight=0.4`；decoder LoRA；2000 步 | 选 step `500`；CER `0.457322` | 59 页；CER `0.454315`；MAE `0.200780`；I/D/S `4283/453/1713`；触顶 `4`；repeated-trigram `0.1072` | 过生成主导，插入占编辑距离 `0.664`；gate 升至 `0.015452`。该结果确立敦煌长程微调的崩坏模式 |
| `glmocr_dunhuang_local_gazetteer_q32_official_content_only_2k_a100_260913_v4` | content_only；full；2000 步 | 选 step `500`；CER `0.695865` | 59 页；CER `0.816485`；MAE `0.211802`；I/D/S `9137/459/1994`；触顶 `7` | 插入占比 `0.788`，最严重的过生成；gate 全程 `0`，说明崩坏不依赖残差通路 |
| `glmocr_q32_free_warmstart_ref600_content_only_600_from200_lr125e-6_dec25e-7_r512_a100_260914_v1` | content_only；full；decoder LR `2.5e-7`；600 步 | 选 step `800`；CER `0.204236` | 59 页；CER `0.170764`；MAE `0.211803`；I/D/S `478/477/1469`；触顶 `0`；trigram `0.0408` | 起点为 `training_initializations/glmocr_dunhuang_local_gazetteer_q32_content_only_ref600_bscc_260913_v2_step200`；低 LR 下过生成未出现 |
| `glmocr_q32_free_warmstart_ref600_geometry_600_lr625e-6_dec125e-6_r512_a100_260914_v1` | geometry；full；`auxiliary_weight=0.2`；decoder LR `1.25e-6`；600 步 | 选 step `600` | 59 页；CER `0.169074`；MAE `0.192503`；I/D/S `478/459/1463`；触顶 `0` | 起点 `..._geometry_ref600_bscc_260913_v2_step200`；终值 gate `0.009494` |
| `glmocr_q32_free_warmstart_ref600_geometry_2000_from600_repair_lr125e-6_dec25e-7_r512_a100_260914_v1` | geometry；full；`auxiliary_weight=0.2`；decoder LR `2.5e-7`；1400 步 | 选 step `1400`；CER `0.197226` | 59 页；CER `0.168862`；MAE `0.195746`；I/D/S `475/485/1437`；触顶 `0` | 从上一行 `checkpoint-600` 接续的修复臂 |
| `glmocr_q32_free_warmstart_ref600_geometry_2000_from600_lr625e-6_dec125e-6_r512_a100_260914_v1` | geometry；full；2000 步 | 无 | 不读取 | `stopped_by_user`；仅 `metadata.json`／`train_metrics.jsonl`，无 checkpoint，不作结果 |

### B. 2026-09-15：MTHv2 layout-only 与 tf_gate005 三臂

| run ID | 配置 | validation | locked test | 关键状态 |
| --- | --- | --- | --- | --- |
| `glmocr_mthv2_sparse24_q32_layout_only_3000_a100_260915_v3` | geometry；full；`layout_only=true`；`auxiliary_weight=1.0`；3000 步 | 选 `validation_layout_box_iou` step `3000`；IoU `0.072805` | 509 页；CER `0.393093`；IoU `0.072533`；MAE `0.226048`；I/D/S `29308/6435/18566`；触顶 `26`；trigram `0.3185` | gate 全程 `0`，语义通路未激活；作为后续 Q32 布局续训的共同起点 |
| `glmocr_q32_tf_attention_gate005_256_a100_260915_v2` | attention；full；`auxiliary_weight=0.4`；256 步；`--no-validation` | 无 | 59 页；**CER `0.167735`**；MAE `0.085796`；I/D/S `476/456/1449`；触顶 `0`；trigram `0.0414` | 59 页历史最优；gate `0.012892` |
| `glmocr_q32_tf_content_only_gate005_256_a100_260915_v1` | content_only；full；256 步；`--no-validation` | 无 | 59 页；CER `0.168017`；MAE `0.211803`；I/D/S `461/459/1465`；触顶 `0` | gate `0.005000`（未动） |
| `glmocr_q32_tf_geometry_gate005_256_a100_260915_v2` | geometry；full；`auxiliary_weight=0.4`；256 步；`--no-validation` | 无 | 59 页；CER `0.169074`；MAE `0.195998`；I/D/S `476/469/1455`；触顶 `0` | gate `0.012006` |
| `glmocr_q32_free_geometry_from600_2500_lr4x_min0p2_r512_a100_260915_v2` | geometry；full；`auxiliary_weight=0.2`；2500 步 | 无 | 不读取 | 从 `ref600_geometry_600/checkpoint-600` 接续；complete 但未做 selection 与 test；终值 gate `0.020056` |
| `glmocr_dunhuang_local_q32_alpha001_from_sparse_layout_256_a100_260915_v5` | geometry；full；`auxiliary_weight=0.4`；256 步 | `fixed_final_step_diagnostic`；`validation_evaluated=true` | 不读取 | 起点 `layout_only_3000_v3/checkpoint-3000`，`init_checkpoint_override_residual_scale=0`；gate `0.017337`；诊断 run，无 locked test |

### C. 2026-09-16/17/18：iou_consistent / giou 布局精修线

布局损失从 box 权重等化转向 `iou_consistent` 系列。该线 `decoder_adaptation=frozen`、`auxiliary_weight=1.0`，只训布局分支；逐 checkpoint 在 MTHv2 sparse24 validation 149 页上评测。

| run ID | 起点 / 步数 | 逐 step 布局验证（IoU / MAE） | 关键状态 |
| --- | --- | --- | --- |
| `glmocr_mthv2_sparse24_q32_layout_continuation_2000_from3000_a100_260916_v1` | `layout_only_3000_v3/checkpoint-3000`（full loss）；2000 步 | 未评测 | complete；gate `0`；无 selection 与 test |
| `glmocr_dunhuang_local_q32_alpha001_from_boxeq58_256_a100_260916_v1` | `boxeq58_3000/checkpoint-3000`（`history_box_equalized_v1`）；256 步 | 未评测 | complete；gate `0.016514`；无 selection 与 test |
| `glmocr_mthv2_sparse24_q32_layout_iou_consistent_giou10x_mlp_refine_3000_from_boxeq820_20260917_v1` | `boxeq820_3000/checkpoint-3000`；3000 步；`iou_consistent_giou10x` | `1000`：`0.422549`／`0.048471`；`2000`：`0.475591`／`0.039474`；`3000`：`0.549673`／`0.037354` | complete；gate 触顶 `0.0300`，`mean_gradient_norm=376.5` 异常大；无 selection 与 test |
| `glmocr_mthv2_sparse24_q32_layout_iou_consistent_mlp_refine_3000_from_boxeq820_20260917_retry2` | 同上；3000 步；`iou_consistent` | 未评测 | 仅到 `checkpoint-1000`，未完成；不作结果 |
| `glmocr_mthv2_sparse24_q32_layout_iou_consistent_giou10x_mlp_refine_10000_continue_from_giou10x3000_20260917_v1` | 上一行 giou10x run 的 `checkpoint-3000`；10000 步 | 无 | 仅 `metadata.json`，无 checkpoint；不作结果 |
| `glmocr_mthv2_sparse24_q32_layout_iou_consistent_giou10x_mlp_refine_10000_continue_from_giou10x3000_20260917_v2` | 同上；10000 步 | `2000`：`0.470294`／`0.045306`；`4000`：`0.514404`／`0.043766`；`6000`：`0.594829`／`0.035771`；`8000`：`0.639213`／`0.028548`；`10000`：`0.647969`／`0.028292` | 训练与布局验证均已完成，`status` 未翻为 complete；未做 selection 与 test |
| `glmocr_mthv2_sparse24_q32_layout_iou_consistent_giou20x_mlp_refine_10000_continue_from_giou10x10000_20260918_v1` | 同上；10000 步 | 无 | 空目录（无 seed 产物）；不作结果 |
| `glmocr_mthv2_sparse24_q32_layout_iou_consistent_giou20x_mlp_refine_10000_continue_from_giou10x10000_20260918_v2` | `giou10x_10000_v2/checkpoint-10000`；10000 步；`iou_consistent_giou20x` | `2000`：`0.535813`／`0.039420`；`4000`：`0.604446`／`0.030769`；`6000`：`0.636638`／`0.028477`；`8000`：`0.671110`／`0.027422`；`10000`：**`0.672084`／`0.027824`** | 本线最优布局结果；作为 09-18 sem_adapter 的起点；未做 selection 与 test |

布局精修线小结：`giou10x` 把 3000 步的 IoU `0.549673` 推高到 10000 步的 `0.647969`；`giou20x` 在同样 10000 步上进一步到 `0.672084`，MAE 同步从 `0.037354` 降到 `0.027824`。同期 OCR 指标（validation CER 约 `0.49–0.58`）未见同向改善，符合「布局分支只优化几何」的两阶段设计预期。

### D. 2026-09-18：sem_adapter 阶段二

阶段二按 `plans/LAYOUT_OCR_DECOUPLING_PLAN.md` 冻结布局分支，只训 `sem_adapter` + `content_gate`（外加 decoder LoRA），损失为 OCR-only。

| run ID | 数据 / 配置 | validation | locked test | 关键状态 |
| --- | --- | --- | --- | --- |
| `glmocr_mthv2_sem_adapter_stage2_from_giou20x10000_20260918_v1` | MTHv2；`layout_ot`；`ocr_only`；LoRA LR `5e-6`；1024 步 | 并行验证选 step `256`；CER `0.949633`（149 页） | 不读取 | 冷启动未修正（gate 起点 `0`）且主 LR 过高；终值 gate 触顶 `0.0300`；判定配置失败 |
| `glmocr_mthv2_sem_adapter_stage2_from_giou20x10000_lrhalf_warmup432_gate001_20260918_v1` | 同上，主 LR 减半、warmup `432`、gate 热启动 `0.01` | 无 | 不读取 | smoke 阶段 `failed`；不作结果 |
| `glmocr_mthv2_sem_adapter_stage2_from_giou20x10000_lrhalf_warmup432_gate001_20260918_v2` | 同行修正后重试；decoder LR `2.5e-6`；1024 步 | 并行验证选 step `256`；CER `0.519398`（149 页） | 不读取 | 冷启动修复生效（终值 gate `0.023306`），validation CER 由 `0.9496` 改善到 `0.5194`，但仍远差于布局起点 |
| `glmocr_mthv2_sem_adapter_stage2_gate001_lr1e5_256_from_giou20x10000_20260918_v1` | MTHv2；主 LR `1e-5`、decoder LR `5e-6`、gate 热启动 `0.01`；256 步；`--no-validation` | 无（固定末步） | 509 页；CER `0.443434`；IoU `0.665427`；MAE `0.031535`；I/D/S `38098/4762/18404`；触顶 `35`；trigram `0.3334` | gate `0.012655`；差于起点 `0.393093`，插入占编辑距离 `0.622`，过生成主导，判定该轮 stage-2 失败 |
| `glmocr_dunhuang_local_sem_adapter_gate001_lr1e5_dec1e6_256_from_giou20x10000_20260918_v1` | 敦煌地方志 `glmocr_compat`；主 LR `1e-5`、decoder LR `1e-6`、gate 热启动 `0.01`（上限 `0.03`）；256 步；`--no-validation` | 无（固定末步） | 59 页；CER `0.170694`；IoU `0.365151`；MAE `0.090792`；I/D/S `476/475/1472`；触顶 `0`；trigram `0.0406` | gate `0.010253 → 0.012923` 单调上升；插入占比 `0.196`。同 test 历史区间 `0.1677–0.1736`，本次居中：未超过最优 `0.167735`，但也未出现 MTHv2 那次的崩坏 |
| `glmocr_dunhuang_local_sem_adapter_continue256_val64_gate0013_lr1e5_dec1e6_20260918_v1` | 敦煌地方志；从上一行 `checkpoint-256` 续训 256 步（gate 热启动 `0.012923` 精确接续）；主 LR `1e-5`、decoder LR `1e-6`；每 64 步存档；`--defer-validation` + 5 路并行 eval-only | 并行验证选 step `256`；CER `0.207219`（80 页）；四个候选点 `0.207219–0.208077`，**step-0 恒等对照 `0.207331`** | 59 页；CER `0.169778`；IoU `0.365151`；MAE `0.090792`；I/D/S `475/470/1465`；触顶 `0`；trigram `0.0410` | gate `0.012923 → 0.015485` 仍单调（距上限仍有 1.9×），但 `ema_ocr_loss_16` 均值 `1.1451` 高于父 run 末值 `1.0801`；**四个候选点与 step-0 恒等对照完全持平**，选点 CER 仅比起点好 `0.0001`；test 微降 `0.0009`（父 run `0.170694`）。判定该配置已无训练空间 |
| `glmocr_dunhuang_local_q32_zeroshot_baseline_20260918_v1` | **zero-shot 对照（非训练）**：官方 GLM-OCR 权重 `ca5d8b3e287e52589e37c28385d9655ee4372f9d`，`content_only` + `initial_residual_scale=0`（适配器对视觉特征严格恒等）、decoder 冻结；`tools/evaluate_glmocr_zeroshot_test.py` | 80 页 validation；CER `0.210687` | 59 页；CER `0.173864`；I/D/S `482/486/1500`；触顶 `0`；trigram `0.0400` | `adapter_gate_after_load=0.0`、`raw_content_gate=0.0`、`residual_relative_norm=0.0`、`decoder_lora_loaded=false`、`training_updates=0`；prompt 与 layout_ot 完全相同（固定 `_messages()`），故为**端到端流水线净收益**的对照，不可拆分为 `sem_adapter` 单独贡献 |
| `glmocr_dunhuang_stage2_ablation_resolution_20260918_v1` | **判定性消融（非训练）**：同 checkpoint / 同 mode / 只改单一变量的 5 臂 eval-only。A/B 改写父 run checkpoint 的 `content_gate` 标量（0 与上限 0.03），C 为微调模型 4M，D/E 为基座 4M/2M | 80 页 validation：A(gate=0,1M) `0.210501`；B(gate=cap,1M) `0.205914`；C(微调,4M) `0.111608`；D(基座,4M) `0.112988`；E(基座,2M) `0.135474` | 不读取 | **分辨率 −0.0977 是主导因素（零训练）**；gate=0 与基座无差异（LoRA 在 1M 下惰性）、gate=0.0129 显著改善（残差通道有效）；gate 单调到上限仍未触顶 |

阶段二结论：同一套 stage-2 配置在 MTHv2 上把 CER 从 `0.393093` 推到 `0.443434`（插入占比 `0.622`），换到敦煌目标域后 CER `0.170694`、插入占比 `0.196`、触顶 `0`——支持「stage-2 应在目标域执行」的设计判断。

**续训验证（09-18 补）**：在父 run 之上续训 256 步并每 64 步存档验证，四个候选点 `0.207219–0.208077`、step-0 恒等对照 `0.207331`，全部持平（0.0001 量级）；锁定 test `0.169778` 对父 run `0.170694` 仅降 `0.0009`，而同一间隔的 validation CER 完全不动。结论：**该配置（主 LR `1e-5`、decoder LR `1e-6`、gate 上限 `0.03`、256 步）已无训练空间**，日志里 gate 仍在单调上升不构成「还有空间」的证据。

**zero-shot 对照（09-18 补）**：官方 GLM-OCR 基座在同一 59 页 test 上 CER `0.173864`、80 页 validation `0.210687`。整条 stage-1 布局精修 + stage-2 语义适配器链路的净收益因此只有 `0.004–0.006` CER（test）/ `0.0035`（validation），且所有微调 run 都挤在 `0.1677–0.1736` 这 `0.006` 宽的带内——**基座在该 test 上已接近饱和，既有指标区分度低于 59 页样本噪声**。此外 zero-shot 的插入占比 `0.195`、触顶 `0`、trigram `0.040` 与微调 run 一致，说明敦煌 2k 步那两次的过生成崩坏（插入占比 `0.66`/`0.79`）是训练过久引入的，不是基座固有行为。

**仍未解决**：`sem_adapter` 相对 `content_only`／`geometry` 的必要性——上面两条对照都不构成 `gate=0` 冻结 vs `gate` 可训的同 mode 消融。要论证增益，需在同一起点、同一 split 上补这一格。

**判定性消融与分辨率扫描（09-19 补）**：同 checkpoint、同 mode、只改单一变量的 5 臂 eval-only（`glmocr_dunhuang_stage2_ablation_resolution_20260918_v1`，80 页 validation，逐页配对 bootstrap）。三条结论：

1. **输入分辨率是主导因素。** `max_pixels=1003520` 把每一页（原生 310–367 万像素）下采样到约三分之一，官方 processor 默认 `longest_edge` 是 `9633792`。仅把它放到原生，**零训练**即让基座 CER 从 `0.210687` 降到 `0.112988`（−0.0977，−46%；CI [−0.1100, −0.0850]），约为此前全部适配器工作（1M 下 0.0048）的 20 倍。增益全部来自替换错误（4580→2056），插入/删除/生成长度基本不变。
2. **有效通道是语义残差，不是 decoder LoRA。** 同 checkpoint 下 `gate=0` 时 `0.210501`，与基座 `0.210687` 无差异（CI 含 0）→ LoRA 在 1M 下惰性；`gate=0.012923` 时 `0.207331`，显著改善（CI [+0.0018, +0.0047]）。**此前「收益全部来自 decoder LoRA」的归因基于跨谱系比较（`content_only` run 的起点本身已是微调模型），不是受控消融，已更正。**
3. **门控被上限卡住。** `0 → 0.0129 → 0.03` 对应 `0.2105 → 0.2073 → 0.2059` 单调，而训练全程 gate 只到 `0.015485`，从未触顶。

原生分辨率下微调仍显著但幅度缩小（微调 4M `0.111608` vs 基座 4M `0.112988`，CI [−0.0022, −0.0006]），同一对适配器在 1M 下贡献 `0.0034`——说明 1M 下的一部分增益是在补偿退化的输入。原生分辨率下替换 2030 处分布在 1279 个不同对上，其中约 7.5% 是异体字/标注噪声（`眾/衆`40、`髙/高`18、`於/扵`17、`逺/遠`14 等，且 `眾`40/`衆`8、`逺`21/`遠`18 参考自身两形并存）。

本段训练运行均记录 `test_manifest_read=false`／`test_used_for_selection=false`；test 协议在各 run 训练结束后单独生成，且只读取一次。唯一的例外是 `..._zeroshot_baseline_20260918_v1`：它是纯 test 对照、不做任何选点，复用续训 run 的 test 协议（59 页 + image sha256 锁定），记录 `test_manifest_read=true`、`test_used_for_selection=false`、`training_updates=0`。

## 每个 run 必填

记录 Git commit、上游 GLM-OCR 版本与权重标识、manifest 哈希、R1/R2 配置、GPU 集合、训练预算、辅助损失权重、validation 选点规则和唯一 run ID。失败 run 保留原 ID，重试使用新 ID。

本表不继承 GOT2 LAVP 历史结果；如后续提取其指标、数据读取或 checkpoint 检查组件，必须单独登记来源 commit 和本分支改动。

## 当前筛选数据边界

MTHv2 原官方 split 是随机页级划分，没有书籍/版本元数据，不满足本项目的隔离要求。本轮不沿用该 split：将 `original_image` 中 `V...P...` 的卷号前缀作为版本/文档组代理，无卷号的数字页按 subset 合并为一组，再划分 train/validation/test。仅纳入 `1–32` 个文本行区域的整页，保证 32-query 无截断。选中页面还需通过 16×16 dHash、Hamming 距离不大于 4 的跨 split 近重复检查。这一卷号映射是可执行代理协议，不声称等价于完整书手或馆藏标注。

上述 `1–32` 个区域限制只适用于历史 128 页机制筛选，不适用于当前全量 MTHv2 DDP。全量协议使用 512 queries，最大区域数为 407，保留全部 2159/240/800 页；validity/no-object 目标只由训练期 Hungarian 匹配生成，布局真值和 query mask 不进入推理输入。

## 2026-09-12 BSCC 四卡 decoder-LoRA 20k 实验

| 字段 | 口径 |
| --- | --- |
| run ID | `glmocr_mthv2_decoder_lora_lr1e5_20k_4gpu_official_layout_260912_v1` |
| 分支 | `glm-ocr-layout-ot` |
| 训练 | BSCC 四卡同步 DDP；global batch `4`；`20000` steps；seed `42` |
| 数据 | full MTHv2：train/validation/test = `2159/240/800`；whole-page；512 queries |
| 模型与目标 | geometry；Hungarian；full layout loss；FP32 adapter；fast processor；`L_official + 0.2 L_layout`；关闭 natural-loop、scheduled sampling、loop escape 和 continuation head |
| decoder LoRA | rank `8`、alpha `8`、dropout `0`；本 run 学习率 `1e-5` |
| checkpoint | `5000/10000/15000/20000` |
| protocol 边界 | training 与 parallel validation 只使用 train/validation；selection 后才创建 test protocol |
| 当前状态 | 原 v1、rollout256 诊断 run 与 BSCC pending job 均不进入结果；BSCC job `1494430/1494433` 在分配节点前取消，当前转由 A100 五卡入口运行 |
| 结果口径 | 用户授权的高 decoder-LoRA 学习率探索；不替代历史五卡协议，不与五卡结果作未校正的严格等预算比较 |

原 `glmocr_mthv2_decoder_lora_lr1e5_20k_4gpu_260911_v1` 仅设置了 `generation_mode=loop_recovery`，未设置 `--natural-loop-loss` 或 `--loop-escape-training`，因此没有循环训练信号；该 run 已取消并保留日志，不进入 validation/test。上一轮 rollout256 诊断 run 也不作为本轮训练依据。上述历史 official-layout run 使用 `L_official + 0.2 L_layout`，训练阶段关闭 natural-loop、scheduled sampling、loop escape 和 continuation head；后续新 run 改用 `L_official + 0.4 L_layout`。`evaluate_glmocr_locked_test.py` 已按 metadata 注入并加载 decoder LoRA，test shard 和 merge 阶段均强制检查 `decoder_lora_loaded=true`。

## 2026-09-12 A100 五卡 decoder-LoRA plain 20k 实验（旧参数，已完成）

| 字段 | 口径 |
| --- | --- |
| run ID | `glmocr_mthv2_decoder_lora_lr1e5_20k_5gpu_a100_official_layout_260912_v1` |
| 入口/会话 | `tools/training/run_glmocr_a100_decoder_lora.sh`；tmux `glmocr_a100_plain_260912` |
| 训练 | A100 五卡同步 DDP；global batch `5`；`20000` steps；seed `42` |
| 数据 | full MTHv2：train/validation/test = `2159/240/800`；whole-page；512 queries |
| 模型与目标 | geometry；Hungarian；full layout loss；FP32 adapter；fast processor；`L_official + 0.2 L_layout`；natural-loop 等附加训练目标关闭 |
| decoder LoRA | rank `8`、alpha `8`、dropout `0`；学习率 `1e-5` |
| checkpoint/validation | `5000/10000/15000/20000`；validation-only selection |
| protocol 边界 | smoke、training、validation 不读取 test；selection 后才执行 locked test |
| 当前状态 | 已完成 step 5000 固定 checkpoint 的 direct test；未做 validation selection |

该 A100 run 与 BSCC 旧计划共享训练代码和公共 DDP launcher，仅 world size/global batch 不同。A100 smoke 已确认 checkpoint reload finite、decoder LoRA finite、natural-loop disabled、loss objective 正确且 test-free；正式训练按用户要求在 checkpoint-5000 后停止，跳过 validation，随后完成 800 页 direct test。后续新 run 使用本登记表顶部的参数优选配置和新 run ID。

## 历史记录：2026-09-12 natural predicted-loop A1 warm-start continuation（累计 1024 步）

以下记录只保留已完成的 natural-loop 诊断证据，不属于当前 BSCC 训练方案；当前方案不启用 natural-loop。

| 项目 | 配置/结果 |
| --- | --- |
| run ID | `glmocr_natural_rollout_A1_warmstart256_260912_v1`（256 步）→ `glmocr_natural_rollout_A1_warmstart256_cont768_260912_v1`（接续 768 步） |
| 训练口径 | seed42；五卡 DDP；geometry/full/Hungarian；512 queries；LoRA；plain generation；无 validation、无推理期循环干预 |
| 接续语义 | 从 finite `checkpoint-256` 加载 adapter 与 decoder LoRA；接续阶段重新建立 optimizer/scheduler；累计 optimizer updates=`1024`，summary 局部 selected step=`768` |
| natural-loop 训练信号 | mean loss=`2.653688`；mean weighted loss=`0.132684`；active-page ratio=`0.946614`；说明 rollout-based 分支实际激活 |
| checkpoint | `checkpoint-768` finite；训练 `test_manifest_read=false`、`test_used_for_selection=false` |
| locked test | 五卡五 shard 完成；800 页；CER=`1.183537`；insert/delete/substitute=`224004/15646/72025`；循环页率=`0.228750`；长度上限率=`0.293750`；EOS=`0.706250` |
| 对照 | plain baseline 800 页 CER=`0.874635`、循环页率=`0.195000`、长度上限率=`0.207500`；warm-start A1 CER 恶化 `+0.308902` |

该 run 证明自然循环惩罚有非零训练梯度，但没有改善自由生成，反而增加插入和触顶。由于采用 `256+768` warm-start 且接续阶段重置 optimizer/scheduler，不能把它当成与从基础权重单次连续 1024 步的严格等价对照；详细证据见 `docs/实验日志/GLMOCR/训练退化诊断/GLMOCR-B-260912-001.md`。
