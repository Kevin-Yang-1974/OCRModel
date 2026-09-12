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

## 运行记录

| ID | 配置 | 数据指纹 | checkpoint 起点 | 状态 | validation | test | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
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

结论：这次“相较上一个实验都下降”同时包含**协议不可比**（硬 EOS 截断被移除）和**真实训练退化**（新 checkpoint 的 TF loss 更高、A1 后程过拟合/梯度竞争）。该结论仅用于封存历史 natural-loop 诊断；当前不再继续 natural-loop/loop-escape 路线，BSCC 正式 run 回到 `L_official + 0.2 L_layout` 的 plain training objective。旧低 CER 也不能作为当前 BSCC 协议下的性能承诺。

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
| 当前状态 | 原 v1 与 rollout256 诊断 run 均不进入结果；新的 official-layout smoke `1494430` 已提交，正式训练 `1494433` 依赖 smoke 成功后运行 |
| 结果口径 | 用户授权的高 decoder-LoRA 学习率探索；不替代历史五卡协议，不与五卡结果作未校正的严格等预算比较 |

原 `glmocr_mthv2_decoder_lora_lr1e5_20k_4gpu_260911_v1` 仅设置了 `generation_mode=loop_recovery`，未设置 `--natural-loop-loss` 或 `--loop-escape-training`，因此没有循环训练信号；该 run 已取消并保留日志，不进入 validation/test。上一轮 rollout256 诊断 run 也不作为本轮训练依据。新的 official-layout run 固定使用 `L_official + 0.2 L_layout`，训练阶段关闭 natural-loop、scheduled sampling、loop escape 和 continuation head；`evaluate_glmocr_locked_test.py` 已按 metadata 注入并加载 decoder LoRA，四个 test shard 和 merge 阶段均强制检查 `decoder_lora_loaded=true`。

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
