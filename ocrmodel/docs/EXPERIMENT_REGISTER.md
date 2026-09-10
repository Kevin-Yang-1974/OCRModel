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
| `glmocr_mthv2_attribution_128_retry_v1` | seed42 三组容量归因：A `no-op`；B 固定 gate `adapter-only`；C 固定 gate＋decoder-LoRA（rank 8、alpha 8、dropout 0）；五卡 DDP；128 steps；32 页 validation；无 selection | 同一 MTHv2 train manifest；同一确定性 validation32 manifest（32 页） | GLM-OCR 固定 revision；三组均从同一基础权重起点；B/C 训练预算一致 | bundle `complete`；A/B 复用已完成 run，C 使用新 run ID 重试完成；所有 checkpoint finite，residual relative norm A/B/C=`0/0.001505/0.001438` | validation CER A/B/C=`0.787648/0.786034/0.794012`；B−A=`−0.001613`，C−B=`+0.007978`；A/B/C teacher-forced OCR loss=`1.666193/1.664493/1.663769`。C 的 layout box MAE=`0.077862`、validity AUROC=`0.953366`，但 invalid gated context share=`0.798970`、p gap=`0.019232`，尚未达到 query 消除阈值；B validity AUROC=`0.494084`、invalid share=`0.924753` | 不读取；`test_manifest_read=false`、`test_used_for_selection=false` | A=`glmocr_mthv2_attribution_128_v1_A_noop`，B=`glmocr_mthv2_attribution_128_v1_B_adapter_only`，C=`glmocr_mthv2_attribution_128_retry_v1_C_decoder_lora`；原 C run 因 NCCL watchdog 超时保留，retry 将 DDP timeout 提高到 3600s；容量归因结论：LoRA 明显改善布局/validity，但在 128 steps 下未转化为 OCR CER，反而较 B 回退 |

## 每个 run 必填

记录 Git commit、上游 GLM-OCR 版本与权重标识、manifest 哈希、R1/R2 配置、GPU 集合、训练预算、辅助损失权重、validation 选点规则和唯一 run ID。失败 run 保留原 ID，重试使用新 ID。

本表不继承 GOT2 LAVP 历史结果；如后续提取其指标、数据读取或 checkpoint 检查组件，必须单独登记来源 commit 和本分支改动。

## 当前筛选数据边界

MTHv2 原官方 split 是随机页级划分，没有书籍/版本元数据，不满足本项目的隔离要求。本轮不沿用该 split：将 `original_image` 中 `V...P...` 的卷号前缀作为版本/文档组代理，无卷号的数字页按 subset 合并为一组，再划分 train/validation/test。仅纳入 `1–32` 个文本行区域的整页，保证 32-query 无截断。选中页面还需通过 16×16 dHash、Hamming 距离不大于 4 的跨 split 近重复检查。这一卷号映射是可执行代理协议，不声称等价于完整书手或馆藏标注。

上述 `1–32` 个区域限制只适用于历史 128 页机制筛选，不适用于当前全量 MTHv2 DDP。全量协议使用 512 queries，最大区域数为 407，保留全部 2159/240/800 页；validity/no-object 目标只由训练期 Hungarian 匹配生成，布局真值和 query mask 不进入推理输入。
