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

### 2026-09-22：decoder-mask head-only 4M 五卡串行筛选（已完成）

| 字段 | 当前记录 |
| --- | --- |
| bundle / arm | 主 run `glmocr_decoder_mask_headonly4m_v1`（G1 已完成）＋恢复 run `glmocr_decoder_mask_headonly4m_v1_retry_ddp3600`（从 G2 起点续跑 G2/G3）；`B0` 为同协议独立 zero-shot 评测，不训练 |
| 分支 / commit | `glm-ocr-layout-mask-routing` / `e1667b2` |
| 入口 | `ocrmodel/tools/training/run_glmocr_decoder_mask_headonly_a100.sh`；G1→G2→G3 串行，每臂一次 `torchrun` |
| 远端产物根 | 主 run：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_decoder_mask_headonly4m_v1`；恢复 run：同目录下 `glmocr_decoder_mask_headonly4m_v1_retry_ddp3600` |
| 模型 / 数据 | GLM-OCR revision `ca5d8b3e287e52589e37c28385d9655ee4372f9d`；MTHv2 whole-page character manifest；train `128` 页、validation `64` 页、seed `42`；train manifest SHA256 `d1193dca63e10cd46003f12afceff906e9ebca19817f04c70450a98c90d6dae1`；validation manifest SHA256 `fbe971e01c0bb2dee03903be75bf57655cd18d14bab3b4c1b5c7068a2785434c` |
| 训练配置 | `max_pixels=4000000`；processor `slow`；`max_steps=1024`；checkpoint / validation step `256/512/768/1024`（本轮 validation 选点实际配置为 `1024`）；head-only；主干冻结；不注入 LoRA；CE 跳过 |
| 五卡语义 | A100 `0,1,2,3,4`；`torchrun --nproc_per_node=5`；每步对 router 参数执行 all-reduce 梯度平均；`NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1` |
| 可训练参数 / 损失 | 仅 mask router；router LR `1e-4`，主 LR 参数保留为 `1e-6`；balanced BCE＋Dice（Dice weight `1.0`、mask loss weight `0.2`）；当前入口同时配置 stop BCE weight `0.05`，G3 另含 VAE KL weight `1.0`，后续结果记录按实际 fingerprint 展开 |
| 当前状态 | 主 run 已 `failed` 但完整保留；恢复 run 的 G2/G3 已 `complete`，G1 由主 run checkpoint 只读复用；12 组四步波形 validation 与 B0 frozen-base zero-shot 均已完成；未读取 test，`test_manifest_read=false`、`test_used_for_selection=false` |
| B0 状态 | `b0_zero_shot_v2_frozen_base` 已完成：同一 64 页 validation、4M、1536 tokens；CER `0.2870459219`，I/D/S `2312/1630/2340`，K1/K3/K5 recall `0.645161/0.644995/0.674716`；`routing_mode=none`、`head_only=true`、`training_updates=0`、`lora_injected=false`、`mask_pages=0`；不读取 test、不参与 selection |
| 健康判据 | `mask_mean` 约 `0.02–0.1`；`pooled_peak` 随训练上升并接近 `1`；`dice` 为 `1−Dice` 且越低越好；显存 `29–40 GB` 为压线区，OOM 立即处理 |
| 已修复契约 | 五卡拓扑必须保留 `NCCL_P2P_DISABLE=1`；`rasterize_polygon` 使用 `sign.sum(dim=-1).abs() == K`，不得改回 `sign.abs().sum() == K` |
| 后处理 | 训练完成后用合并前单份 checkpoint 做 `evaluate_decoder_mask_waves.sh`，`arms=G1,G2,G3`、`steps=256,512,768,1024`、`max_pixels=4000000`、`max_new_tokens=1536`；随后补 B0 zero-shot |
| 结果 | G1/G2/G3 训练、12 组波形 validation、B0 zero-shot 和文档核验均已完成；本表不包含 test 结果 |

**健康检查 1（2026-09-22 00:43，Asia/Shanghai）**：G1 仍在运行，已生成 `step-256`；rank-0 checkpoint 含 13 个 tensor、989451 个参数，远端 finite 检查通过（`all_finite=true`）。`trainable_report` 确认 `world_size=5`、`decoder_lora_parameters=0`、`router_parameters=989449`。G2/G3 尚未启动；未发现 CUDA/OOM/NaN/Inf/Traceback。

**健康检查 2（2026-09-22 01:06，Asia/Shanghai）**：G1 已生成 `step-512`；checkpoint 含 13 个 tensor、989451 个参数，远端 finite 检查通过。五个 rank 仍在运行，G2/G3 尚未启动，错误扫描为空。连续两次健康检查完成，监控已按长程训练切换为每 1 小时一次；阶段切换、异常和指标就绪仍立即检查。

**故障与恢复（2026-09-22 02:07，Asia/Shanghai）**：主 run 在 G1 step-1024 checkpoint 已写完后失败。`train.log` 报告 `WorkNCCL(SeqNum=11266, OpType=ALLREDUCE, NumelIn=1, Timeout(ms)=600000)`，随后 `SIGABRT`；对应尾部同步是 rank-0 单独执行 64 页、4M、`max_new_tokens=1536` validation 时，其余 rank 在验证后的 `dist.barrier()` 等待超过默认 600 秒。该 `NumelIn=1` 集合与梯度 all-reduce 不同，故诊断为 rank-0 validation 超时而非 `NCCL_P2P_DISABLE=1` 修复失效。原失败产物保留，不覆盖。

**恢复启动（2026-09-22 02:07，Asia/Shanghai）**：已将 DDP `init_process_group` 的 timeout 显式设为 `3600s`，launcher 增加 `--start-arm`，并以新 run ID `glmocr_decoder_mask_headonly4m_v1_retry_ddp3600` 从 G2 启动；G1 复用主 run 的四个 finite checkpoint，不重训。恢复 run 初始状态为 `running`，launcher 已报告五卡 `world_size=5` 的 G2 启动；当前等待新的 checkpoint/log 进展后进行两次恢复期健康检查，再降回每小时监控。

**恢复期健康检查 1（2026-09-22 02:21，Asia/Shanghai）**：恢复 run `status/screen.json=running`，G2 训练相关进程仍在，G2 日志更新时间推进到 02:12，处于权重加载／首批 forward 阶段，尚未形成 step checkpoint。五卡显存约 `35–39 GB`，采样时部分 GPU 利用率达到 `100%`；错误扫描未发现 CUDA/OOM/NaN/Inf/Traceback。当前未把“尚无 checkpoint”判为失败，继续按 5 分钟频率等待实际 step 产物。

**恢复期健康检查 2（2026-09-22 02:33，Asia/Shanghai）**：恢复 run 仍为 `running`；五个训练 rank 均在持续计算（进程状态 `Rsl`、CPU 约 `99%`），五卡 `pmon` 均有实际 GPU 工作，显存约 `29–40 GB`。G2 仍处于 4M 首步计算，尚未形成 step checkpoint 或 train summary；错误扫描为空。两次恢复期健康检查均未见异常，监控已切换为每 1 小时一次；阶段切换、checkpoint/指标就绪和故障仍立即处理。

**恢复进展（2026-09-22 03:35，Asia/Shanghai）**：G2 已形成 `step-256`（02:34:43）、`step-512`（02:57:15）和 `step-768`（03:19:31）；每个目录均包含 `decoder_mask.safetensors`、`training_state.pt`、配置和 fingerprint。远端 finite 检查均通过，均为 13 个 tensor、989451 个参数；fingerprint 确认 `head_only=true`、`target_mode=token`、router LR `1e-4`。恢复 run 仍为 `running`，G3 尚未启动；step-1024 前按约定尚无 `train_log.jsonl`／validation summary，错误扫描为空。

**G2 完成、G3 启动（2026-09-22 04:35，Asia/Shanghai）**：G2 已完成 step-1024；`step-256/512/768/1024` 四个 checkpoint 均 finite（13 个 tensor、989451 个参数），无 CUDA/OOM/NaN/Inf/Traceback。G2 step-1024 最后训练记录为 `loss=0.234630`、`bce=0.193758`、`dice=0.978062`、`mask_mean=0.125300`、`pooled_peak=0.972569`；step-1 对照为 `mask_mean=0.020167`、`pooled_peak=0.043633`，说明 pooled peak 已升高且 mask_mean 仅略超预设低位，未接近 0.9 退化区。64 页 validation：CER `0.4772675348`，I/D/S `5285/1364/3796`，reference characters `21885`，exact-page rate `0`，K1/K3/K5 recall `0.579793/0.659191/0.666466`。G3 已启动并形成 `step-256`，checkpoint finite（29 个 tensor、5299228 个参数）；恢复 run 仍为 `running`，等待 G3 的 512/768/1024 与 validation 指标。

**G3 中段进展（2026-09-22 05:36，Asia/Shanghai）**：G3 已形成 `step-256`、`step-512`、`step-768`，对应 checkpoint 均 finite（29 个 tensor、5299228 个参数）；fingerprint 为 `head=vae`、`target_mode=token`、`head_only=true`。恢复 run 仍为 `running`，G3 尚未形成 step-1024 的 train log／validation summary，错误扫描为空。

**G3 完成与波形评测启动（2026-09-22 06:37，Asia/Shanghai）**：G3 已完成 step-1024，恢复 run `status/screen.json=complete`，五卡已释放；四个 G3 checkpoint 均 finite（29 个 tensor、5299228 个参数）。G3 step-1024 最后训练记录为 `loss=0.288307`、`bce=0.210395`、`dice=0.975277`、`mask_mean=0.105160`、`pooled_peak=0.963335`、`kl_raw=0.016403`、`kl_term=0.050000`。64 页 validation：CER `0.4642449166`，I/D/S `5920/1700/2540`，reference characters `21885`，exact-page rate `0`，K1/K3/K5 recall `0.685419/0.744063/0.747816`。随后已把主 run G1 checkpoint 复制到恢复 bundle 的 `arms/G1`（原目录保留），启动 `evaluate_decoder_mask_waves.sh` 共 12 个 job，输出目录为 `eval_mask_waves_v1`，配置为 4M、`max_new_tokens=1536`、validation64，首波任务正在加载模型且无错误。

**波形评测进展（2026-09-22 07:11，Asia/Shanghai）**：12 个 job 中首波 5 个已完成并生成 `summary.json`／`predictions.jsonl`：G1@256 CER `0.371990`、G2@256 `0.485767`、G3@256 `0.459630`、G1@512 `0.582317`、G2@512 `0.400183`；均为 64 页、4M、`max_new_tokens=1536`、`test_used_for_selection=false`。第二波任务已启动，当前 GPU 资源正常，未发现真实 CUDA/OOM/NaN/Inf/Traceback 错误。

**波形评测完成（2026-09-22，Asia/Shanghai）**：12/12 job 均返回 `status=complete`；每行均为 64 页、reference characters `21885`、`exact_page_rate=0`、4M、`max_new_tokens=1536`、`test_used_for_selection=false`。完整关键指标如下（I/D/S 为插入/删除/替换）：

| arm / step | CER | I/D/S | K1 / K3 / K5 recall | generation-limit hit | elapsed (s) | tok/s |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| G1 / 256 | 0.371990 | 4297 / 1344 / 2500 | 0.688172 / 0.682147 / 0.708097 | 0.0625 | 793.6 | 38.14 |
| G1 / 512 | 0.582317 | 7928 / 786 / 4030 | 0.559140 / 0.575851 / 0.614347 | 0.1406 | 887.0 | 39.18 |
| G1 / 768 | 0.480146 | 5688 / 1156 / 3664 | 0.505376 / 0.519092 / 0.557528 | 0.1094 | 842.7 | 38.84 |
| G1 / 1024 | 0.478821 | 5644 / 1533 / 3302 | 0.508065 / 0.514964 / 0.556818 | 0.0938 | 858.4 | 37.22 |
| G2 / 256 | 0.485767 | 6211 / 1140 / 3280 | 0.623656 / 0.633643 / 0.660511 | 0.1094 | 1107.8 | 30.05 |
| G2 / 512 | 0.400183 | 4740 / 1358 / 2660 | 0.650538 / 0.641899 / 0.664063 | 0.0625 | 787.2 | 38.99 |
| G2 / 768 | 0.560612 | 7363 / 1144 / 3762 | 0.500000 / 0.510836 / 0.542614 | 0.1250 | 886.9 | 38.63 |
| G2 / 1024 | 0.477268 | 5285 / 1364 / 3796 | 0.473118 / 0.485036 / 0.522017 | 0.0938 | 816.5 | 38.49 |
| G3 / 256 | 0.459630 | 5183 / 1809 / 3067 | 0.591398 / 0.573787 / 0.605824 | 0.0781 | 810.7 | 37.81 |
| G3 / 512 | 0.483345 | 5757 / 1161 / 3660 | 0.508065 / 0.507740 / 0.545455 | 0.0938 | 861.7 | 37.83 |
| G3 / 768 | 0.537857 | 6700 / 804 / 4267 | 0.489247 / 0.518060 / 0.555398 | 0.1250 | 885.9 | 38.19 |
| G3 / 1024 | 0.464245 | 5920 / 1700 / 2540 | 0.607527 / 0.616099 / 0.644886 | 0.0781 | 852.4 | 37.06 |

**B0 启动与恢复（2026-09-22，Asia/Shanghai）**：波形评测 12/12 完成后，B0 首次尝试因手工启动遗漏系统 CUDA/CUPTI 路径失败，第二次尝试因旧 evaluator 的 `routing_mode=none` 分支默认要求 `lora.safetensors` 失败；两次产物和日志均保留。第三次以 `b0_zero_shot_v2_frozen_base` 启动，fingerprint 明确为 `routing_mode=none`、`head_only=true`、`zero_shot=true`、`training_updates=0`，不注入 LoRA、不加载 mask head；当前运行中，使用 GPU0、同一 64 页 validation、4M、`max_new_tokens=1536`，test 不读取。

**B0 完成（2026-09-22 08:06，Asia/Shanghai）**：B0 `status=complete`，64 页 frozen-base zero-shot 已生成 `summary.json`／`predictions.jsonl`，GPU0 已释放。指标为 reference characters `21885`、character errors `6282`、I/D/S `2312/1630/2340`、CER `0.2870459219`、exact-page rate `0`、K1/K3/K5 recall `0.645161/0.644995/0.674716`、generation-limit hit rate `0.03125`、elapsed `641.3s`、`42.88 tok/s`、`mask_pages=0`；协议字段为 `max_pixels=4000000`、`max_new_tokens=1536`、`routing_mode=none`、`head_only=true`、`zero_shot=true`、`training_updates=0`、`lora_injected=false`、`test_used_for_selection=false`。

**最终核验（2026-09-22，Asia/Shanghai）**：主失败 run 和两次失败 B0 尝试均保留；恢复训练 run `complete`；G1/G2/G3 四步 checkpoint 均 finite；波形 validation `12/12` complete；B0 summary complete；所有评测均使用同一 64 页 validation、4M 和 `max_new_tokens=1536`，未读取 test。`NCCL_P2P_DISABLE=1` 和 `rasterize_polygon` 的 `sign.sum(dim=-1).abs() == K` 契约均保留。

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
| `glmocr_dunhuang_layout_branch_finetune_compare_20260919_v1` | **受控 with/without 分支对比**：三臂共用 stage-1 `giou20x_10000` 起点、同数据同步数同学习率、4M 原生分辨率。N=`content_only`（分支旁路），W=`layout_ot`（gate 起点 0.01），Wx=`layout_ot`（gate 起点 0.03=cap，残差钉在上限）；各 256 步 | 80 页 validation：N `0.112988`、W `0.111981`、Wx `0.111795` | 不读取 | **W − N 显著（CI [−0.001691, −0.000368]，P=0.999）→ 带布局分支的微调显著优于不带分支的微调**，为本项目首次受控证明；N 的 CER 与官方基座 zero-shot @4M 逐位相同（均 3030 处编辑），即不带分支的微调什么都没改变；Wx 点估计最好但逐页方差变大、不显著。**归因未收口**：W 与 N 还差 sem_adapter 的可训容量（8.46M vs 3.73M），且尚不能断言增益来自布局信息本身，需补随机查询机制/uniform transport 对照 |
| `glmocr_dunhuang_sem_adapter_nativeres256_from_parent256_20260919_v1` | 敦煌地方志；**原生分辨率** `max_pixels=4000000`；从父 run `checkpoint-256` 续训 256 步，gate 续接 `0.012923`；其余超参与 09-18 续训相同；64 步一存档 + 5 路并行 eval-only（全部 4M） | 并行验证选 step `256`；CER `0.111496`（80 页）；四个候选点 `0.111496–0.111795`，**step-0 恒等对照 `0.111608`**（与消融 C 臂逐位相同） | 59 页；CER `0.136879`；I/D/S `524/405/1014`；触顶 `0`；trigram `0.0485` | 门控轨迹与 1M 那次几乎逐位相同（`0.012934 → 0.015487`），教师强制损失均值 `0.9996`（1M `1.1418`）；256 步耗时 `764.9 s`（1M `163.9 s`），单卡峰值 31–37 GB / 40 GB；**step-256 vs step-0 配对 CI [−0.000618, +0.000512]，不显著**——原生分辨率下同样无训练空间；test 上显著优于 1M（CI [−0.0409, −0.0244]），为本项目锁定 test 最好值 |

阶段二结论：同一套 stage-2 配置在 MTHv2 上把 CER 从 `0.393093` 推到 `0.443434`（插入占比 `0.622`），换到敦煌目标域后 CER `0.170694`、插入占比 `0.196`、触顶 `0`——支持「stage-2 应在目标域执行」的设计判断。

**续训验证（09-18 补）**：在父 run 之上续训 256 步并每 64 步存档验证，四个候选点 `0.207219–0.208077`、step-0 恒等对照 `0.207331`，全部持平（0.0001 量级）；锁定 test `0.169778` 对父 run `0.170694` 仅降 `0.0009`，而同一间隔的 validation CER 完全不动。结论：**该配置（主 LR `1e-5`、decoder LR `1e-6`、gate 上限 `0.03`、256 步）已无训练空间**，日志里 gate 仍在单调上升不构成「还有空间」的证据。

**zero-shot 对照（09-18 补）**：官方 GLM-OCR 基座在同一 59 页 test 上 CER `0.173864`、80 页 validation `0.210687`。整条 stage-1 布局精修 + stage-2 语义适配器链路的净收益因此只有 `0.004–0.006` CER（test）/ `0.0035`（validation），且所有微调 run 都挤在 `0.1677–0.1736` 这 `0.006` 宽的带内——**基座在该 test 上已接近饱和，既有指标区分度低于 59 页样本噪声**。此外 zero-shot 的插入占比 `0.195`、触顶 `0`、trigram `0.040` 与微调 run 一致，说明敦煌 2k 步那两次的过生成崩坏（插入占比 `0.66`/`0.79`）是训练过久引入的，不是基座固有行为。

**仍未解决**：`sem_adapter` 相对 `content_only`／`geometry` 的必要性——上面两条对照都不构成 `gate=0` 冻结 vs `gate` 可训的同 mode 消融。要论证增益，需在同一起点、同一 split 上补这一格。

**原生分辨率续训（09-19 补）**：把分辨率修到原生后重跑同一续训（`glmocr_dunhuang_sem_adapter_nativeres256_from_parent256_20260919_v1`），四个候选点 `0.111496–0.111795`、step-0 恒等对照 `0.111608`，配对 CI [−0.000618, +0.000512] **不显著**——**训练在两个分辨率下都已饱和**。锁定 test `0.136879`（59 页），较 1M 同配置 `0.169778` 显著改善 （CI [−0.0409, −0.0244]），为本项目最好值。

**本段的总结论**：真正的大杠杆是输入分辨率而不是训练——`max_pixels=1003520` 把每页（原生 310–367 万像素）压到约三分之一，官方 processor 默认 `longest_edge` 是 `9633792`；仅把它放到原生、零训练即得 −0.0977（validation）。而 stage-2 在 1M 与 4M 下的续训效应都不显著，微调起点相对基座的增益（4M 下 0.0014）也不是靠继续训练能扩大的。仍有证据支持、尚未证伪的训练侧方向只剩一个：`content_gate` 上限（消融显示 `0 → 0.0129 → 0.03` 单调改善，而训练从未触顶）。

**仍未解决**：`sem_adapter` 相对 `content_only`／`geometry` 的必要性 eval-only（`glmocr_dunhuang_stage2_ablation_resolution_20260918_v1`，80 页 validation，逐页配对 bootstrap）。三条结论：

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

## 2026-09-22 decoder-mask oracle 对照 + 128 步修复重训（五卡单进程，已完成）

| 字段 | 口径 |
| --- | --- |
| screen ID | `glmocr_decoder_mask_oracle_128step_v1` |
| 分支/commit | `glm-ocr-layout-mask-routing`；HEAD `e1667b2`；叠加未提交改动（mask-loss 增 centroid+sparsity、LR warmup/cosine、`target_mode="line"`、oracle 生成工具、五卡 launcher） |
| 入口 | `tools/training/run_glmocr_decoder_mask_128step_oracle_a100.sh`；远端产物根 `/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_decoder_mask_oracle_128step_v1` |
| 数据协议 | 128 训练页 / 64 validation 页，seed 42；`manifest.char.jsonl`；隔离单元与近重复检查沿用上一轮 128 页筛选代理；`test` 未读取 |
| 组别 | GPU0=G1(window)、GPU1=G2(token)、GPU2=G3(vae) 三训练臂；GPU3=oracle line、GPU4=oracle window(3-5) 两上限；每臂单进程（非 DDP）、per-device batch 1、无梯度累积；effective batch 1 |
| 训练配置 | max-steps `128`、head-only、routing learned、split-layer `8`、dim `256`、bias-max `2.0`、bias-warmup `100`；FP32 adapter；base LR `1e-6`、router LR `1e-4`、router warmup `32` 步、cosine 退火到 `0.1×`（`router-lr-min-ratio 0.1`）；checkpoint/validation every `64`、validation-steps `128`；可训练参数仅 mask head（G1/G2 MLP ≈989K、G3 VAE ≈5.3M） |
| 损失改动（本轮唯一变量） | 相对上一轮 head-only 屏（密集列级失败），唯一变量 = 损失函数与 LR schedule：`mask_and_dice_loss` 增 `centroid_weight 1.0`（质心定位，零重叠可导）与 `sparsity_weight 1.0`（L1 质量，反密集），保留 balanced BCE + Dice(1.0)；mask-loss-weight `0.2`、stop-loss-weight `0.05`；gate 初值未覆写（默认） |
| oracle | 自由生成期按位置注入 GT mask（line / 3-5 window），`bias-max 2.0`；`reads_ground_truth=true`、`usable_for_selection=false`、`test_used_for_selection=false` |
| 状态 | **已完成**。三训练臂 step-128 validation CER：G1(window)=`0.4445`、G2(token)=`0.3867`、G3(vae)=`0.2963`，**均劣于 B0 `0.287`**；oracle line=`0.356`、window=`0.297`，也劣于 B0。失败模式为插入/过度生成（I/D/S 全线偏高，G1 I=5251、oracle line I=3623）。**mask 仍未定位**：mask_mean `0.20–0.24`（目标仅 ~0.005）、dice `~0.99`、centroid `~0.10`、G3 `kl_active_dims=0.0`——centroid+sparsity 损失未能把密集 mask 压下去。结论见下方 bias 扫描 |
| protocol 字段 | `test_manifest_read=false`；`test_used_for_selection=false`（机制筛选/诊断，不进入 selection） |

## 2026-09-22 decoder-mask bias 强度扫描（window oracle 五档，运行中）

| 字段 | 口径 |
| --- | --- |
| screen ID | `glmocr_decoder_mask_bias_sweep_v1` |
| 分支/commit | `glm-ocr-layout-mask-routing`；HEAD `e1667b2`；叠加未提交改动（新增 `tools/training/run_glmocr_decoder_mask_bias_sweep_a100.sh`） |
| 入口 | `tools/training/run_glmocr_decoder_mask_bias_sweep_a100.sh`；远端产物根 `/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_decoder_mask_bias_sweep_v1` |
| 动机 | beta=2.0 的 window oracle（0.297）与 line oracle（0.356）均劣于 B0 `0.287`，失败模式为插入/过度生成；判定「加法偏置」在 2.0 下无法把 routing 信息转正，需扫 `bias_max` 找强度拐点（是否存在任一强度使 GT 注入持平或优于 B0） |
| 数据协议 | 复用上一屏同一 64 页 validation manifest（`split/validation64_screen_seed42.jsonl`）与 128 页 train manifest（仅 K1/K3/K5 指标用）；隔离单元与近重复检查沿用；`test` 不读取 |
| 组别 | 五档 `bias_max` = `0.1 / 0.25 / 0.5 / 1.0 / 1.5`，各占一张卡（GPU0–4），`window(3-5)` 模式 GT 注入；每臂单进程（非 DDP）、per-device batch 1 |
| 唯一变量 | `--bias-max`；其余与上一屏 oracle 完全相同（同 model `ca5d8b3e…`、同 manifest、同 `max_pixels=4M`、同 `max-new-tokens=1536`、同 `processor-mode slow`） |
| oracle 语义 | `reads_ground_truth=true`、`usable_for_selection=false`、`test_used_for_selection=false`；纯诊断，不进入 checkpoint selection |
| 状态 | **已完成**。五档 window oracle CER：β=0.1 `0.287`、β=0.25 `0.2852`、β=0.5 `0.2883`、β=1.0 `0.2865`、β=1.5 `0.2924`（对照 B0 `0.287`、β=2.0 `0.297`）；**整条曲线平、无剂量效应**，最好点 β=0.25 仅比 B0 好 `0.0018`（64 页噪声量级）。GT mask 注入在该机制下中性——与 attention-routing 的 −20% 陡崖矛盾，混淆变量待隔离（模型 trained vs zero-shot / 框大小单字 vs 窗行 / 指针 synced vs positional / 打分 hard vs soft 峰值归一） |
| protocol 字段 | `test_manifest_read=false`；`test_used_for_selection=false` |

## 2026-09-22 decoder-mask hardsync window beta=1（149 页五卡快速分片，已完成）

| 字段 | 口径 |
| --- | --- |
| run ID | `glmocr_decoder_mask_window_beta1_5gpu_quick_20260922_045950` |
| 分支/commit | `glm-ocr-layout-mask-routing`；HEAD `e1667b2`；叠加未提交改动 |
| 远端产物根 | `/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_decoder_mask_window_beta1_5gpu_quick_20260922_045950` |
| 模型 / checkpoint | base revision `ca5d8b3e287e52589e37c28385d9655ee4372f9d`；trained layout+decoder checkpoint `glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000` |
| 数据协议 | 同一 149 页 validation `manifest.char.jsonl`，SHA256 `36ec845875e1ea18a48d4a523b6c5a3f007b46e2a07de0cafd2c26139929a348`；train manifest 仅用于字符统计；`test` 未读取 |
| 固定推理协议 | window `3–5` 字；`bias_max=1.0`；processor `fast`；`max_pixels=4000000`；`max_new_tokens=1536`；math-SDPA 确定性执行；hard mask；synced pointer；`split_layer=0`（全部层注入）；seed `42` |
| 五卡语义 | GPU `0,1,2,3,4` 各运行一个单卡 worker；149 页按原 manifest round-robin 分为 `30/30/30/30/29` 页五个 shard，按 I/D/S 与 reference characters 聚合；**不是 DDP，也不是五次重复 run** |
| 状态 | **已完成**。五个 shard 均 `status=complete`，无 CUDA/OOM/NaN/Inf/Traceback；五卡已释放 |
| 汇总结果 | reference characters `41654`；character errors `5704`；CER `0.1369376290`；I/D/S `948/769/3987`；generation tokens `50321`；generation-limit hits `0/149`（`0.0`） |
| oracle 语义 | 五个 summary 均为 `reads_ground_truth=true`、`usable_for_selection=false`、`test_used_for_selection=false`；纯机制诊断，不进入 selection |
| 结论 | 相对同 checkpoint、同 149 页协议的 B0 `CER=0.1699236568`，window beta=1 降至 `0.1369376290`，绝对下降 `0.0329860278`、相对约 `19.4%`。这说明 trained checkpoint 下 3–5 字 window 的 GT routing oracle 已出现明显收益；但它仍是 GT-mask 上限，不能直接证明 decoder 预测 mask 已经有效，也不能与 attention-tracking 的单字符框结果作严格同口径归因。 |

| shard | pages | CER | I/D/S | generation tokens | limit hits |
| ---: | ---: | ---: | --- | ---: | ---: |
| 0 | 30 | `0.1741883491` | `385/354/774` | `10514` | `0` |
| 1 | 30 | `0.1437110834` | `103/132/919` | `9575` | `0` |
| 2 | 30 | `0.1197111587` | `145/102/814` | `10853` | `0` |
| 3 | 30 | `0.1101161665` | `127/100/683` | `9801` | `0` |
| 4 | 29 | `0.1364742030` | `188/81/797` | `9578` | `0` |


## 2026-09-22 line100-window GT acceptance

`glmocr_line100_window_gt_accept_20260922_145532`：已登记待启动，a100-yky GPU0，完整 sparse24 validation149，GT 3–5 字 hard window + synced + beta1，移除小版面分支；固定 checkpoint-3000，不训练、不读取 test、不做 selection。验收 CER <0.13，成绩待完成；详细配置、源码指纹与产物位置见 [实验日志](实验日志/GLMOCR/架构收益对照/GLMOCR-line100-window-gt-acceptance-20260922.md)。

**启动确认**：`glmocr_line100_window_gt_accept_20260922_145532` preflight通过，GPU0 admission=0%，tmux存在，status=running；Torch2.8.0+cu128/Transformers5.3.0。heartbeat `line100-window-gt` 初始5分钟，两次健康后30分钟。完整CER待完成。

**五卡替换**：用户要求中止旧单卡run（15页保留，TERM143，非完整验收），新run `glmocr_line100_window_gt_accept_5gpu_20260922_150507` 待启动；GPU0–4，30/30/30/30/29页，原数据/模型/生成协议不变，单卡batch1，非DDP、不训练、不读test；合并完整149页后按micro CER<0.13判定。详情追加在同一实验日志。

**五卡提交确认**：执行commit `21afa4a` 已推送远端分支；独立git archive SHA256 `819450ecebdeae89a51a4e3784f457784ca037727d7152bf2c3cabfe3b4c4469`，新run已派发，同一heartbeat已切换。

**五卡验收完成**：`glmocr_line100_window_gt_accept_5gpu_20260922_150507` complete；五片 `30/30/30/30/29` 合并完整149页，无重无漏；manifest SHA256=`36ec845875e1ea18a48d4a523b6c5a3f007b46e2a07de0cafd2c26139929a348`。GT 3–5字 hard window、synced、beta1、全层、prefill不注入、4M/1536/fast/math-SDPA/BF16、小版面分支移除；固定checkpoint-3000，不训练、不选点、不读test。完整 micro CER=`0.13693762903922793`，I/D/S=`948/769/3987`，reference characters=`41654`，generation tokens=`50321`，EOS=`149/149`，触顶=`0/149`，循环页=`0/149`、循环率=`0.0`；JSON 结果无非有限数值，日志无 CUDA/OOM/NaN/Inf/Traceback/异常退出。`acceptance.eligible=true`、`passed=false`（严格 CER<0.13 未通过）。产物已下载至 `D:/yangky/glm-ocr-assets/line100-window-acceptance/glmocr_line100_window_gt_accept_5gpu_20260922_150507/`；详情见实验日志。

**生成上限核对（2026-09-22，补登）**：验收 run 的 `results/summary.json` 与五片 `protocol.json` 均记录 `max_new_tokens=1536`，`validation.generation_limit_hits=0`，因此 `0.13693763` 不是生成预算截断的产物。历史整行 `line100`（`0.124694`）走 `tools/training/run_glmocr_layout_oracle_line_eval_a100.sh`，其 `max_eval_new_tokens` 同为 `1536`、触顶 `0`、分辨率同为 4M。两轮生成预算同口径，`0.13693763` 与 `0.124694` 的差可直接比较。

## 2026-09-22 line100-window 失败归因诊断（两臂，五卡并行，已完成）

验收未过线时，`LINE100_WINDOW_MASK_ROUTING.md` 规定先拆开「移除小版面分支」与「整行换 3–5 字窗口」两项同时变化，不得直接调 bias、指针、分辨率或评价子集。

| 字段 | 口径 |
| --- | --- |
| run ID | `glmocr_line100_legacy_line_diag_20260922_160451`（`--mode legacy-line`）、`glmocr_line100_layout_control_diag_20260922_160451`（`--mode gt --legacy-layout-control`） |
| 分支 / commit | `glm-ocr-layout-mask-routing` / `a980c4e` |
| 入口 | `ocrmodel/tools/evaluation/run_window_mask_acceptance_a100.sh <run_root> <mode> [--legacy-layout-control]` → `evaluate_window_mask_routing.py` |
| 源码指纹 | `git archive` 独立压缩包 SHA256 `a8224cef505a16f1792393564471a5ac8dff27fa277f877b568b120b249bd222`，已远端核验 |
| 远端产物根 | `/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/acceptance_runs/<run ID>` |
| 数据协议 | 同一 sparse24 validation 149 页 manifest，SHA256 `36ec845875e1ea18a48d4a523b6c5a3f007b46e2a07de0cafd2c26139929a348`；train 未读取，test 未读取 |
| 模型 / checkpoint | base `ca5d8b3e287e52589e37c28385d9655ee4372f9d`；`glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000`（decoder LoRA SHA256 `ada4cdd3bb1f2f417d1c0a68054dc61e70ab59f215b3bc2018074b4f2fdcd5d5`） |
| 固定推理协议 | `bias=1.0`、`synced` 指针、hard raster、prefill 不注入、fast processor、`max_pixels=4000000`、`max_new_tokens=1536`、BF16、math-SDPA、seed 42；全部 decoder 层 |
| 五卡语义 | 物理 GPU `0–4` 各一个单卡 worker，149 页 round-robin `30/30/30/30/29`；两臂各五个 worker 同时在跑（每卡两个进程），**不是 DDP、不是重复 run** |
| 组别设置 | 两臂与验收 run 共用同一起点 checkpoint、同一 manifest、同一生成协议，预算一致；唯一变量见下 |
| 唯一变量 | `legacy-line`：保留原 geometry 小分支、GT 整行框（复核 `0.124694` 与装置可比性）；`gt + --legacy-layout-control`：新路径但把小分支放回，隔离「小分支移除」与「整行换窗口」 |
| 产出边界 | 两臂均为诊断；`acceptance_eligible=false`，merge 只写 `results/merged.json`，**不写 `results/summary.json`**，不得读成验收结论 |
| protocol 字段 | `test_manifest_read=false`；`test_used_for_selection=false`；不训练、不选点、无优化器/学习率/梯度累积 |
| 结果 | 两臂均 complete。`legacy-line` CER=`0.12469390694771211`（替换/插入/删除 `3887/738/569`，每步命中 `77.62` token，gen tokens `50400`）；`layout_control` CER=`0.13693762903922793`（`3987/948/769`，每步命中 `8.38`，gen tokens `50321`）。两臂均 149 页、reference characters `41654`、`exact_page_rate=0`、`generation_limit_hits=0`，日志与 shard 日志错误扫描为空 |
| 装置可比性 | **成立**。`legacy-line` 复现历史整行值到小数点后 11 位（`0.12469390694771211` 对 `0.124694`）；其 routing 诊断与历史一致（`bias=1.0`、`pointer=synced`、`box_source=line`、`line_map=regions`、`gated_steps=0`、`missing_box_steps=1.32`） |
| 归因结论 | **小分支移除为精确零效应**：`gt+control` 与验收 `gt` 的 `validation_predictions.jsonl` SHA256 完全相同（`846acbede98c2c96c00e40408c529bb7c9334f669d8d622eaf24289d5baff37a`），三份副本逐一核对；控制臂 protocol 为 `layout_branch_present=true`，即小分支确实加载而 149 页输出一个 token 未变。**差距 100% 来自空间目标**：`legacy-line` − `gt` = `−0.012244`，逐页配对 bootstrap 按页 CI `[−0.022881, −0.003519]`、按卷分组 CI `[−0.043107, −0.003280]`，**两档均显著**（最坏情况仍支持 `−0.043107`）。失败签名与 `line100` 收益签名方向相反、量级相当：插入 `738→948`（+28.5%）、删除 `569→769`（+35.1%）、替换 `3887→3987`（+2.6%） |
| 机制读数 | 每步命中 token 从 `77.62`（整行）降到 `8.38`（3–5 字窗口），`B=1.0` 两臂相同。3–5 字窗口并非更精确的点目标，而是每个字符只在自己那一步被覆盖、相邻窗口几乎不重叠，故行级持续上下文丢失 |
| 归因边界 | 本轮为「框大小＋B＋页集多变量同变」的跨实验比较。按 `LAYOUT_ORACLE_LINE_RESULT.md` §7.1 预先划定的口径，**不能据此断定**是「每 key 强度」还是「覆盖重复度」在起作用；分离二者需刻意做「固定 B 变覆盖」或「固定覆盖变 B」的扫描 |
| 后续处置 | 验收未过线的原因已收口为空间目标形态，且按 `LINE100_WINDOW_MASK_ROUTING.md` 完成处置：未改动 bias、指针、分辨率或评价子集。下一步为受控单变量阶梯（整行 → 行内窗口 → 窗口＋锚点），先零 GPU 量出第三种形态的每步覆盖分布，再投 GPU |
| protocol 字段 | `test_manifest_read=false`；`test_used_for_selection=false`；`usable_for_selection=false`；`reads_ground_truth_for_routing=true`；`acceptance=null`；`acceptance_eligible=false`；不训练、不选点 |

**判读口径（写在跑之前，已完成核对）**：
1. `legacy-line` 复现 `0.124694`（或落在其误差内）→ 装置可比，后续差值可归因；不复现 → 先修装置，不读其余结论。**已复现，装置可比。**
2. `gt + --legacy-layout-control` 与验收 `0.13693763` 的差 = **小分支移除**单独贡献多少；该臂与 `0.124694` 的差 = **整行换 3–5 字窗口**单独贡献多少。**前者为精确 0；后者为全部 `0.012244`。**
3. 两个诊断臂都不作验收、不作选点，也不用于调整 bias 或窗口。**已遵守。**
