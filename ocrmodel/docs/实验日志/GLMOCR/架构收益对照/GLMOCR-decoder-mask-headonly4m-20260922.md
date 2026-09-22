# GLM-OCR decoder-mask head-only 4M 五卡筛选日志

## 1. 运行边界

- bundle：`glmocr_decoder_mask_headonly4m_v1`
- 分支：`glm-ocr-layout-mask-routing`
- commit：`e1667b2`
- 入口：`ocrmodel/tools/training/run_glmocr_decoder_mask_headonly_a100.sh`
- 远端代码：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/code/ocrmodel`
- 远端产物：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_decoder_mask_headonly4m_v1`
- 模型：`ca5d8b3e287e52589e37c28385d9655ee4372f9d`
- 设备：A100 `0,1,2,3,4`；world size `5`

本日志只记录本轮 head-only mask routing 实验，不与旧的 LoRA 训练臂、1M fine-grid 臂或 line-level 结果混合比较。

## 2. 固定协议与实现

| 项目 | 配置 |
| --- | --- |
| 输入 | MTHv2 whole-page image＋prompt；`max_pixels=4000000`；processor `slow` |
| 数据 | train `128` 页、validation `64` 页、seed `42`；train SHA256 `d1193dca63e10cd46003f12afceff906e9ebca19817f04c70450a98c90d6dae1`；validation SHA256 `fbe971e01c0bb2dee03903be75bf57655cd18d14bab3b4c1b5c7068a2785434c` |
| 训练 | `1024` steps；G1→G2→G3 串行；每臂一次 `torchrun --nproc_per_node=5` |
| 参数 | 主干无 autograd 图；不注入 LoRA；仅 router 可训练；router LR `1e-4`，主 LR 参数为 `1e-6` |
| 损失 | mask balanced BCE＋Dice，Dice weight `1.0`，mask loss weight `0.2`；当前入口还传入 stop BCE weight `0.05`，G3 传入 VAE KL weight `1.0`，最终以每臂 fingerprint 为准 |
| 分布式 | rank 按页面切片；每步 router 梯度 all-reduce 后平均；禁用 NCCL P2P/IB 以适配本机五卡拓扑 |
| 选择 | validation-only；test 不读取、不用于选点或后处理 |
| B0 | G1–G3 完成后单独执行冻结官方基座的 zero-shot validation；不训练、不参与 selection |

## 3. 状态快照

2026-09-22 00:17（Asia/Shanghai）观察到：

- `status/screen.json` 为 `running`。
- G1 五个 rank 已启动并完成权重加载，尚未写出 validation summary。
- G2、G3 尚未启动；这是 launcher 的串行执行顺序。
- 未发现活动 B0 进程；B0 由监控流程在训练臂完成后补齐。
- 未发现 CUDA/OOM/NaN/Inf/Traceback；本次快照尚无可报告指标。

健康判据固定为：`mask_mean` 约 `0.02–0.1`；`pooled_peak` 随训练上升并接近 `1`；`dice` 使用 `1−Dice` 定义，越低越好；显存 `29–40 GB` 属于本配置的压线区，OOM 需要立即停止当前失败 run、保留产物并以新 run ID 从明确恢复点续跑。

本轮不可回退的工程修复：五卡 A100-PCIE 拓扑必须保留 `NCCL_P2P_DISABLE=1`；`rasterize_polygon` 的内部判定必须是 `sign.sum(dim=-1).abs() == K`，不能替换为 `sign.abs().sum() == K`。

训练完成后的后处理固定为：对 `G1/G2/G3` 的合并前单份 rank-0 checkpoint 运行 `evaluate_decoder_mask_waves.sh`，覆盖 step `256/512/768/1024`，`max_pixels=4000000`、`max_new_tokens=1536`；随后运行同一 validation 协议的 B0 zero-shot。test 不读取。

## 4.1 健康检查 1：G1 step-256

2026-09-22 00:43（Asia/Shanghai）确认：

- `status/screen.json` 仍为 `running`，G1 五个 rank 仍存活并持续占用 GPU；G2/G3 尚未启动。
- `arms/G1/step-256/` 已形成，文件写入时间约为 00:40:30；rank-0 单份 checkpoint 包含 13 个 tensor，989451 个参数，远端 `torch.isfinite` 检查通过。
- `trainable_report.json`：`world_size=5`、`decoder_lora_parameters=0`、`router_parameters=989449`；与 head-only、无 LoRA 约束一致。
- 尚未到 step-1024，故 `train_log.jsonl`、validation summary 及 `mask_mean/pooled_peak/dice` 尚未落盘；错误扫描未发现 CUDA/OOM/NaN/Inf/Traceback。

## 4.2 健康检查 2：G1 step-512

2026-09-22 01:06（Asia/Shanghai）确认：

- G1 已形成 `step-512/`，文件写入时间约为 01:02:25；13 个 tensor、989451 个参数，远端 finite 检查通过。
- 五个 rank 仍存活，G2/G3 尚未启动；未发现 CUDA/OOM/NaN/Inf/Traceback。
- `train_log.jsonl` 和 validation summary 仍按约定等待 step-1024；因此当前尚无 `mask_mean`、`pooled_peak`、`dice` 或 OCR validation 指标。
- 连续两次健康检查完成，监控从 5 分钟切换为长程训练的 1 小时频率；阶段切换、故障和产物就绪仍立即检查。

## 4. 结果登记

以下字段待远端产物形成后追加：每臂 run fingerprint、trainable parameter count、checkpoint finite 检查、validation step 与 CER/I-D-S、EOS/触顶/循环率、mask 指标、loss 曲线关键点、B0 zero-shot summary。任何失败或中止 run 保留原产物，并以新 run ID 记录重试。

## 5. 故障与恢复

### 5.1 主 run 的 G1 尾部超时

2026-09-22 02:07（Asia/Shanghai）确认主 run `glmocr_decoder_mask_headonly4m_v1` 已失败，但 G1 的 `step-256/512/768/1024` 均已写出。失败日志为 `WorkNCCL(SeqNum=11266, OpType=ALLREDUCE, NumelIn=1, Timeout(ms)=600000)`，随后 rank 收到 `SIGABRT`；`NCCL_P2P_DISABLE=1` 仍在 launcher 中保留。`NumelIn=1` 对应尾部 `dist.barrier()`，不是每步 router 梯度 all-reduce；step-1024 checkpoint 写完后，rank-0 的 64 页 4M validation 超过了默认 600 秒，其他 rank 在验证后的 barrier 等待超时。原 run 及全部 checkpoint 保留为诊断产物。

### 5.2 新 run 的恢复边界

已在本地和远端同步以下最小修复：`train_decoder_mask.py` 支持显式 `--ddp-timeout-seconds`，恢复 launcher 使用 `3600` 秒；launcher 支持 `--start-arm G2`。新 run 为 `glmocr_decoder_mask_headonly4m_v1_retry_ddp3600`，远端状态在 02:07 已为 `running`，launcher 已报告 G2、五卡 `world_size=5` 启动。该 run 只负责 G2→G3，G1 后处理时引用主 run 的四个 finite rank-0 checkpoint；不重训 G1、不覆盖失败 run。恢复期先完成两次带实际 checkpoint/log 进展的健康检查，再回到每小时 heartbeat。

### 5.3 恢复期健康检查 1

2026-09-22 02:21（Asia/Shanghai）确认恢复 run 的 `status/screen.json` 仍为 `running`；G2 训练相关进程仍在，日志已推进到权重加载／首批 forward，尚未生成 step checkpoint。五卡显存约 `35–39 GB`，采样时部分 GPU 利用率为 `100%`；错误扫描未发现 CUDA/OOM/NaN/Inf/Traceback。继续按 5 分钟频率等待实际 step 产物。

### 5.4 恢复期健康检查 2

2026-09-22 02:33（Asia/Shanghai）确认恢复 run 仍为 `running`；五个训练 rank 均在持续计算，进程状态为 `Rsl`、CPU 约 `99%`，五卡 `pmon` 均有实际 GPU 工作，显存约 `29–40 GB`。G2 仍处于 4M 首步计算，尚未生成 step checkpoint 或 train summary；错误扫描为空。两次恢复期健康检查均通过，监控已切换为每 1 小时一次；阶段切换、产物就绪和故障仍立即处理。

### 5.5 G2 中段 checkpoint 进展

2026-09-22 03:35（Asia/Shanghai）确认 G2 已形成 `step-256`（02:34:43）、`step-512`（02:57:15）和 `step-768`（03:19:31）。三个 checkpoint 均包含 `decoder_mask.safetensors`、`training_state.pt`、配置和 fingerprint；远端 finite 检查均通过，均为 13 个 tensor、989451 个参数。fingerprint 确认 `head_only=true`、`target_mode=token`、router LR `1e-4`。恢复 run 仍为 `running`，G3 尚未启动；step-1024 前尚无 `train_log.jsonl`／validation summary，错误扫描为空。

### 5.6 G2 完成与 G3 启动

2026-09-22 04:35（Asia/Shanghai）确认 G2 已完成 step-1024；`step-256/512/768/1024` 四个 checkpoint 均 finite（13 个 tensor、989451 个参数），无 CUDA/OOM/NaN/Inf/Traceback。G2 step-1024 最后训练记录为 `loss=0.234630`、`bce=0.193758`、`dice=0.978062`、`mask_mean=0.125300`、`pooled_peak=0.972569`；step-1 为 `mask_mean=0.020167`、`pooled_peak=0.043633`，因此 pooled peak 已明显升高，mask_mean 仅略超预设低位，未出现接近 0.9 的目标退化。64 页 validation 为 CER `0.4772675348`、I/D/S `5285/1364/3796`、reference characters `21885`、exact-page rate `0`，K1/K3/K5 recall `0.579793/0.659191/0.666466`。G3 已启动并形成 `step-256`，checkpoint finite（29 个 tensor、5299228 个参数）；恢复 run 仍为 `running`，等待 G3 后续 checkpoint 与 validation 指标。

### 5.7 G3 中段进展

2026-09-22 05:36（Asia/Shanghai）确认 G3 已形成 `step-256`、`step-512`、`step-768`，对应 checkpoint 均 finite（29 个 tensor、5299228 个参数）；fingerprint 为 `head=vae`、`target_mode=token`、`head_only=true`。恢复 run 仍为 `running`，G3 尚未形成 step-1024 的 train log／validation summary，错误扫描为空。

### 5.8 G3 完成与波形评测启动

2026-09-22 06:37（Asia/Shanghai）确认 G3 已完成 step-1024，恢复 run `status/screen.json=complete`，五卡已释放；四个 G3 checkpoint 均 finite（29 个 tensor、5299228 个参数）。G3 step-1024 最后训练记录为 `loss=0.288307`、`bce=0.210395`、`dice=0.975277`、`mask_mean=0.105160`、`pooled_peak=0.963335`、`kl_raw=0.016403`、`kl_term=0.050000`。64 页 validation 为 CER `0.4642449166`、I/D/S `5920/1700/2540`、reference characters `21885`、exact-page rate `0`，K1/K3/K5 recall `0.685419/0.744063/0.747816`。

训练完成后将主 run 的 G1 四个 checkpoint 复制到恢复 bundle 的 `arms/G1` 作为只读复用（原目录保留），并启动 `evaluate_decoder_mask_waves.sh` 共 12 个 job；输出 `eval_mask_waves_v1`，配置为 G1/G2/G3 × step 256/512/768/1024、4M、`max_new_tokens=1536`、validation64、test 不读取。首波任务已进入模型加载，GPU 资源正常，暂无错误。

### 5.9 波形评测第一波

2026-09-22 07:11（Asia/Shanghai）确认 12 个 job 中首波 5 个已完成并生成 `summary.json`／`predictions.jsonl`：G1@256 CER `0.371990`、G2@256 `0.485767`、G3@256 `0.459630`、G1@512 `0.582317`、G2@512 `0.400183`。均为 64 页、4M、`max_new_tokens=1536`、`test_used_for_selection=false`。第二波任务已启动，GPU 资源正常；日志中未发现真实 CUDA/OOM/NaN/Inf/Traceback 错误。

### 5.10 波形评测完成

12/12 job 均返回 `status=complete`；每行均为 64 页、reference characters `21885`、`exact_page_rate=0`、4M、`max_new_tokens=1536`、`test_used_for_selection=false`。I/D/S 为插入/删除/替换，generation-limit hit 为触顶率。

| arm / step | CER | I/D/S | K1 / K3 / K5 recall | hit | elapsed (s) | tok/s |
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

### 5.11 B0 启动与恢复

波形评测 12/12 完成后，B0 首次尝试因手工启动遗漏系统 CUDA/CUPTI 路径失败，第二次尝试因旧 evaluator 的 `routing_mode=none` 分支默认要求 `lora.safetensors` 失败；两次产物和日志均保留。第三次以 `b0_zero_shot_v2_frozen_base` 启动，fingerprint 明确为 `routing_mode=none`、`head_only=true`、`zero_shot=true`、`training_updates=0`，不注入 LoRA、不加载 mask head；已在 GPU0 完成同一 64 页 validation、4M、`max_new_tokens=1536`，test 不读取。

### 5.12 B0 完成与最终核验

B0 `status=complete`，生成 `summary.json`／`predictions.jsonl`，GPU0 已释放。完整指标：reference characters `21885`、character errors `6282`、I/D/S `2312/1630/2340`、CER `0.2870459219`、exact-page rate `0`、K1/K3/K5 recall `0.645161/0.644995/0.674716`、generation-limit hit rate `0.03125`、elapsed `641.3s`、`42.88 tok/s`、`mask_pages=0`。协议字段：`max_pixels=4000000`、`max_new_tokens=1536`、`routing_mode=none`、`head_only=true`、`zero_shot=true`、`training_updates=0`、`lora_injected=false`、`test_used_for_selection=false`。

最终核验确认：主失败 run 和两次失败 B0 尝试均保留；恢复训练 run `complete`；G1/G2/G3 四步 checkpoint 均 finite；波形 validation `12/12` complete；B0 summary complete；所有评测均使用同一 64 页 validation、4M 和 `max_new_tokens=1536`，未读取 test。`NCCL_P2P_DISABLE=1` 和 `rasterize_polygon` 的 `sign.sum(dim=-1).abs() == K` 契约均保留。

## 6. 待结果

恢复 run 完成后，按主 run 与恢复 run 的 checkpoint 合并前单份权重执行 `evaluate_decoder_mask_waves.sh`（G1/G2/G3、step 256/512/768/1024、4M、`max_new_tokens=1536`），再执行同一 64 页 validation 的 B0 frozen-base zero-shot；test 不读取。随后把每臂完整 fingerprint、mask/loss/validation 指标和 B0 协议字段补回本日志与 `EXPERIMENT_REGISTER.md`。
