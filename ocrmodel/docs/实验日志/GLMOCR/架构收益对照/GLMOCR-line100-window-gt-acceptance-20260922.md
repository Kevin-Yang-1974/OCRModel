# line100-window GT 验收：2026-09-22

## Run 与产物

- run ID：`glmocr_line100_window_gt_accept_20260922_145532`。
- 日期：2026-09-22，Asia/Shanghai；分支 `glm-ocr-layout-mask-routing`。
- commit：`e1667b2f52f8a7878bebb76bf7f6b6b1f5fb81a3` 加当前未提交代码；实际执行快照以 `source_fingerprint.json` 为准。
- 快照压缩包 SHA256：`fee74c7766e95f1eeea63bd92be5ba5bcb36155881405779fdbc7c650a3a5cad`，102 个源文件。
- 主机：`a100-yky`；物理 GPU 允许集合 `[0]`，启动阈值严格 `<50%`，不等待、不抢占。
- 远端根目录：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/acceptance_runs/glmocr_line100_window_gt_accept_20260922_145532`。
- tmux session：同 run ID；Slurm job/array ID：不适用，单卡 tmux。
- 入口：独立快照内 `tools/evaluation/run_window_mask_acceptance_a100.sh` → `tools/evaluation/evaluate_window_mask_routing.py --mode gt`。
- 产物：`pipeline.log`、`status/run.json`、`results/protocol.json`、`results/validation_predictions.jsonl`、`results/summary.json`。

## 数据与评价协议

- MTHv2 q32 sparse24 validation，完整 149 页；manifest：`/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1_q32_sparse24/validation/manifest.char.jsonl`。
- SHA256：`36ec845875e1ea18a48d4a523b6c5a3f007b46e2a07de0cafd2c26139929a348`，本轮远端核对一致。
- 沿用已有 sparse24 划分，本轮不重切；隔离单元/划分生成证据需引用父协议，不新增独立隔离保证。
- 本轮读取 validation 图像、page_text、字符框与 line_index；train 未读取，test 未读取；seed=42。
- `test_manifest_read=false`，`test_used_for_selection=false`；GT oracle 仅验收机制，不参与 checkpoint selection。
- 所有页聚合 I/D/S 与 reference characters 后计算 micro CER，不对 shard/page CER 做无权平均。
- 验收：GT-window 完整 validation CER 严格 `<0.13`；输出 `acceptance.eligible/passed`。

## 模型与配置

- 官方模型 revision：`ca5d8b3e287e52589e37c28385d9655ee4372f9d`。
- decoder LoRA 起点：`/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/training_runs/glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000`，rank=8、alpha=8、dropout=0；不训练，加载文件指纹写入 protocol。
- 小 geometry/layout 分支移除；GT mask 直接来自既有 3–5 字窗口构造，hard raster、synced 指针、bias=1.0、所有 decoder 层、prefill 不加偏置。
- fast processor、max_pixels=4,000,000、max_new_tokens=1536、greedy、KV cache、BF16 骨干、确定性 math-SDPA。
- 新 head 为 FP32，GT 验收绕过预测，不使用随机预测 mask；无新增 checkpoint。
- 训练总步数=0；adapter/head LR=0（未优化）、decoder LR=0（冻结）；warmup/schedule 不适用；batch=1、梯度累积不适用、GPU=1、DDP=否、effective inference batch=1。
- trainable range：无优化；auxiliary_weight/layout loss/dice/stop loss 权重均不适用；gate 不存在。
- 单臂验收，没有同期训练对照；历史 `line100` CER 0.124694 为整行 GT 结果，窗口变化与移除小分支均按用户要求，不能把该历史数当作本 run 成绩。

## 状态与结果

初始登记：准备启动。validation selection step：不适用，固定 backbone checkpoint-3000；完整 CER/I/D/S、EOS/触顶/循环、finite 和错误检查均待运行完成后追加。locked test：未读取。无训练 loss 曲线。任何失败与产物均保留，修正使用新 run ID。


## 启动确认

远端 preflight 通过：149页指纹及line_index存在；Torch 2.8.0+cu128，Transformers 5.3.0。GPU0 admission utilization=0%，tmux存在，status=running；510个模型tensor已加载。heartbeat `line100-window-gt` 已绑定当前任务，初始每5分钟，连续两次健康后改为每30分钟。尚无完整CER结论。


## 用户要求切换五卡（2026-09-22）

旧 run `glmocr_line100_window_gt_accept_20260922_145532` 已按用户要求 TERM 中止，退出码143，15页输出保留，不能视为完整验收。`status/terminated.json`保留原退出状态，`status/run.json`标为stopped_by_user；无完整CER、无选点、test未读取。

新 run：`glmocr_line100_window_gt_accept_5gpu_20260922_150507`。远端根目录：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/acceptance_runs/glmocr_line100_window_gt_accept_5gpu_20260922_150507`；tmux同名；Slurm不适用。分支仍glm-ocr-layout-mask-routing，commit以即将提交的源码指纹为准。

数据、checkpoint、mask/window、synced指针、seed42、4M、1536、BF16/math-SDPA与前述协议一致；唯一执行变更为GPU0–4五个独立worker，按manifest轮转分片30/30/30/30/29页，每卡batch1、非DDP、不训练、无梯度累积/学习率/优化器。总页数和每页生成预算不变，旧15页不复用。test_manifest_read=false，test_used_for_selection=false。

入口仍run_window_mask_acceptance_a100.sh，worker结果在shards/0..4与对应.log；merge_window_mask_shards.py校验覆盖和协议并按完整I/D/S聚合，最终写results/summary.json。validation/locked-test/EOS/触顶/循环/finite/错误检查：完整成绩待运行，locked test未读取。所有旧、新产物保留。


**五卡源码提交与启动**：执行commit `21afa4ac8afa4537cc22bdc7e33bf590ccded2a9`，已push到ocr-shared/glm-ocr-layout-mask-routing；使用git archive导出纯提交代码，未混入剩余工作区修改。压缩包SHA256 `819450ecebdeae89a51a4e3784f457784ca037727d7152bf2c3cabfe3b4c4469` 已在A100核验。五卡tmux已派发，沿用同一heartbeat `line100-window-gt` 监控新run，初始5分钟、两次健康后30分钟。待完成后追加完整结果。

## 最终结果：五卡 GT-window 验收完成

- 远端 run：`glmocr_line100_window_gt_accept_5gpu_20260922_150507`；commit `21afa4ac8afa4537cc22bdc7e33bf590ccded2a9`；tmux 已正常退出，五个 worker 均以 complete 结束。
- 五片页数：`30/30/30/30/29`，合计 `149`；汇总脚本确认页 ID 无重复、无遗漏且恢复为 manifest 顺序。五片 manifest SHA256 均为 `36ec845875e1ea18a48d4a523b6c5a3f007b46e2a07de0cafd2c26139929a348`。
- 配置核验：GT 3–5 字 hard window、synced 指针、bias `1.0`、所有 decoder 层、prefill 不注入、fast processor、`max_pixels=4000000`、`max_new_tokens=1536`、BF16、math-SDPA、小版面分支移除；test 未读取。
- 完整 micro 指标：149 页、reference characters `41654`、编辑错误 `5704`，I/D/S=`948/769/3987`，CER=`0.13693762903922793`，exact-page rate=`0`，generation tokens=`50321`。
- 生成终止：EOS `149/149`，触顶 `0/149`；按 `repetition_diagnostics`（recent window 96、cycle length 8–32、3 repeats）检测循环页 `0/149`、循环率 `0.0`、最大重复次数 `1`。
- 结果 JSON 共 11 个（总汇总、5 个 shard summary、5 个 shard protocol）均无非有限数值；日志中没有独立词边界意义上的 NaN/Inf/CUDA OOM/Traceback/Exception/failed/ERROR。
- 验收：`acceptance.eligible=true`，阈值严格 `<0.13`，`acceptance.passed=false`。相对历史整行 `line100` 的 `0.124694`，当前 3–5 字窗口 GT 结果高 `0.0122436290`；因此本次验收未通过。该结果是完整协议结果，不读取 test、不选点、不调参。
- 本地归档：`D:/yangky/glm-ocr-assets/line100-window-acceptance/glmocr_line100_window_gt_accept_5gpu_20260922_150507/`，包含 `results/summary.json`、`results/validation_predictions.jsonl` 及每片 `summary.json`/`protocol.json`。
