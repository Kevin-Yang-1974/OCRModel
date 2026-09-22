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
