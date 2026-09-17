# GLM-OCR aligned_recovery_v1 执行协议

从固定基础 revision 初始化，不加载历史 checkpoint；一组 seed42、五卡同步 DDP、global batch 5、1024 steps。整页输入，512 queries，geometry/full/Hungarian。adapter/decoder LoRA LR 为 5e-5/1e-6，LoRA 8/8/0，warmup 102，cosine horizon 1024，最低 LR 比例 **0.5**。最低 LR 分别为 2.5e-5、5e-7。

新模式每四步采集真实 greedy rollout，最多 768 token；完整扫描首次三次连续重复，周期为 8–32 token。生成不带 GT，临时 eval 后恢复各模块原模式、use_cache 和 RoPE 状态。

仅为监督对齐将重复压缩为一次；原始错误前缀仍作为恢复输入。单位编辑代价对齐 GT 前缀，未消费 GT 后缀免费。接受条件：最优终点唯一，编辑代价除以 max(压缩前缀长度, 匹配 GT 长度, 1) 不超过 0.2，尾部连续匹配至少 8 token，相比原前缀改善至少 ceil(周期长度/2)。编辑路径同分时优先对角，再删除生成插入，最后删除 GT。歧义样本仅做原始 OCR。

尚未读完时监督最多 16 个正确恢复 token；读完且未删除 GT 内容时监督原始标签协议中的 EOS。恢复段短于 16 且抵达无删除的 GT 末尾时包含 EOS。所有错误前缀标签为 -100。第二次 forward 重新构造序列字段，保持图像不变；超模型上下文时跳过该恢复监督，不截断输入。

损失为 `L_OCR + 0.2 L_layout + min(step/102,1) * (0.01 L_UL + 0.05 L_recover + 0.05 L_end)`。每项按全局有效 token 数归一化，正确补偿 DDP 梯度平均；UL 仅在恢复边界作用于不同于正确 token 的循环候选。保留旧模式用于复现，新入口不混用旧恢复目标。

入口：`tools/training/run_glmocr_aligned_recovery_v1.sh`。流程为 CPU 合约检查、两进程 CPU DDP 检查、五卡 2-step smoke、正式训练与 validation、selection-locked test。smoke 独立 run、每步采样、5 页 validation 验证 checkpoint 重载；正式阶段重新从基础权重开始。

正式 train/validation/test 为 2159/240/800。初始化、step512、step1024 分别完整 validation；仅在后两个点中按 CER、较早 step 选点。验收要求 CER、循环率、触顶率均改善，删除率增加不超过 1 个百分点。验收结论不修改选点、不触发自动调参。选点后只运行一次 test，并披露历史 test 暴露。推理始终为 image+prompt、plain greedy、768 token，无强制 EOS 或循环处理器。

初始 run ID：`glmocr_aligned_recovery_v1_base1024_260912`，smoke 后缀 `_smoke2`。已存在 run 不覆盖。阶段文件为 group 的 `pipeline_status.json` 及 seed42 的 `phase.json`，完成验收写入 `aligned_recovery_acceptance.json`。启动仅准入 GPU 0–4 且利用率均低于 50%；不查询集合外 GPU。

实现来源：真实轨迹上的 UL 借鉴 Welleck 等的 *Neural Text Generation with Unlikelihood Training*（ICLR 2020，https://arxiv.org/abs/1908.04319）。OCR 前缀对齐、恢复与结束监督是本项目待验证修改；代码检查和 smoke 不代表循环抑制效果。

## 接受门实现修正（2026-09-12）

首轮实现中接受率恒为 0，诊断为两处实现缺陷（阈值数值未变）：

1. 压缩前缀的最优终点在循环内容不等于 GT 时，会与 `最早终点 + 周期长度` 必然并列（删除多余循环 vs 将其替换到 GT 上代价相同），`unique` 被误判为歧义。修正：所有最优终点落在最早终点的同一周期长度内即视为非歧义；真正的 GT 前缀歧义仍然拒绝。
2. 回溯对尾部多余生成 token 优先走替换，导致 `tail_matches` 误记为 0。修正：预测端删除不再重置尾部匹配计数，尾部连续匹配按真实对齐路径统计。

保持固定：`error≤0.2`、`tail_matches≥8`、`improvement≥ceil(周期/2)`，以及全部损失权重、`ramp_steps`，不做搜索。该修正只让"模型已读对一段 GT 前缀后再发生循环"的样本被接受，从而产生正向 `L_recover`/`L_end`；早期乱循环、循环内容错位仍被拒绝。检出即计算 `L_UL`（负样本为循环首个 token，边界为真实前缀末位，`gold_next` 排除合法重复）为独立改动，一并记录。

诊断证据（合成轨迹，`gold=list(range(1000,1100))`、`wrong=list(range(10,18))`）：`gold[:40]+wrong*3` 修正后接受，`endpoint=40`、`error=0.167`、`tail=40`、`suffix=gold[40:56]`；`gold[:20]+wrong*3` 仍以 `error=0.286` 拒绝；`wrong*4` 以 `error=1.0` 拒绝。
