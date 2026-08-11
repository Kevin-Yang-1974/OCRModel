# A100 布局训练编排器

`run_layout_a100.py` 是 GOT2＋VLQA 合成预训练的服务器侧编排入口。它不生成页面，而是读取本机已经生成并整体上传的 `manifest.jsonl`、`images/` 和 `html/`。

默认 `--mode smoke` 依次执行：

1. 校验 GOT Python、PyTorch、Transformers、DeepSpeed 版本及活动源码位置；
2. 检查物理 GPU 0，忙碌时以退出码 75 结束；
3. 复审 manifest、图片哈希、DOM、字体证据和跨 split 泄漏；
4. 在 GPU 0 上以放大视觉输入运行独立 VLQA 小张量 forward/backward，并拒绝异常初始 logit 尺度；
5. P1 使用原始 GOT2 权重，确认 checkpoint 没有 VLQA 键后显式完整初始化适配器，再执行 1 条记录、1 个 optimizer step；
6. 验证 P1 的 VLQA 权重、有限性和严格为 0 的 `residual_gate`；
7. P2 必须从 P1 模型目录加载完整 VLQA 状态，部分或不兼容状态会在训练前被拒绝，再执行 1 条记录、1 个 optimizer step；
8. 验证 P2 的 VLQA、`mm_projector_vary`、训练指标和完整模型重载。

完整 Trainer 输出只写入 run 目录，终端只显示紧凑状态。参数化同步与 A100 启动命令见 `../../docs/SYNC_AND_RUN.md`。

A100 run `layout_pilot_20260811_023528` 已打通上述工程链路，但其 2 页、每阶段 500 epoch 的配置没有 validation，P1 也未拟合到接近零，不能作为有效 pilot 或性能结果。

A100 run `layout_overfit_20260811_110317` 已执行首次受控诊断并返回 fail。第 1 步 object/direction logits 已约为 1680，bbox 同时饱和到约 0/1，说明主要异常在优化前的 VLQA 初始化或前向尺度，而不是训练后期逐渐发散。当前修复在加载原始 GOT2 后显式完整初始化 VLQA，严格区分无布局权重、完整布局权重和部分布局权重，并在辅助头前增加最终 LayerNorm 与小尺度 bbox 输出层初始化。

A100 run `layout_overfit_20260811_113817` 已确认初始化修复有效：首步 object/direction/bbox raw logits 分别为 0.1592、0.3984 和 0.0010。200 步后 object 与 direction 已通过阈值；bbox L1 从 0.9761 降至尾段均值 0.1159，bbox mean IoU 从 0.0726 升至 0.3496，说明 bbox 正在学习但现有步数不足以完成严格实现检查。

`layout_overfit_20260812_002747` 已完成固定 P1、2 条记录、1000 optimizer steps 的实现诊断并返回 `overfit_assessment.status=pass`。末 20 步均值为 object loss `0.00201755`、bbox L1 `0.00531464`、bbox GIoU `0.05508578`、direction loss `0.00154159`、object/direction accuracy `1.0`、bbox mean IoU `0.94497279`。`status=pass` 只代表两样本实现检查通过，不代表泛化性能。

当前 validation 入口为 `--mode validate`。它只执行环境/manifest/component preflight 和 prompt-only 整页评测，不启动 DeepSpeed 训练；必须通过 `--source-model` 指向包含 VLQA 权重的 checkpoint，并可用 `--tokenizer-model` 指向原始 GOT tokenizer。评测输出统一写入 run 目录，布局 metadata 只用于离线指标。

首轮 run `layout_validate_20260812_012924` 在模型加载前被 evaluator 的 tokenizer 文件名白名单错误拦截。当前 loader 已改为直接让 Transformers 离线验证本地 GOT/Qwen tokenizer；修复后的 `layout_validate_20260812_014816` 已在两页 `train` split 上通过 prompt-only checkpoint 重载和指标链路。该结果只属于同页 P1 overfit 诊断，正式 held-out validation 仍待执行。

`pilot` 没有默认训练步数，必须同时提供 `--allow-unvalidated-pilot`、`--p1-max-steps` 和 `--p2-max-steps`。在真实 validation 和统一实验协议锁定前，不应启动 pilot。
