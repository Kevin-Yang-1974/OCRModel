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

当前下一步仍使用一次 `--mode overfit` 复测修复版。该模式固定只运行 P1、2 条记录和 200 optimizer steps；布局损失以 FP32 计算，Trainer 记录分项损失、object/direction accuracy、bbox mean IoU、query/raw-logit/bbox 范围、残差门和 query 梯度，结束时直接返回初始化方式、首步观测、末 20 步均值和 `overfit_assessment`。`status=pass` 只代表两样本实现检查通过，不代表泛化性能。

`pilot` 没有默认训练步数，必须同时提供 `--allow-unvalidated-pilot`、`--p1-max-steps` 和 `--p2-max-steps`。在 `overfit` 通过并实现 validation loader 前，不应再次启动 pilot。
