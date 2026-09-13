# Free-generation OCR loss experiment

这是一个独立的短程机制实验，不修改旧的 Geometry/Baseline run，也不读取
MTHv2。实验使用敦煌/地方志 `q32` 兼容数据集的既有划分：240 页 train、80
页 validation、59 页 held-out test；本阶段只训练和 validation，暂不执行 test。

## 目标

现有训练的 teacher-forcing OCR loss 持续下降，但自由生成 validation CER 变差，
且生成触顶率从早期 checkpoint 的 0 持续上升。这里每个训练页执行：

1. 以图像和 prompt-only 输入做 `no_grad` greedy free rollout；
2. 把生成得到的 token 前缀作为 detached history，做一次带梯度的 free-prefix forward；
3. 在这一次 forward 的生成前缀上计算 OCR CE，同时从同一次 forward 的 layout side-channel
   计算 layout loss；
4. 不额外计算 teacher-forcing OCR CE，也不把 rollout token 放入梯度图。

目标函数为：

`L = 0.05 * L_free_generation + 0.2 * L_layout`

其中 `0.05` 是根据 8-step calibration smoke 中 teacher-forcing OCR loss 约 `1.24`、
free-generation OCR loss 约 `24.74` 估计的固定常数，用于让两者大致处于同一数量级；
训练期间不按页面或 step 动态调整。生成 token 本身不在梯度图中，梯度来自唯一的
free-prefix forward 的 logits。

## 参考配置

- mode：`geometry`，decoder LoRA；
- steps：600；checkpoint：200、400、600；
- adapter LR：`1.25e-5`；decoder LoRA LR：`2.5e-6`；
- warmup：216；`max_grad_norm=1`；`auxiliary_weight=0.2`；
- seed：42；queries：32；plain generation；训练 free rollout 上限 512 tokens；validation
  仍使用 1536-token 评测上限；
- 先执行 8-step smoke，smoke 通过后才执行正式训练；
- validation 只在三个 checkpoint 上运行，按 validation CER 选点；test 由后续明确指令触发。

该实验用于判断“训练目标暴露偏差”这一机制是否得到改善，不把结果表述为
严格复现历史参数优选结果。
