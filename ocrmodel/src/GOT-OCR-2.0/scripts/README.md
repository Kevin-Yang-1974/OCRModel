# GOT2/PVLD 训练脚本

本目录包含共享主线使用的 GOT2 whole-page 数据、训练和评估实现。服务器外部路径由仓库根目录的 `config/paths.env` 提供；不得在源码树内建立数据、权重或 checkpoint 软链接。

## 当前模型路径

当前模型输入为 `whole_page_image + ocr_prompt`。global layout prompts 从整页视觉特征汇聚布局证据，causal PVLD 生成变量长度 REGION/EOS 序列。bbox、书写方向和阅读顺序只用于训练辅助监督或评测真值，不作为模型输入。

关键文件：

- `layout_page_dataset.py`：whole-page manifest dataset 与布局监督 collator；
- `train_GOT_layout.py`：P1/P2/P3 的训练实现；
- `evaluate_GOT_layout.py`：prompt-only whole-page 生成与统一页面指标；
- `layout_validation_metrics.py`：OCR、区域、bbox、方向、顺序和停止指标；
- `../GOT/model/layout_prompt_decoder.py`：causal PVLD decoder；
- `../GOT/model/GOT_ocr_2_0.py`：PVLD 接入和 `visual_value_layout_routing`。

PVLD 使用 `layout_evidence=A` 的 cross-attention 和视觉 Value 路由；区域数量由生成的 REGION 记录和 EOS 决定。`max_layout_records` 与 `max_layout_tokens` 是工程上限，不代表 Fixed-Slot query 数。

## 运行边界

`../../../tools/training/run_variable_layout_a100.py` 负责当前阶段 runner、GPU 准入和 checkpoint 衔接；`../../../tools/training/run_pvld_causal_cuda_smoke.sh` 是有界 CUDA forward/backward 检查。完整 P1 → P2 → P3 编排、validation selection 和 selection-locked test 见 `../../../docs/SYNC_AND_RUN.md`。

Fixed-Slot VLQA/VQLCA、Chunk、AncientDoc、BSCC、SOTA、line-level 和双 GOT2 训练脚本已移至 `archive/legacy-vlqa-chunk-20260829`。模型代码中保留的历史名称仅用于加载旧 checkpoint，不代表它们属于当前主线。
