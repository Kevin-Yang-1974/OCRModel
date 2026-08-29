# 主线、归档分支与数据划分

更新日期：2026-08-29

## 1. 协作者先看什么

协作者只需要从共享远程 `main` 开始。当前主线是原始 GOT2 whole-page 输入上的单一 PVLD 流程：

```text
P1 layout warm-up
  -> validation-only checkpoint selection
P2 OCR + layout training
  -> validation-only checkpoint selection
P3 domain adaptation
  -> validation-only checkpoint selection
  -> selection-locked test
```

bbox、书写方向和阅读顺序是训练期辅助监督或离线评测真值，不是模型推理输入。P2/P3 必须从同一实验组前一阶段的 validation-selected checkpoint 初始化；test 不参与训练、选点、阈值或后处理。

当前活动入口：

| 作用 | 主线文件 |
|---|---|
| P1/P2/P3 编排 | `tools/training/run_time_constrained_pvld_baseline.sh` |
| 阶段 runner、GPU 准入和 checkpoint 衔接 | `tools/training/run_variable_layout_a100.py` |
| validation checkpoint selection | `tools/evaluation/select_layout_ablation_checkpoint.py` |
| selection-locked test | `tools/evaluation/evaluate_layout_ablation_test.py` |
| 固定 validation manifest | `tools/preprocessing/prepare_time_constrained_validation.py` |
| 结果和训练稳定性汇总 | `tools/evaluation/summarize_time_constrained_pvld.py` |
| GOT2/PVLD 架构 | `src/GOT-OCR-2.0/GOT/model/layout_prompt_decoder.py`、`layout_query.py`、`GOT_ocr_2_0.py` |

`layout_query.py` 中仍存在旧名称，是为了加载既有 checkpoint；主线只使用 `visual_value_layout_routing` 的 PVLD 路径。它不代表 Fixed-Slot VLQA 回到主线。

## 2. 旧方案在哪里

`ocr-shared/archive/legacy-vlqa-chunk-20260829` 保存整理前完整 subtree，包含：

- Fixed-Slot VLQA、VQLCA 及 A0–A5 结构消融；
- oracle-chunk/Chunk training 和旧 MTHv2 C1–C5 流程；
- AncientDoc C0/C1/C4/C5/C6、历史 pilot 和 BSCC M1–M4；
- 外部 SOTA 部署与比较；
- 旧 line-level、双 GOT2、恢复脚本和对应测试/报告。

这些内容只能用于复现、结果溯源或诊断，不能被当前主线的训练器、selection 或 test 读取。需要查历史实现时，单独 clone 归档分支，不把旧脚本复制回主线。

## 3. 数据集层级

### 3.1 S3/S4 正式合成页面

正式数据根目录由机器本地 `GOT_LAYOUT_DATA` 指定，包含独立的 `train/validation/test` manifest。先按来源组、书手/版本/馆藏或合成源页面划分，再生成同一 split 内的模板和退化版本；禁止把同一源页、近重复页或同一内容跨 split。主线训练使用 whole-page image，页面区域标注仅作为布局监督。

时间受限策略验证从正式 `validation/manifest.jsonl` 锁定 400 页：S3 `s3-ancient-hard` 200 页、S4 `s4-mixed` 200 页；每个 tier 内按 region-count bucket 与复杂度 tertile 分层，再以固定 seed 稳定 round-robin 取样。锁定文件记录来源 manifest SHA-256、固定 manifest SHA-256、分层计数和 `test_used_for_selection=false`。P1、P2、P3 共用这一份 manifest。

### 3.2 MTHv2 replay 与 Real-OOD test

主线只从 MTHv2 的官方 `train/manifest.jsonl` 读取 P1 replay 和 P3 训练数据；MTHv2 `validation` 不进入训练，MTHv2 `test` 只在 P3 validation selection 完成后作为 selection-locked test 读取。MTHv2 页面保持 whole-page 粒度，`label_textline` 转换得到的是有序区域候选，不应称为严格 column 真值。

MTHv2 的官方三份 split 必须在数据准备阶段固定，按源页面继承 split；不能先生成 oracle chunks 再随机划分。Chunk 数据若需复现，只能使用归档分支的独立协议，不能与主线 whole-page 结果直接比较。

### 3.3 少样本报告

领域级少样本限制独立来源组可用的标注页数或比例；稀有符号级 K-shot 限制每个符号的独立来源实例。每个报告必须给出 split、来源组数、页面数、区域数、manifest SHA-256 以及是否存在近重复审计。页面增强数量不能伪装成新的独立样本。

## 4. 结果解释

主线最终 summary 必须同时给出 OCR CER、页面精确匹配、区域 P/R/F1、bbox IoU、方向/顺序指标、EOS/count 截断、训练稳定性、可训练参数量和显存/吞吐。只有在同一 test manifest 上提供此前冻结策略的 selection-locked summary 时，才计算新旧策略差值；不同输入粒度、split 或预算的结果不得直接比较。

当前主线尚未产生新的正式性能结论时，文档应写“待验证”，不能把 smoke、pilot、VLQA 或 Chunk 结果写成 PVLD 主线结果。
