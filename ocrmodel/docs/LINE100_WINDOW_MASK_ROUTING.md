# line100 与第一层窗口 mask 融合

2026-09-22。状态：本地实现；正式模型的 GT CER 验收尚未运行。

## 最终需求与来源

按本次最后一次需求，以 `line100` 为锚点，不使用 `win50_g3`。
`line100` 在 sparse24 validation 149 页的历史 CER 为 **0.124694**，来源为
`LAYOUT_ORACLE_LINE_RESULT.md` 和 `LAYOUT_LINE_DETECTOR_AND_PREDMAP_RESULT.md`。
这是整行 GT 框配合 synced 指针的结果，不是新窗口的成绩。

新空间目标复用 `mask_targets.build_mask_targets` 已实现的 `window`：每个 token
对应其所在行中的 3–5 字窗口，行尾向左补足，短行允许不足 3 字，使用字符框角点的凸包，
在合并后的视觉 token 网格上按中心点生成 hard mask。通常单字 token 对应 3 字；
窗口大小由 token 跨度限制到 3–5，并不是随机选择宽度。

## 配置

| 项目 | 新主路径 |
| --- | --- |
| 基座 | `ca5d8b3e287e52589e37c28385d9655ee4372f9d` |
| decoder LoRA 来源 | `glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000/decoder_lora.safetensors` |
| LoRA | rank=8、alpha=8、dropout=0，加载后冻结 |
| 原 geometry/layout 小分支 | 移除；新路径不安装 `LayoutAwarePatchMerger` |
| 图像与生成 | fast processor、4,000,000 pixels、1536 new tokens、greedy、KV cache、seed 42 |
| attention | 确定性 math-SDPA，BF16 骨干 |
| 注入 | 所有 decoder 层，hard mask × 1.0；prefill 不施加偏置 |
| GT 验收指针 | 原 `AttentionRouting.observe_inputs` 的 synced 指针，LOOKAHEAD=16 |
| 模型预测 | 第 0 层输出 → 原 `DecoderMaskRouter` MLP → 下一解码步窗口 |
| mask 分支输入 | 第一层 decoder hidden state、合并后的全页视觉特征、网格坐标、上一预测 mask |
| 正式推理输入 | 整页图像与 prompt，不传 page_text、框、真值 mask 或真值顺序 |

新实现复用 `AttentionRouting.hook`，所以因果/padding mask 合并、prefill 跳过、每步缓存
一次偏置并供全部层共享的行为与锚点一致。第一层输出产生下一步 mask，不能作用于
已经完成的当前第一层。第一步生成来自无偏置 prefill；第一个 decode 使用 prefill
最后一个位置产生的预测窗口。GT 窗口由 synced 指针选择当前待识别位置，没有预测延迟。

`line100` 不是 attention tracker 臂，因此本版本不携带 win50_g3 的门控 3.0、
第 8 层探针或下一行份额 0.5。模型预测时也不保留 synced 真值指针。

## 实现入口

- `src/layout_ocr/window_mask_routing.py`：新运行时，第一层输出分支、预测/GT provider、跨页清理。
- `tools/evaluation/evaluate_window_mask_routing.py`：三种模式：
  - `legacy-line`：原 geometry 分支、GT 整行、synced；用于复核 0.124694。
  - `gt`（默认）：移除小分支、GT 3–5 字窗口、synced；正式 GT 验收。
  - `predicted`：移除小分支、训练后的预测窗口；图像与 prompt 推理。
- `tools/training/train_window_mask_routing.py`：只训练第一层 mask head。

评测必填 `--model-path`、`--backbone-checkpoint`、`--validation-manifest`、`--output-dir`。
预测模式另需 `--mask-checkpoint`。训练必填对应基座/骨干 checkpoint、`--train-manifest`
和 `--output-dir`。所有模型、数据和产物路径必须位于源码树外。

评测的 `--legacy-layout-control` 只用于 GT 对照：保留原小分支，隔离“小分支移除”和
“整行换窗口”两项变化；它不能计作新框架验收。不得为了过阈值把它冒充默认新路径。
checkpoint 的 LoRA SHA256、manifest SHA256、生成协议与运行模式写入 `protocol.json`。
新 head checkpoint 带独立 `window_routing_profile.json` 和训练骨干指纹；评测拒绝把
旧同一步/后半层 mask checkpoint 或其他骨干的 head 当成此版本。

## 新分支训练

旧锚点没有该分支的训练配置。新增 head 的默认配置是待验证设置：FP32 head、学习率
1e-4、weight decay=0.01、16-page-step warmup、cosine 至 0.1 倍、128 page steps、
单卡 batch=1、每 32 token 截断递归梯度、每页累积后更新一次、梯度裁剪 1.0。
仅 mask balanced BCE 与 stop BCE，权重均 1；dice、layout auxiliary 与 decoder 学习率均 0。
骨干保持 eval，head 可训练，无 GT mask 反馈、无 query noise 或 feedback noise。

训练使用逐 token teacher forcing 和 KV cache，保持预测 mask 的真实注入时序。
第 q 个位置的 head 输出监督到 `label[q+2]` 的窗口，因为它供下一位置 q+1 使用。
GT 只用于 loss，施加的偏置始终来自预测。预投影视觉特征在截断边界重新建立 head
投影计算图，避免视觉投影只从每页第一段收到梯度。训练不读取 validation 或 test；
后续必须用独立 predicted validation 选点，GT oracle 不用于 checkpoint selection。

## 验收

**完整 149 页 GT-window validation CER 严格小于 0.13。**
固定 manifest SHA256：`36ec845875e1ea18a48d4a523b6c5a3f007b46e2a07de0cafd2c26139929a348`。
默认 GT 模式、完整 manifest、无 `--pages` 子集、无 legacy layout control 才有验收资格。
`summary.json` 的 `acceptance.passed` 为 true/false；其他模式或子集为 null。
测试集未读取，`test_manifest_read=false`，`test_used_for_selection=false`。

历史整行的 0.124694 不能保证窗口仍低于 0.13。已有另一路窗口 GT 日志报告
0.1369376290，但其保留小分支等实现细节不同；它同样不能证明本次融合过线。
若新 GT 不过线，保留失败结果，先比较 legacy-line 与 legacy-layout-control，
不自行更改 bias、指针、分辨率或评价子集。

## 本地验证与运行边界

测试覆盖原 hook 对同一 hard support 的逐位偏置一致性、prefill 不变、padding 保留、
第一层到下一步的 mask 时序、synced 指针跳过插入、参考结束后清空、跨页状态隔离、
checkpoint round-trip 和验收阈值。随机初始化微型 GLM-OCR 测试真实 SDPA 生成、
KV cache 与跨两段 TBPTT 反向传播，确认梯度只进入 mask head。

本轮验证结果：上述新测试、原 attention-routing / attention-tracking 与 mask-target
测试合计 **80 项通过**；五个新增 Python 文件 Ruff 检查通过，两个 CLI `--help`
正常。环境为源码树外的 `D:/yangky/glm-ocr-assets/venvs/mask-tracking-cpu`，
Python 3.11、CPU PyTorch 2.14、Transformers 5.3 系列。这些是实现验证，不是 CER 实验。

本轮没有同步代码、启动 A100/BSCC、执行真实 149 页 CER 或读取 test，未产生新的
正式实验 run。以后启动任何 GT 验收/训练/诊断均须在 `EXPERIMENT_REGISTER.md` 登记
全部配置与协议字段，并按 AGENTS.md 设置当前任务 heartbeat 和规定的 GPU admission。
旧脚本保留用于已有 run 复现；此需求的新路径使用上述独立入口。

## 2026-09-22 五卡 validation

用户要求改为物理 GPU 0–4 并行：每个 worker 加载相同模型，按原 manifest 的
`records[shard_index::5]` 分页，页数 30/30/30/30/29。每卡仍 batch=1，非 DDP。
`--shard-count` / `--shard-index` 不改变模型、mask、指针或生成参数。
`merge_window_mask_shards.py` 验证所有 worker complete、协议一致、149 页无重无漏，
按原 manifest 顺序恢复预测并从完整 reference/prediction 重算 micro CER。
单 shard 不具备验收资格；最终 `results/summary.json` 才给出是否小于 0.13。
`run_window_mask_acceptance_a100.sh` 现默认使用允许集合 0–4，并要求所有卡瞬时利用率
严格小于50%后同时启动。原单卡 run 被用户要求中止，15页产物保留、不混入新run。
