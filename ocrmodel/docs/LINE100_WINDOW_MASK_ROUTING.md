# line100 与第一层窗口 mask 融合

2026-09-22。状态：**GT 验收已运行且未通过**（CER `0.136938`，阈值 `<0.13`）；
失败已归因到空间目标形态，bias 优选已完成（窗口天花板 `0.1319`）；
下一步是空间目标形态阶梯，待重投。

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

## 验收结果与失败归因（2026-09-22，已完成）

**验收未通过。** 五卡合并完整 149 页的 micro CER = `0.13693762903922793`，`acceptance.eligible=true`、
`passed=false`（阈值严格 `<0.13`）。I/D/S = `948/769/3987`，reference characters `41654`，
`generation_limit_hits=0/149`，EOS `149/149`，循环页 `0`。`max_new_tokens=1536` 与历史整行同口径，
差值不是生成预算造成的。产物见 `D:/yangky/glm-ocr-assets/line100-window-acceptance/`。

按本节上述处置，先跑了 `legacy-line` 与 `gt + --legacy-layout-control` 两个诊断臂
（详见 `实验日志/GLMOCR/架构收益对照/GLMOCR-line100-window-attribution-20260922.md`）：

| 臂 | 小分支 | 空间目标 | CER | 每步命中 token |
| --- | --- | --- | ---: | ---: |
| `legacy-line` | 装 | GT 整行框 | **0.12469390694771211** | 77.62 |
| `gt`（验收） | 不装 | GT 3–5 字窗口 | **0.13693762903922793** | 8.38 |
| `gt` + `--legacy-layout-control` | 装 | GT 3–5 字窗口 | **0.13693762903922793** | 8.38 |

1. **装置可比**：`legacy-line` 复现历史整行值到小数点后 11 位。
2. **小分支移除是精确零效应**：控制臂与验收臂的 `validation_predictions.jsonl` **SHA256 相同**
   （`846acbede98c2c96c00e40408c529bb7c9334f669d8d622eaf24289d5baff37a`），
   而控制臂 protocol 记 `layout_branch_present=true`。**自由生成路径上，小分支装上与否输出逐字节一致。**
3. **差距 100% 来自空间目标**：`−0.012244`，逐页配对 bootstrap 按页 CI `[−0.022881, −0.003519]`、
   按卷分组 CI `[−0.043107, −0.003280]`，**两档均显著**。
   失败签名与 `line100` 收益签名方向相反：插入 `738→948`、删除 `569→769`、替换几乎不动。

## bias 优选：窗口的天花板（2026-09-22，已完成）

窗口的 `B=1.0` 继承自整行臂、从未针对窗口扫过，故补扫 `1.0/1.5/2.0/2.5/3.0`
（详见 `实验日志/GLMOCR/架构收益对照/GLMOCR-line100-window-beta-sweep-20260922.md`）：

| beta | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CER | 0.136938 | **0.131944** | 0.174341 | 0.151438 | 0.214481 |
| 插入 | 948 | 826 | 2575 | 1651 | 3753 |

- 峰在 `[1.0, 2.0)`、靠近 `1.5`；`B≥2.0` 崩坏，崩坏量在插入、替换不动。
- `B=1.5` 相对 `B=1.0` 按页显著（CI `[−0.010909, −0.000494]`）但**按卷分组不显著**
  （CI `[−0.005977, +0.003045]`），逐页为「更好 39 / 更差 35 / 打平 75」、改善集中在少数页。
  故 `B=1.0` 不是最优但差距不足以改记录配置。
- **窗口的天花板是 `0.1319`。** 参数优选到此为止：即使取峰，仍高于历史整行的 `0.1247`，
  **剩下的 0.007 靠调 `B` 补不上**，只能在空间目标形态上解决。

## 下一步：空间目标形态阶梯（待执行）

剂量测量（`tools/measure_mask_dose.py`，零 GPU，已通过自检：window 测得 `7.386` 对聚合记录值 `7.502`）
把可选的形态放在同一条覆盖轴上：

| 目标形态 | 每步命中 token | 相对 window |
| --- | ---: | ---: |
| `token` | 3.56 | 0.48× |
| `window`（当前） | 7.39 | 1.00× |
| `anchored`（窗口∪本行剩余） | 28.33 | 3.83× |
| `line`（字符框凸包） | 56.48 | 7.64× |
| 整行（臂的 `regions` 框） | 69.27 | 9.37× |

`anchored` 落在 window 与 line 之间，故阶梯单调、可作受控的单变量步进；它同时是
`LAYOUT_ORACLE_LINE_RESULT.md` §7.1 要求的「固定框形状、变覆盖」那类实验的第一次落地。

**三臂（`window` 对照 / `anchored` / `linehull`）已实现并有测试（commit `cc5f8c4`），但首次派发因
每卡 3 个 worker 触发 OOM 而失败（见 `RUN.md` 的显存并发上限）。** 重投时必须改为
**每卡一个 worker、149 页全量串行**（约 1 小时），或分两次各跑一两臂。

`anchored` 的两种可能结果都有信息量：若明显向 `0.125` 靠拢，说明**覆盖量**是决定性的；
若剂量已达 window 的 3.8 倍而收益有限，则整行的优势来自**覆盖模式**而非覆盖量，
`anchored` 这条设计方向应放弃，改为真正覆盖整行的形态。

入口：`evaluate_window_mask_routing.py --mode gt --target-mode <window|anchored|line>`。
只有 `--target-mode window` 具验收资格；`anchored` 与 `line` 的 `acceptance.eligible=false`、
`passed=null`，不得被读成记录配置通过。

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

补充（2026-09-22，同轮）：五卡并发已验证的显存上限见 `RUN.md`「显存并发上限」——
单 worker 约 `29 GB`，**每卡最多 2 个**，3 个会 OOM。
`run_window_mask_acceptance_a100.sh` 固定 5 个 worker，一次只能跑一个臂；
要并行多个臂须改用每卡单 worker 的全量串行形式。
另：远端 `/tmp` 与 `$HOME` 已不可写，所有临时文件与 `TMPDIR` 必须置于 `/data3` 下（见 `RUN.md`）。
