# PVLD 四项有界诊断与修改方案（2026-08-25）

## 1. 材料与执行口径

- 证据状态：原始 run 与 JSON 字段为 `VERIFIED`；原因判断与修改优先级为 `ANALYZED`。
- 成功 run：`pvld_bounded_diagnostics_20260825_v2`。
- 输出目录：`/data3/yky/yangky_ocr_models/evaluation_runs/GOT/pvld_bounded_diagnostics_20260825_v2`。
- C4：`mthv2_pvld_causal_20260822_v1_C4_seed42/p2/model/checkpoint-40000`。
- C5：`mthv2_pvld_causal_20260822_v1_C5_seed42/p2/model/checkpoint-30000`。
- 数据：MTHv2 原始整页 `mthv2_layout_page_v1`；train 与 validation 各按真实区域数 `0-8/9-16/17-32/>32` 固定取首个样本，共 8 页；OCR routing 只取 validation 的 `0-8` 与 `>32` 两页。
- 输入仅为 `whole_page_image + ocr_prompt`；bbox、direction、reading_order 未作为推理输入。
- 执行设备：显式 GPU 4；峰值已分配显存 `4,936,875,520` bytes（约 4.60 GiB）。
- 边界：未读取 MTHv2 test，未运行 optimizer step，未写 checkpoint，未启动训练，未查询或使用 GPU 2。
- 首次 run `pvld_bounded_diagnostics_20260825_v1` 因直接视觉塔调用的输入为 FP32、权重为 BF16 而受控失败；失败产物保留。修复只统一输入 dtype，未改变诊断定义。

## 2. 诊断一：oracle-prefix 与 free-prefix

| 模型 | oracle 边界准确率 | oracle EOS 概率 | oracle bbox IoU | oracle duplicate | free 预测/真实区域 | free EOS | token cap | free duplicate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C4 | 0.9496 | 0.4090 | 0.2113 | 0.0886 | 32.75 / 26.375 | 0.875 | 0.125 | 0.1157 |
| C5 | 0.9524 | 0.2631 | 0.1927 | 0.1016 | 34.25 / 26.375 | 0.875 | 0.125 | 0.1219 |

正确历史下约 95% 的 REGION/EOS 边界可判对，因此现有 2-layer、hidden-size 256 decoder 并非完全没有结构建模能力。但真正终止点的 EOS 概率仍低，且自由历史会放大错误。最复杂 train 页真实 71 个区域：C4/C5 均生成 103 个 REGION 后在 512-token 上限截断；真实终点的 oracle EOS 概率分别仅 `0.0130/0.0311`。

重复框并非只由自由历史或 FSM 失败造成。上述复杂 train 页即使使用 oracle prefix，C4/C5 预测框的重复率都已达到 `0.6056`，首个重复均出现在第 15 个 REGION；free prefix 进一步升至 `0.7184`。这表明 bbox/空间记忆在正确 token 历史下也会塌缩。当前 previous-REGION hidden cumulative mean 不能可靠表达“哪些空间位置已经覆盖”。

`max_layout_records=128` 在本次 512-token 诊断中未触发；每条记录需要 5 个 token，token cap 会先于该 record cap 生效。正式配置 `2048 tokens/512 records` 同样最多容纳约 409 条完整记录，因此 512-record cap 不是当前异常生成的有效保护。

## 3. 诊断二：train 小样本与 validation 自由生成

| 模型 / split | 预测/真实区域均值 | EOS | token cap | duplicate |
|---|---:|---:|---:|---:|
| C4 train | 35.75 / 28.25 | 0.75 | 0.25 | 0.1796 |
| C4 validation | 29.75 / 24.50 | 1.00 | 0.00 | 0.0518 |
| C5 train | 38.00 / 28.25 | 0.75 | 0.25 | 0.1796 |
| C5 validation | 30.50 / 24.50 | 1.00 | 0.00 | 0.0643 |

同一训练集页面也存在过预测、重复和 token-cap，故问题不是单纯的 held-out 泛化不足。结合既有 P1 validation 曲线（step 2000-12000 的 EOS/count/F1 非单调，step 4000 优于 step 12000），当前证据反对“只增加 P1 steps”。样本很小，不能据此估计总体误差，但足以否定“训练页自由生成已经正常”这一解释。

## 4. 诊断三：逐 loss 梯度范数与方向

梯度审计使用同一个 validation `17-32` 页、只执行 forward/backward 型 `autograd.grad`，没有 optimizer step。梯度组互不重叠：`layout_evidence`、`causal_decoder`、`record_heads`、`visual_writeback`、`projector`。

| 模型 | weighted OCR | weighted layout 分量和 | evidence: OCR | evidence: bbox L1 | evidence: bbox GIoU | evidence: count |
|---|---:|---:|---:|---:|---:|---:|
| C4 | 0.5001 | 3.1688 | 0.0344 | 5.5996 | 13.0675 | 7.7988 |
| C5 | 0.3410 | 2.7723 | 0.2800 | 1.6905 | 8.8794 | 8.1721 |

C4 的 evidence 梯度范数中，bbox L1、bbox GIoU、count 分别约为 OCR 的 `163x/380x/227x`；C5 约为 `6.0x/31.7x/29.2x`。C5 的 OCR 与 bbox L1 在 evidence 上 cosine 为 `-0.3501`，存在明确方向冲突；bbox GIoU 与 OCR 的 cosine 仅 `0.0602`，基本正交。C4 的 bbox GIoU cosine 也仅 `0.0710`。

OCR 对 `causal_decoder/record_heads` 的梯度严格为 0，布局 loss 对 `visual_writeback/projector` 的梯度严格为 0。这与当前计算图一致：布局解码质量只能通过共享 `layout_evidence` 间接影响 OCR；生成的 bbox、顺序、EOS 和 coverage 并不进入 OCR routing。因此“布局 decode 成功”在当前架构中不能保证 OCR 提升。

## 5. 诊断四：OCR routing 因果消融

| 模型 / 条件 | teacher-forced OCR NLL | 两页 generation CER |
|---|---:|---:|
| C4 normal | 2.3956 | 0.7781 |
| C4 alpha=0 | 2.4241 | 0.7653 |
| C4 shuffled evidence | 2.4034 | 0.7883 |
| C5 normal | 2.3338 | 0.7857 |
| C5 alpha=0 | 2.3768 | 0.7398 |
| C5 shuffled evidence | 2.4055 | 0.7806 |

normal 的 teacher-forced NLL 均优于 `alpha=0`，且 shuffled evidence 会使 NLL 变差，说明 residual 与页面 evidence 确实被计算图使用，并非恒等死路。但在这两个固定页面上，`alpha=0` 的自由 generation CER 反而更低：C4 低 `0.0128`，C5 低 `0.0459`。这只是一项因果 smoke，不是 validation 性能结论；它说明当前 routing 尚无稳定的 OCR 收益证据，也说明 token-level NLL 改善不能替代自由生成 CER。

## 6. 原因排序

1. **首要：终止目标不足且 count 与生成脱节。** 普通 full-sequence CE 被大量 FSM 唯一合法转移稀释；每页只有一个 EOS，而 REGION 边界随区域数增长。count head 在抽样页的 oracle MAE 已明显小于自由生成 count MAE，但生成完全不读取 count head。
2. **首要：coverage 缺少空间占用状态。** previous-REGION hidden mean 丢失顺序和空间位置；oracle prefix 下复杂页仍大量重复，不能只归因于 exposure bias。
3. **首要：P2 共享 evidence 梯度失衡。** bbox GIoU 与 count 的共享梯度远大于 OCR，部分 loss 与 OCR 冲突；直接统一乘一个 layout weight 会同时削弱本应继续学习的 decoder/record heads，控制粒度过粗。
4. **首要：OCR routing 未使用预测记录。** 当前 OCR 只读生成前的 global evidence，不读预测 bbox/order/EOS；因此没有“layout decode 改善必然带动 OCR”的结构因果链。
5. **次要且待证：decoder 容量或 prompt query 数。** oracle 边界约 95%，且 32 prompts 不是 32 个区域槽。现有证据不足以支持先改成 Qwen decoder或先将 32 增到 64。

## 7. 修改方案与有界消融顺序

### M1：边界损失与 count-conditioned stopping

保留当前 token head 和 FSM，新增仅在 record boundary 计算的 page-balanced loss：

`L_boundary = 0.5 * mean_page(mean_nonterminal(-log p(REGION))) + 0.5 * mean_page(-log p(EOS at gold terminal boundary))`

该项与原 sequence CE 并存，避免 EOS 被每页多个 REGION 及确定性内部 token 稀释。将现有 count head 前移为 page prior `c_hat`，在 training forward 与 generation 的同一 boundary logits 上加入相同的 learned count condition；不使用 test 调系数。另设只在合法边界生效的动态工程上限 `ceil(c_hat + margin)`，margin 只由 validation 锁定，并继续与 token-cap/静态 record-cap 分开报告。

第一组消融固定为：current、`+L_boundary`、`+L_boundary+count condition`。只有 validation 的 EOS、count MAE、F1 同时改善才进入 M2。

实施说明（2026-08-25）：当前先落地 M1a=`L_boundary+count condition` 单线，用最少一条正式线验证组合是否值得继续。动态 `ceil(c_hat+margin)` record cap 未在 M1a 中启用，避免同时改变停止 bias 和硬上限而无法归因；静态 `max_layout_records=512` 仍仅作工程安全上限。M1a 不新增参数，默认系数 0 可回退旧路径；本轮不运行 test，也不据既有 test 调系数。

### M2：空间 memory 与显式 coverage

decoder memory 从仅 `A[32,256]` 改为 `[A; P(F)]`，其中 `F` 为 16x16 高分辨率视觉 token，`P(F)` 投影到 decoder hidden size。优先复用 prompt attention 的视觉投影，避免无必要增加参数。

每层保留紧凑空间 coverage `C^l in [B,T,256]`：只累积此前 REGION 对 16x16 空间 token 的 cross-attention，下一层对已覆盖位置施加可学习负 bias。首版 coverage attention 在更新后 detach，避免模型通过 attention 权重投机降低 loss；不保存完整 attention 到 prediction。training 与 generation 都调用同一 decoder block 和同一 coverage 规则。首版仍不加入 duplicate loss。

M2 以 oracle/free duplicate、matched IoU、count/EOS 为判断指标。只有 M2 后 duplicate 仍严重，才单独试验简单 coverage/duplicate penalty。

### M3：P2 路径级梯度缩放

P1 保持布局梯度完整。P2 对共享 evidence 与 record-to-decoder 采用前向恒等、反向缩放：

`A_layout = stopgrad(A) + s_shared * (A - stopgrad(A))`

`H_record = stopgrad(H) + s_record * (H - stopgrad(H))`

这样 bbox/count heads 仍收到完整监督，causal decoder 主要由 sequence/boundary 学习，共享 prompt evidence 不再被大尺度布局回归梯度压过 OCR。预注册 `s_shared/s_record` 小集合，只按 validation 选择；不先引入 PCGrad 等更复杂方法。每个候选同步记录本诊断中的梯度范数与 cosine。

### M4：让预测布局真正条件化 OCR

在 M1-M3 使自由布局稳定后，再让 OCR router读取**模型自由预测**的 bbox/order/direction/confidence；不得读取 teacher-forced gold token 或 gold bbox。预测记录只调制 attention query/key、空间 mask 和 reliability gate，OCR attention Value 继续只来自视觉 token。

首版建议在 validation-selected P1 后冻结布局分支，使用同一 P1 对 train 页面预计算自由预测，并在 P2 与推理使用同一 predicted-layout routing。对未生成 EOS、token-cap 或低 confidence 页面降低 residual reliability，保留 `alpha=0` 原始 GOT2 回退。M4 必须在预注册 validation 子集及完整 validation 上同时报告 normal、`alpha=0`、shuffled evidence；只有 normal 稳定优于两者才声称 routing 有效。

### M5：最后再测 query 与 decoder 容量

完成 M1-M4 后，才比较 prompt queries `K=16/32/64`；K 表示 global evidence prompts，不表示区域槽位。若 oracle-prefix 仍明显差，再比较 decoder hidden/layers；若 oracle 良好而 free 仍差，继续修目标与反馈，不替换成另一套 Qwen。所有容量消融必须保持数据、P1/P2 exposure、seed、选点规则和 OCR routing 相同，并单列参数量与计算量。

## 8. 预注册判断门槛与结论边界

- P1 选点继续只读 validation：优先停止错误、count MAE、region F1、bbox IoU、duplicate、较早 step；不按 OCR CER 选 P1。
- P2 以 validation OCR CER/去空白 CER 为主，同时要求 layout 指标不发生灾难性退化。
- 每个结构候选必须重复本四项诊断；normal routing 未优于 `alpha=0` 与 shuffled evidence 时，不进入性能主张。
- 既有 MTHv2 test 数字只用于描述已发生的失败，不参与本方案权重、margin、query 数或 checkpoint 选择。
- 当前证据来自单 seed 和极小诊断子集，只能确定故障机制与修改优先级；尚无新结构性能结论。
## M2-M4 bounded diagnostic contract

The current bounded smoke checks `[A;P(F)]` memory, spatial padding-aware coverage `[B,L_F]` with detached updates, finite decoder/record/routing gradients, and M3 shared-evidence OCR/layout gradient norms plus cosine. M4 checks free predicted-layout routing, reliability reduction for EOS/truncation, shuffled controls, visual-only OCR Value, and exact alpha-zero fallback. These are implementation checks only; no M2-M4 validation or test performance is reported.

Bounded results: A100 GPU 0 run `pvld_m2_m3_m4_cuda_smoke_20260825_v7` completed successfully with finite loss `8.807126`, visual-value gradient norm `0.011863`, M3 shared-evidence cosine `0.386748`, coverage shape `[3,16]`, valid 0/1/3-region generation, and distinct token-cap/record-cap states. BSCC Slurm job `1452473` on `paraai-n32-h-01-agent-41` reproduced the same checks (`status=ok`, elapsed 20 s). Neither run read test data or started formal training.
