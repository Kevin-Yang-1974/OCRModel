# line100-window 失败归因：小分支移除 vs 整行换窗口

2026-09-22。状态：两臂运行中，完整结果待追加。

## 为什么跑这两个臂

`glmocr_line100_window_gt_accept_5gpu_20260922_150507` 的完整 3–5 字窗口 GT 验收为
CER `0.13693762903922793`，未过 `<0.13`，比历史整行 `line100` 的 `0.124694` 高
`0.0122436290`。但这一次融合同时改了两件事：

1. 移除了原 geometry/layout 小分支（不再安装 `LayoutAwarePatchMerger`）；
2. 空间目标从整行框换成行内 3–5 字窗口。

`LINE100_WINDOW_MASK_ROUTING.md` 的失败处置写死：先比较 `legacy-line` 与
`legacy-layout-control`，**不得**自行改 bias、指针、分辨率或评价子集。本文就是这两臂。

## 生成预算核对（先做，避免预算混淆）

| 项 | 3–5 字窗口 GT（本次验收） | 历史整行 `line100` |
| --- | ---: | ---: |
| `max_new_tokens` | **1536** | **1536** |
| 分辨率 | 4,000,000 | 4,000,000 |
| 触顶页 | `0/149` | `0` |

验收 run 的 `results/summary.json` 与五片 `protocol.json` 均记录
`max_new_tokens=1536`、`validation.generation_limit_hits=0`。历史整行由
`tools/training/run_glmocr_layout_oracle_line_eval_a100.sh` 产生，其
`max_eval_new_tokens` 同为 `1536`。**两轮同口径，`0.13693763` 与 `0.124694` 可直接比较**，
这个差值不是 `512` 截断造成的。

## 设置

| 项 | 值 |
| --- | --- |
| 臂 A | `glmocr_line100_legacy_line_diag_20260922_160451`，`--mode legacy-line` |
| 臂 B | `glmocr_line100_layout_control_diag_20260922_160451`，`--mode gt --legacy-layout-control` |
| 分支 / commit | `glm-ocr-layout-mask-routing` / `a980c4e` |
| 源码指纹 | `git archive` 压缩包 SHA256 `a8224cef505a16f1792393564471a5ac8dff27fa277f877b568b120b249bd222`（102 文件级工作树外的独立导出） |
| 入口 | `run_window_mask_acceptance_a100.sh <run_root> <mode> [--legacy-layout-control]` |
| 数据 | sparse24 validation 全 149 页，SHA256 `36ec845875e1ea18a48d4a523b6c5a3f007b46e2a07de0cafd2c26139929a348` |
| checkpoint | `glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000`，decoder LoRA SHA256 `ada4cdd3bb1f2f417d1c0a68054dc61e70ab59f215b3bc2018074b4f2fdcd5d5` |
| 推理协议 | `bias=1.0`、`synced` 指针、hard raster、prefill 不注入、全部 decoder 层、fast processor、4M、1536、BF16、math-SDPA、seed 42 |
| 五卡 | 物理 GPU `0–4`，每臂五个单卡 worker，`30/30/30/30/29`；两臂同时在跑（每卡两个进程），非 DDP |
| 不变量 | 两臂与验收 run 共用同一起点、同一 manifest、同一生成协议、同预算 |
| 产物 | `acceptance_runs/<run ID>/`：`status/run.json`、`shards/0..4.{log,summary.json,protocol.json}`、`results/merged.json`、`results/validation_predictions.jsonl` |
| 产出边界 | `acceptance_eligible=false`；merge 不写 `results/summary.json`，避免诊断臂被读成该融合的验收结论 |
| protocol 字段 | `test_manifest_read=false`、`test_used_for_selection=false`；不训练、不选点、无优化器 |

## 判读口径（预登记，写在跑之前）

1. **装置可比性**：`legacy-line` 应复现 `0.124694`。不复现则先修装置，其余差值不读。
2. **归因**：
   - `gt + control` 与 `0.13693763` 的差 = **小分支移除**的单独贡献；
   - `gt + control` 与 `0.124694` 的差 = **整行换 3–5 字窗口**的单独贡献。
3. 两臂均为诊断，不作验收、不作选点，也不用于调 bias 或窗口大小。

## 状态快照

**派发与启动（2026-09-22 16:04，Asia/Shanghai）**：两个 run 派发完成，preflight 通过
（149 页、Torch `2.8.0+cu128`、Transformers `5.3.0`），五卡 admission 利用率 0%、
`status/run.json=running`、`acceptance_eligible=0`。首轮进度检查时十个 worker 均在正常推进
（两臂各 shard 已产出 3–6 页），无 CUDA/OOM/NaN/Inf/Traceback。

**完成（2026-09-22 16:50 前后，Asia/Shanghai）**：两臂 `status=complete`，五片均 `complete`，
tmux 会话已退出。日志与 shard 日志错误扫描为空；两臂 `generation_limit_hits=0`。

## 结果

三臂同装置、同起点 checkpoint、同 manifest、同生成协议、同预算；差异只有空间目标形态
和是否装小分支。`gt` 列为本次验收 run，另外两列为本次诊断。

| 臂 | 小分支 | 空间目标 | CER | 替换 | 插入 | 删除 | 每步命中 token | gen tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `legacy-line` | 装 | GT 整行框 | **0.12469390694771211** | 3887 | 738 | 569 | 77.6 | 50400 |
| `gt`（验收） | 不装 | GT 3–5 字窗口 | **0.13693762903922793** | 3987 | 948 | 769 | 8.4 | 50321 |
| `gt`＋`--legacy-layout-control` | 装 | GT 3–5 字窗口 | **0.13693762903922793** | 3987 | 948 | 769 | 8.4 | 50321 |

三臂均 149 页、reference characters `41654`、`exact_page_rate=0`、`generation_limit_hits=0`。

### 1. 装置可比性：成立

`legacy-line` 复现历史整行值到小数点后 11 位：`0.12469390694771211` 对记录值 `0.124694`。
装置可比，下文的差值可以归因。该臂的 routing 诊断与历史一致：`bias=1.0`、`pointer=synced`、
`box_source=line`、`line_map=regions`、`gated_steps=0`、平均 `biased_steps=332.3/337.3` 步、
平均 `mean_boxes_hit=77.62`、平均 `missing_box_steps=1.32`。

### 2. 小分支移除是精确零效应（逐字节级）

`gt+control` 与验收 `gt` 的 `validation_predictions.jsonl` **SHA256 完全相同**：

```
846acbede98c2c96c00e40408c529bb7c9334f669d8d622eaf24289d5baff37a
```

三份文件（含本地归档副本）逐个核对一致。protocol 中控制臂的 `layout_branch_present=true`，
即小分支确实安装、adapter 权重确实加载，但 149 页输出**一个 token 都没变**。

**这是自由生成路径上的直接证实**，不只是 teacher forcing：`LAYOUT_WRITEBACK_INTERVENTION_RESULT.md`
在 teacher-forced 前向上测到 decoder 只使用页级共有分量 `H̄`、逐 patch 空间分量为零，
本轮说明该结论在 149 页 greedy 自由生成下同样成立。

**因此 0.012244 的差距不含任何「小分支移除」成分。**

### 3. 差距 100% 来自空间目标；配对 bootstrap 两档均显著

`legacy-line` 对 `gt`（逐页配对 bootstrap，10000 次，`tools/analyze_cer_significance.py`）：

| 口径 | ΔCER | 95% CI | 判定 |
| --- | ---: | --- | --- |
| 按页 | `−0.012244` | `[−0.022881, −0.003519]` | **显著**（P(`<=0`)=0.999） |
| 按卷号分组（33 卷） | `−0.012244` | `[−0.043107, −0.003280]` | **显著**（P(`<=0`)=0.998） |

最坏情况仍支持 `−0.043107`。**按卷分组这一更严口径下依然显著**——该口径此前打掉了
`track100` 的 `+17.8%`（`LAYOUT_TRACKING_CORRECTION_RESULT.md`），因此这个结论不依赖
「同书页相互独立」的偏窄假设。

### 4. 失败签名：替换不动、插入与删除同向上升

| | 替换 | 插入 | 删除 |
| --- | ---: | ---: | ---: |
| 整行 → 窗口 | 3887 → 3987（+2.6%） | 738 → 948（**+28.5%**） | 569 → 769（**+35.1%**） |

与 `line100` 相对 `noroute` 的收益签名（插入 −52%、删除 −37%、替换 −0.1%）**方向完全相反、量级相当**。
窗口削掉的正是行级约束提供的那部分收益：「少跳过、少重复」。认字本身（替换）几乎不受影响。

### 5. 机制：每步覆盖 token 从 77.6 掉到 8.4

| 臂 | 平均每步命中 token | 每次命中每 key 加性总量 |
| --- | ---: | ---: |
| `legacy-line`（整行） | 77.62 | 77.62 |
| `gt`（3–5 字窗口） | 8.38 | 8.38 |

`B=1.0` 两臂相同。3–5 字窗口并非「更精确的点目标」：每个字符只在自己那一步被覆盖，
相邻字符的窗口几乎不重叠，所以平均每步只有约 8 个格子被偏置。整行框则让同一行的
约 78 个格子**在整行持续被偏置**，提供的是跨整行的持续行级上下文。

**归因边界（必须标注）**：本轮是「框大小＋B＋页集多变量同变」的跨实验比较，
`LAYOUT_ORACLE_LINE_RESULT.md` §7.1 已预先划定此类比较**不能断定**是「每 key 强度」
还是「覆盖重复度/总量」在起作用。要分离需刻意做一次「固定 B 变覆盖」或「固定覆盖变 B」
的扫描。本轮数据给不出该判定。

### 6. 对验收未过线的处置结论

- 验收 `0.13693763` 与历史整行 `0.124694` 的差 `0.012244`，**全部由整行框换成 3–5 字窗口引起**，
  与小版面分支的移除无关（后者为精确零）。
- 生成预算不是原因：两轮均 `max_new_tokens=1536`、触顶 `0/149`。
- 失败处置按 `LINE100_WINDOW_MASK_ROUTING.md` 执行完毕：先比较 `legacy-line` 与
  `legacy-layout-control`，**未改动 bias、指针、分辨率或评价子集**。bias 仍为 `1.0`。

### 7. 协议字段

两臂：`test_manifest_read=false`、`test_used_for_selection=false`、`usable_for_selection=false`、
`reads_ground_truth_for_routing=true`、`acceptance=null`、`acceptance_eligible=false`。
不训练、无优化器/学习率/梯度累积、无选点。诊断臂不写 `results/summary.json`，
只写 `results/merged.json`，避免被读成该融合的验收结论。

### 8. 下一步（登记时确定）

按现成目标形态构成阶梯，做**受控单变量**对照以厘清第 5 节的归因边界：
`line`（整行，77.6 token/步）→ 行内窗口（8.4）→ **窗口＋锚点**（行内窗口与当前行剩余范围的并集）。
先以零 GPU 成本重放 `build_mask_targets` 量出第三种形态的每步覆盖分布，确认阶梯单调后再投 GPU。
