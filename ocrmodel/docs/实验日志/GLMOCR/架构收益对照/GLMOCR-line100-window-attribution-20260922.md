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

## 结果

待两个 run 进入终态后追加：每臂的完整 micro CER、I/D/S、reference characters、
generation tokens、EOS/触顶/循环率、finite 与错误检查，以及上面两条归因差值。
