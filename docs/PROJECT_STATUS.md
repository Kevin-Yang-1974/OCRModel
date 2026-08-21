# 项目状态

> 更新日期：2026 年 8 月 20 日

## 当前状态

- 正式数据：`ancientdoc_layout_260707_group_isolated_seed20260815`，train/validation/test=`1548/516/516`，五类跨 split 泄漏均为 0。
- 正式流程：C1/C4 训练、C4 validation-only 分支点选择、同起点 C5/C6 replay、C1/C5/C6 validation 选优和一次 frozen test 均已完成。
- C4 分支点：`checkpoint-6000`，validation 页面 CER `0.567493`；C5/C6 的起点路径、step、权重哈希一致，optimizer 和 scheduler 均为 fresh。
- Frozen test：`ancientdoc_12k_frozen_test_seed20260815_gpu0_retry1`，每项 516 页。C0/C1/C4/C5/C6 页面 CER 为 `1.328217/0.485815/0.457264/0.546852/0.515768`，去空白 CER 为 `1.113916/0.459547/0.438587/0.490939/0.464478`，完整页面 exact match 均为 `0/516`。
- 当前模型排序：C4、C1、C6、C5、C0；当前 AncientDoc 主 checkpoint 为 C4 `checkpoint-6000`。
- 离线验证：已对 C4-C1、C5-C4、C6-C4、C6-C5 执行逐页分析和 27 个书籍组、10,000 次 paired cluster bootstrap。只有 C5-C4 的 95% interval `[0.009897, 0.185221]` 排除 0；其余三项均跨 0。完整报告见 `docs/ANCIENTDOC_GROUP_ISOLATED_ANALYSIS_20260816.md`。
- GOT2 结构消融：A1–A5 已在 `formal_pdf_short_seed20260812` 完成 validation-only 选点和一次 Synthetic-ID frozen test。A1/A2/A3/A4/A5 test page CER 分别为 `0.086320/0.158236/0.069977/0.235795/0.070531`；A4/A5 complete layout F1 为 `0.609785/0.695752`。A3 OCR 最低；A5 OCR 与 A3 接近且布局更完整，但 A5 额外包含 P1 4000 steps，不能视为同总预算。完整记录见 `docs/GOT2_LAYOUT_ABLATION_RESULTS_20260817.md`。
- 多样化 synthetic→AncientDoc 新协议：已实现 ancient-photo-diverse preset、跨 split 生成/审计入口、synthetic A5 P1/P2 validation-only 选点、selection-locked C4 初始化，以及 C0/C1/C4/C5/C6 分阶段 selection/test。新 C5/C6 默认 AncientDoc:synthetic=`7:1`（12.5% replay），旧 frozen run 的 `3:1`（25%）结果不改写。代码已同步 A100；`layout_ablation_smoke_20260817_114133` 在 GPU 0 通过 1-step P1/P2、checkpoint 与 7:1 schedule smoke。正式多样化数据、长程训练和新 AncientDoc test 尚未运行。
- MTHv2 数据转换：已将官方 `label_textline` 转为有序区域标注，并把超过 `max_regions=16` 的页面按连续阅读顺序转换为 oracle chunks。当前数据为 train 2159 源页/5589 chunks、validation 240 源页/593 chunks、test 800 源页/1968 chunks；该输入协议不是 whole-page 端到端列发现，不能与 whole-page 指标直接比较。
- MTHv2 当前运行：A100 正在对 C1–C5 执行 chunk-level validation checkpoint 选择和 selection-locked test，run prefix 为 `mthv2_chunk_ablation_20260819_multi`。截至 2026-08-19 本次检查，五组日志均已记录 `validation_started`，但 `selection.json` 和 test `summary.json` 均未生成，尚无性能结果。
- MTHv2 chunk test 更新（2026-08-20）：五组 validation selection 已完成且均为 `selection_split=validation`、`test_used_for_selection=false`。C1 `projector_only` 选择 step 30000，validation page CER `0.621899`；C2 `generic_adapter_projector` 选择 step 42000，validation page CER `0.632319`；C3 `vlqa_ocr_only` 选择 step 42000，validation page CER `0.599550`；C4 `vlqa_layout_direct` 选择 step 39000，validation page CER `0.677042`；C5 `vlqa_layout_p1_p2` 选择 step 30000，validation page CER `0.642191`。C1、C3 和 C5 的 selection-locked test 已完成：C1 page CER `0.647299`、page exact match `3/1968`；C3 page CER `0.587229`、page exact match `1/1968`，但 complete region precision/recall/F1 仅 `0.000159/0.000198/0.000176`，matched direction accuracy `0`，说明无布局监督对照不具备有效布局预测；C5 page CER `0.660571`、page exact match `2/1968`，complete region precision/recall/F1 `0.416397/0.418375/0.417384`，matched bbox mean IoU `0.749395`，matched direction accuracy `1.0`，reading-order pair accuracy `0.999492`，Kendall tau `0.998984`。C2、C4 已进入 test 但尚无完成事件或 summary，不能写成 suite 已结束。
- 该轮 test 的 selection/test manifest 路径仍为 `mthv2_layout_column_chunks16_v1`，所以全部结果只能称为 oracle-chunk 中间诊断，不能与 whole-page 训练或整页 PVLD 直接比较。C1/C5 summary 中的 `input_granularity=whole_page_image` 与实际 chunk manifest 路径不一致，已按 manifest 证据保守记录为 chunk-level，后续需修复 evaluator 元数据字段。
- 当前可做的有限比较是：C5 相对 C1 的 chunk page CER 高 `0.013272`（`0.660571-0.647299`），exact match 少 1 个，尚无 OCR 收益；C5 同时给出 complete layout F1 `0.417384`，但不能用布局指标替代 OCR。validation 上 C3 的 page CER 暂时最低，但其 frozen test 尚未完成；在 C2–C4 test 和源页面聚合完成前，不形成 C1–C5 最终排序。
- MTHv2 报告边界：当前 evaluator 产出 chunk-level `layout_validation_metrics.json` 和 predictions；正式报告还需按 `source_page_id`、`chunk_index` 合并为 240 个 validation 源页和 800 个 test 源页。现有脚本明确标记 `grouped_source_page_evaluation=pending`，因此当前 chunk 结果只能作为中间诊断。
- MTHv2 whole-page 训练：已启动独立 tmux 会话 `mthv2_page_train_20260820_r2`，使用原始页面 manifest `mthv2_layout_page_v1`（train/validation/test=`2159/240/800`），不是 oracle-chunk 图像；C1–C5 在 GPU 0 串行执行，`max_regions=512` 覆盖整页最多 407 个区域。C1 已进入 P2 optimizer steps，暂无性能结果。此前 `mthv2_page_ablation_20260820` 因审计条件未识别 `real_mthv2_official` 失败，`mthv2_page_ablation_20260820_r1` 因 `max_regions=64` 不足以容纳 71 区域页面失败；两个失败 run 均保留，不得复用。
- 方案备案：原有布局结构现统一称为 Fixed-Slot VLQA baseline，保留 `Fixed-Slot VLQA-K16`（原始 `max_regions=16`）和 `Fixed-Slot VLQA-K32`（仅提高固定槽位容量到 32）两个对照。两者的 query 在训练期按阅读顺序对应固定区域，超过上限时仍需 oracle chunk 或截断；K32 不是变量长度解码创新。
- PVLD-32 原型：新增独立的 Prompted Variable-Length Layout Decoder 工程候选。32 表示全局 layout prompt tokens，不表示 32 个区域槽位；区域数量由 decoder 的 `REGION` 记录和 `EOS` 决定。新增模块、target 序列化、评估契约和 A100 编排器均不覆盖 Fixed-Slot 代码；当前尚未接入 GOT2 视觉塔/`forward`，也未产生正式训练或性能结果，只有 standalone feature tensor smoke/preflight 能力。
- PVLD 路径修订（2026-08-20）：`GOTQwenModel.forward` 现将 `vision_tower_high(image[1])` 的高分辨率中间特征作为 `F`/第一阶段 Key-Value，将 `mm_projector_vary` 输出作为 `V_i`。全局 prompt 读取 `F` 得到命名为 `layout_evidence=A` 的布局证据；A 只进入布局分支和第二阶段视觉路由。新增显式 `layout_writeback_mode=visual_value_layout_routing`，用两跳 `V_i→A→V_i` 因子化路由保证最终 Value 只来自视觉 token，输出保持 `[B,L_v,D_v]` 后再送入 Qwen OCR。旧 `layout_value` 与 `vqlca` 保留为历史对照，不改写既有 VQLCA 训练结果。
- A100 VQLCA smoke：`vqlca_wholepage_smoke_20260820_r1` 已在 GPU 2、MTHv2 原始整页 train manifest 上通过 `2159` 页/`72688` 区域审计、CUDA component forward/backward、gate=0 原路径等价、gate 打开后的 visual Q/K/V、layout-conditioning、context-key、output 与 layout-query finite/nonzero gradient 检查，以及 P1 1 step→checkpoint 重载→P2 1 step。P1/P2 train loss 为 `9.048427/9.792988`，只证明工程链路。
- GPU 默认策略已写入根目录和 `ocrmodel/AGENTS.md`：训练、验证与评估默认在命令允许的 GPU 池中使用全部瞬时 `utilization.gpu < 50` 的卡；合格卡足够时按控制一一绑定，不足时全部合格卡组成多卡作业并按控制串行运行。
- 原 VQLCA 会话 `mthv2_page_vqlca_train_20260820` 按用户要求在 C1 P2 约第 `4245/42000` 步停止。第一次重启 `_r1` 因 GPU0 在子任务启动前升至 `72%` 而按准入规则退出，未启动控制任务且保留诊断日志。
- 当前 VQLCA whole-page C1–C5 会话为 tmux `mthv2_page_vqlca_train_20260820_r2`，run prefix `mthv2_page_vqlca_ablation_20260820_r2`，使用 GPU `1,2,3,4` 四卡；C1 已通过 `3199` 页/`105579` 区域审计并进入真实 P2，`physical_gpus=["1","2","3","4"]`、`world_size=4`。数据固定为原始整页 `mthv2_layout_page_v1`，不是 chunk，`max_regions=512` 仍是 Fixed-Slot K512 工程容量。旧 `mthv2_page_train_20260820_r2` 未停止或覆盖；尚无 VQLCA 性能结论，也未启动新的 frozen test。
- 训练状态与未来报告从 `/data3/yky/yangky_ocr_models/training_runs/GOT/mthv2_page_vqlca_ablation_20260820_r2*` 提取；总启动日志为对应 `_tmux_logs/launcher.log`，单项状态和汇总分别看 `metadata/status.txt` 与 `summary.json`。

## 已完成的正式流程

1. 对全部 C4 周期 checkpoints 运行 validation-only selection，选择 `checkpoint-6000`。
2. 从该 C4-best 独立训练 C5/C6，并完成分支 provenance 检查。
3. C1/C5/C6 按 validation 选 best，C4 固定为分支点 C4-best。
4. 冻结五个模型后，在同一 516 页 test 上统一评估一次。

代码已提供 `train-core`、`select-c4`、`train-replay` 三阶段入口。旧 `C4-final→C5/C6` 流程已废弃。正式命令与紧凑回传见 `docs/SYNC_AND_RUN.md`。

## 归因边界

C5/C6 共享 C4-best 后，C6-C5 的页面 CER 为 `-0.031084`，说明 synthetic layout supervision 相对纯 OCR replay 减轻退化；但 C5-C4 和 C6-C4 分别为 `+0.089588` 和 `+0.058504`，两种 replay 均未超过 C4。C1 与 C4 的结构、可训练参数和上游训练历史仍不同；普通 GOT2 的同上游历史、同数据和近似参数预算 C2 尚未实现，因此不能把 C1-C4 差异完全归因于 VLQA。当前 frozen test 不再用于 replay 配置选择，后续调整只允许使用 validation。

PVLD-32 必须与 Fixed-Slot VLQA-K16、Fixed-Slot VLQA-K32、无布局监督 query、等参数量普通 adaptor 在统一 whole-page 数据协议和统一 optimizer budget 下比较。PVLD-32 的评估必须额外报告 EOS 成功率、提前 EOS、max-length 截断、区域数量 MAE、按真实区域数量分桶的区域召回率/bbox IoU 和 OCR CER；在这些消融完成前，不把 PVLD-32 称为已成立的结构创新。
