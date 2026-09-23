# Line-mask v3 Stage D：复杂度覆盖诊断

## 预注册信息

- Run ID：`glmocr_line_mask_v3_stage_d_20260923_v1`
- 状态：Stage D tmux 正在运行MTHv2 diagnostic32四臂推理。57文件远端源码快照已同步并与本地一致（source-tree SHA-256 `9ebd6017f1f7a97071837b010b188689f7bbc8237381d7f15268544b7ada7c85`）；raw GLM-OCR原始revision指纹：1个权重文件SHA-256 `a16eb0de98d199293371c560f95f83130d2a2c9612449df16839f08ff9498815`，aggregate SHA-256 `cde6532e869801998c91ea7ac209f349e281d89859d0ed3eba59066a27f1eabe`；配对LoRA指纹匹配预注册SHA。
- 日期：2026-09-23（Asia/Shanghai）
- 仓库：`D:\yangky\学推计划-glm-ocr`；branch `glm-ocr-layout-mask-routing`；启动前HEAD `ea73f182bc7ebe69f7c7386024b3b97bf89e392a`；工作区有其他已存在改动，不提交、不覆盖。
- 入口：`ocrmodel/tools/evaluation/run_line_mask_v3_diagnostic_a100.sh`；执行器 `diagnose_line_mask_v3.py`；配置 `ocrmodel/configs/line_mask_v3/diagnostic.json`。
- A100 主机：`a100-yky`；远端产物根：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/line_mask_v3/glmocr_line_mask_v3_stage_d_20260923_v1/`；tmux=session/run ID 同名；Slurm job/array：不适用。
- 远端源码暂存目标：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/line_mask_v3/code_snapshots/glmocr_line_mask_v3_stage_d_20260923_v1/ocrmodel/`；run 内不可变源码快照 SHA：启动后生成并追加。
- 本地结果归档目标：`D:/yangky/glm-ocr-assets/line-mask-v3/glmocr_line_mask_v3_stage_d_20260923_v1/`。

## 数据协议

所有诊断和候选比较均属于 `val_tune`。此前参与 v2 head 选点的 MTHv2 sparse24 149 页继续属于 `val_tune`；本次改用完整官方240页 validation，其中包含该149页和另外91页。其标注区域/行数范围为8–162，历史 sparse24 集合上限为24。32页筛查按来源、全局区域/行数密度分位、密度分位内文本长度分位分层，且至少8页必须高于历史24行上限，不能退化成 sparse-only 样本。敦煌／地方志80页已有4–32个区域的复杂度跨度。

| Split / domain | Pages | Manifest | SHA-256 |
|---|---:|---|---|
| MTHv2 train | 2159 | `/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1/train/manifest.char.jsonl` | `1016198040944e39329712eb2a7bdfe6db7526134b91d750c83afa91d854f9b3` |
| MTHv2 val_tune | 240 | `/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1/validation/manifest.char.jsonl` | `efc29e22b42fb81c8915282f935ef6c77ddcdcb7169f7a48bae6471c6c0cd315` |
| Dunhuang/local-gazetteer train | 240 | `/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat/train/manifest.jsonl` | `00ae8c30fc12046586cce836897af26b7a701a749fa10ff805bb4ae8022fb29d` |
| Dunhuang/local-gazetteer val_tune | 80 | `/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat/validation/manifest.jsonl` | `e20d2f9b07e535ccfab03c95a8f81222f75ee4f5a793e27dabb28338f48316e2` |

- Seed 42. The run locks 32 val_tune pages/domain before candidate inference; after screen it evaluates full MTHv2 240 and Dunhuang 80 pages with D0/D1 plus at most one selected candidate.
- Source-group/page-ID/image SHA/dHash overlap audits are generated before GPU launch. Their result is pending. No source-isolation claim is made before that audit.
- `val_verify`: no eligible independent manifest is locked or read. Results remain development/val_tune results.
- Test: **not read**. `test_manifest_read=false`; `test_used_for_selection=false`.

## Shared model start and fixed protocol

- Raw base: original GLM-OCR revision `ca5d8b3e287e52589e37c28385d9655ee4372f9d` at `/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d`. Directory presence is confirmed. Original safetensors inventory and aggregate SHA are computed and locked by the launcher before GPU admission; value pending.
- Frozen decoder LoRA: `/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/training_runs/glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000/decoder_lora.safetensors`; SHA-256 `ada4cdd3bb1f2f417d1c0a68054dc61e70ab59f215b3bc2018074b4f2fdcd5d5`; rank 8, alpha 8, dropout 0. This is frozen and loaded in Stage D to match the line-mask head’s training input distribution. Non-diagnostic R/C stages use raw GLM-OCR without this LoRA.
- Frozen head: `/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_line_mask_v2_full_20260922_1940/epoch-8.pt`; SHA-256 `d56023f270fd51812b58272141f8903bbd2ca1b437b8565b20ab5d12c7f59cac`; epoch 8/step 3456.
- Input: whole-page image and fixed `Text Recognition:` prompt. Fast processor, max image pixels 4,000,000, greedy generation, max new tokens 1536, BF16 backbone, FP32 mask head, Transformers SDPA forced to PyTorch `SDPBackend.MATH` (`torch_sdpa_math`).
- No training or optimizer: head LR=0, decoder-LoRA LR=0; warmup/schedule/steps/loss weights/gradient accumulation are not applicable; all backbone, LoRA and head parameters are frozen. Inference uses five independent single-GPU workers, physical GPUs 0–4, one shard per worker; DDP=false; per-worker page batch=1.
- The first GPU admission passed with physical GPUs 0–4 at `0/0/0/0/0%`. Before each inference wave the launcher queries only physical GPUs 0–4 and proceeds only if every instantaneous `utilization.gpu < 50%`; otherwise it exits and retains the prepared protocol. Before launch, the head SHA, decoder-LoRA SHA and both val_tune manifest SHAs were rechecked remotely and matched the locked values.

## Arms and budgets

| Arm | Only changed variable | Setting |
|---|---|---|
| D0 | Routing disabled | Shared raw GLM-OCR + frozen checkpoint-3000 LoRA + frozen v2 head; no mask bias |
| D1 | Reference route | Hard mask threshold 0.5; all decoder layers; bias 1.0 |
| D2 | Bias dose | D1 with bias 0.5 |
| D3 | Layer scope | D1 with layers `index >= floor(L/2)` |

All four screen arms share the same base revision, LoRA SHA, head SHA, data order per locked manifest, processor, prompt, token budget and backend. Screen budget is 32 pages × 4 arms × 2 domains = 256 page/arm inferences. Candidate selection uses only the two-domain 32-page val_tune screen: D2/D3 are eligible only if no worse than D1 in both domains; eligible candidates rank by equal-domain mean relative CER change, ties favor D2. Full val_tune includes D0, D1 and at most the selected candidate, giving equal per-arm page counts within each domain.

## Results and completion fields

- Current status: MTHv2 diagnostic32 D0–D3 inference is running; first-wave GPU admission passed. Source and protocol fingerprints are locked; Dunhuang diagnostic32 inference and the full val_tune stage have not started.
- CER/I/D/S, exact-page rate, EOS, generation-limit hits, loop rate, line-mask IoU/mass metrics, alignment coverage, trace outputs, checkpoint finite status and error scan: pending; no values inferred from previous runs.
- Validation selection: no checkpoint selection in this frozen-weight diagnosis. The 32-page rule selects at most one D2/D3 configuration for full val_tune; full 240/80-page results remain val_tune and do not create an independent val_verify result.
- Failed/interrupted status, last phase, output retention and reuse judgment: pending. Any failure will be appended here and registered; no automatic restart.
- `test_manifest_read=false`; `test_used_for_selection=false`.

## 远端启动与首轮预检（2026-09-23 13:56–13:57 Asia/Shanghai）

- 远端 run root 与目标 tmux session 均不存在；源码快照目标为本run隔离的新目录。使用只读检查确认 MTHv2 train/val_tune 为2159/240页、敦煌 train/val_tune为240/80页；两域page_id及`source_group_id` train/val交集均为0。
- 远端关键输入SHA核验通过：epoch-8 head `d56023f270fd51812b58272141f8903bbd2ca1b437b8565b20ab5d12c7f59cac`；checkpoint-3000 decoder LoRA `ada4cdd3bb1f2f417d1c0a68054dc61e70ab59f215b3bc2018074b4f2fdcd5d5`；MTHv2 val_tune `efc29e22b42fb81c8915282f935ef6c77ddcdcb7169f7a48bae6471c6c0cd315`；敦煌 val_tune `e20d2f9b07e535ccfab03c95a8f81222f75ee4f5a793e27dabb28338f48316e2`。源码快照57文件的本地/远端tree fingerprint一致：`9ebd6017f1f7a97071837b010b188689f7bbc8237381d7f15268544b7ada7c85`。远端launcher `bash -n`通过。
- 13:56 tmux session启动，状态结构化文件为`running`，tmux仍存在。raw模型与LoRA fingerprint均已输出，MTHv2图像审计日志刚创建；尚未读取或查询test，尚未查询GPU、启动worker或生成推理结果。数据图像审计结束后，launcher才会检查GPU0–4瞬时利用率并决定是否进入推理。

## 诊断样本锁定与运行进展（2026-09-23 14:01–14:09 Asia/Shanghai）

- 两次健康检查均确认tmux有效、`launcher_status.status=running`、worker输出持续增长且日志无NaN/Inf/CUDA/OOM/Traceback/异常；第二次检查后同一heartbeat已从5分钟更新为30分钟。launcher只在创建时写入`launcher_status.phase=source_snapshot`，内部阶段应根据锁定manifest、结果文件和worker日志判断。
- 首波GPU准入记录为物理GPU0–4瞬时`utilization.gpu=0/0/0/0/0%`。MTHv2 diagnostic32每个密度分位8页，区域数范围依次为10–21、23–24、24–32、43–72；总范围10–72，超过历史sparse24上限的页面12/32。敦煌/地方志每个密度分位8页，区域数范围8–32。抽样manifest SHA：MTHv2 `20221a809232596d04a1c36c28d07adae8151e0d36a6c1dbf021fb50f69b65a9`；敦煌 `82998b196bdb516de9a79d642aa5557430546fe7f7df010808e1e59a14b7cdc8`。
- MTHv2审计：page_id与精确图像跨集交集均为0；33个dHash≤4候选经联系表视觉检查及GT文本相似度复核均为不同页，最高归一化文本相似度0.4779；screen32未命中任何候选。训练集内部存在1组完全相同图像：`mthv2_mth1000_24-V009P0379`与`mthv2_mth1000_26-V009P0379`（文本长度887/889）；此组不跨train/val，后续训练样本/日志需披露。官方拆分的`group_isolation_status=unavailable_official_random_page_split`，逐页`source_group_id(s)`不能代表书籍/版本级来源隔离。
- 敦煌/地方志审计：page_id/source-group无跨集重叠，精确图像重复与dHash近重复候选均为0。第二次健康检查时MTHv2五分片均在生成D0–D3预测，worker进度约2–5/6–7页/臂，错误扫描为空；尚无合并指标，完整val_tune阶段未开始。
- 近重复候选联系表位于本地结果审计目录`D:/yangky/glm-ocr-assets/line-mask-v3/glmocr_line_mask_v3_stage_d_20260923_v1/near-duplicate-review/`；只含train/val_tune图像缩略审计图，不含test数据。

## v1 失败归档（2026-09-23 14:16 起，追加记录，不覆盖上文）

上文第52行“MTHv2 diagnostic32 D0–D3 inference is running”是失败前的状态快照，现按追加更正：**v1 已失败终止，不再运行**。

- **终止状态**：tmux session `glmocr_line_mask_v3_stage_d_20260923_v1` 已不存在；`launcher_status.json` 为 `{"status":"failed","phase":"five_gpu_diagnostic32","exit_code":1}`。
- **失败阶段与原因**：敦煌/地方志 diagnostic32 五个 worker 在首个页面臂上全部抛 `ValueError`，位置为 `build_mask_targets(target_mode="line", line_source="annotation")`（`mask_targets.py` 的 annotation 守卫）。根因是该域 manifest 为 `layout_level=textline`、`bbox_format=xyxy_normalized`，每个 region 带 `text` 与四元素 `bbox`，**没有 `characters` 字段**，因此没有任何 `line_index`。D0（无路由）不构造行目标，故不触发；D1–D3 均失败。MTHv2 的 char manifest 带 `line_index`，不受影响，其五个 shard 的 D0–D3 已全部完成。
- **v1 已产出且保留的产物**：`results/mthv2/shard-{0..4}` 的 D0–D3 `predictions-*.jsonl`、`summary-*.json`、`status-*.json`，页数 7/7/6/6/6（合计32，与锁定抽样一致）；`shard-1`、`shard-3` 下的 `full-mask-traces/`；`manifests/`、`logs/`、`source/` 源码快照与 fingerprint；`base-model-fingerprint.json`、`decoder-lora-fingerprint.json`；`admission.json`（首波物理 GPU0–4 均为0%）。敦煌各 shard 仅有 `worker_protocol.json`，无预测。
- **复用判断**：**不得重启或覆盖 v1**。v1 的 MTHv2 shard 结果不进入 v2 的任何合并或选点：v2 改用新的 `line_evidence` 协议、新的源码快照和新 run ID，merge 身份字段（含 `line_evidence_sha256`、`line_source`、`spatial_targets`、`spatial_target_granularity`）与 v1 不同，v1 产物无法通过 v2 的身份校验。v1 的 MTHv2 局部结果**未完成跨域 merge 与候选选择，不构成完整筛查结论**，只能作为“MTHv2 该抽样下四臂曾跑通”的过程证据，不得据此报告任何 CER 或机制结论。
- **协议字段**：v1 全程 `test_manifest_read=false`、`test_used_for_selection=false`；未读取或查询 test。`val_verify` 未锁定、未读取。
- **后续修正（v2 的前提）**：为无 `characters` 的 textline manifest 增加 `line_source="region_textline"`，按 `reading_order` 拼接 region `text` 并与 `page_text` 逐字核对，用该行 region 的 `bbox` 作为该行所有字符的监督框；无法可靠映射的页面**明确拒绝**，不填猜测标签。同时锁定 `line-evidence.json`，由 evaluator 校验后才允许 D1–D3 运行，避免再次以 `auto` 静默退化为逐字目标。详见[恢复计划](../../../../../plans/LINE_MASK_V3_RECOVERY_PLAN.md)与 `EXPERIMENT_REGISTER.md` 的 v3 条目追加说明。

## v2a 中止（2026-09-23 15:48–15:52 Asia/Shanghai，启动后 4 分钟主动中止）

- **状态**：`glmocr_line_mask_v3_stage_d_20260923_v2` 首次启动，进入 `phase=source_snapshot` 的 MTHv2 图像审计（train 到约 750/2159 页）后被**主动中止**。**`results/` 文件数为 0**，未进入 GPU 准入、未启动任何 worker、未产生任何推理结果或指标。tmux session 已 kill；run 目录与快照原样保留并重命名为 **`..._v2a`**（`code_snapshots/glmocr_line_mask_v3_stage_d_20260923_v2a/` 与 `glmocr_line_mask_v3_stage_d_20260923_v2a/`），未被覆盖。
- **中止原因**：启动后复查新接线代码，发现 `merge_line_mask_v3_diagnostic.py` 中我把每臂的 `spatial_target_granularity` 写成了 Python `set`（`{row.get(...) for row in rows}`），而该值进入 `json.dumps(..., allow_nan=False)`，会抛 `TypeError: Object of type set is not JSON serializable`。该错误只在**推理全部完成后**的 merge 阶段触发，若不予修正，会浪费约 2 小时五卡推理才失败。已修正为写入锁定的字符串 `protocol["line_evidence"]["box_granularity"]`，比较仍用集合，并已用序列化冒烟测试确认（`set` 复现 `TypeError`，修正后通过）。
- **复用判断**：v2a 无任何可复用产物（0 结果文件），仅作为"含 merge 序列化缺陷的快照版本"留痕，不复用、不合并。
- **协议字段**：v2a 全程 `test_manifest_read=false`、`test_used_for_selection=false`；未读取或查询 test。

## v2b 失败（2026-09-23 15:57 Asia/Shanghai）—— 新加的 line-evidence 守卫自身有误

- **状态**：v2（15:53 启动）通过图像审计、写全 6 个 manifest 文件、通过 GPU 准入（`admission.json` = `0:0,1:0,2:0,3:0,4:0`），但在五个 worker **首个页面臂之前**即全部退出：`launcher_status={"status":"failed","phase":"five_gpu_diagnostic32","exit_code":1}`。`results/` 仍为 0 文件——**未做过任何推理**。
- **失败原因**：五个 MTHv2 worker 均在 `diagnose_line_mask_v3.py:353` 抛 `ValueError: line evidence does not cover every locked val_tune page`。这是**本次新加的守卫自身写错了比较对象**：`line_evidence.json` 按该域**整个 val_tune manifest** 计算（MTHv2 240 页、敦煌 80 页），而 diagnostic32 阶段只评估其中的 32 页，守卫却把证据的 `pages_with_line_mapping`（240）与本阶段页数（32）相比，必然不等。两个域都会同时被误拒。证据文件本身是正确的：`pages=240`、`pages_with_line_mapping=240`、`unmappable_pages=[]`。
- **修正**：比较改为"证据覆盖其来源 manifest 且无不可映射页"——`pages == protocol["val_tune_pages"]` 且 `pages_with_line_mapping == pages` 且 `unmappable_pages` 为空。两种阶段（diagnostic32 子集、full_val_tune 全集）均满足该不变式。已加 CPU 回归测试 `test_evidence_covers_the_source_manifest_not_the_stage_subset`（含 32/240 子集反例）。
- **保留**：run 目录与源码快照重命名为 `..._v2_b`（`glmocr_line_mask_v3_stage_d_20260923_v2_b/`、`code_snapshots/glmocr_line_mask_v3_stage_d_20260923_v2_b/`），未覆盖。**复用判断**：无可复用推理产物；其 `manifests/*/line-evidence.json` 与 `protocol.json` 信息正确，可作为证据格式的对照留痕。
- **协议字段**：`test_manifest_read=false`、`test_used_for_selection=false`；未读取 test。

## v3b 中止（2026-09-23 16:37 Asia/Shanghai）—— 诊断器的逐 token 解码有误，mask 质量指标全为空

- **中止时机**：v3 已跑到 **40/40 个臂级 summary、9/10 个 worker_status**（仅敦煌 shard-0 未收尾），`launcher_status` 仍为 `running`，无任何报错。**在它进入完整 val_tune 阶段之前主动中止**——若继续，会在诊断指标已失效的前提下再烧约 2 小时五卡。
- **发现方式**：读取已完成的敦煌 `predictions-D1.jsonl` 时注意到 `alignment.alignment_coverage = 0.0`、`reliable_spatial_character_count = 0`。逐域逐臂复核后确认 **8/8（2 域 × 4 臂）全部为 0**，即没有任何一个位置产生过 line-IoU——而 mask 定位质量正是 Stage D 的主要产出。
- **根因**：`diagnose_line_mask_v3._decode_piece` 用 `tokenizer.decode([单个 token id])` 逐 token 解码。GLM-OCR 是**字节级 BPE**：一个汉字跨多个 token，单独解码其中之一得到的是 `�`(U+FFFD) 而非该字的首字节。于是 `"".join(pieces)` 永远不等于 `prediction`，`token_character_mapping_reliable` 恒为 False，`normalized_token_ids` 全置 None，所有 token 的对齐统计与 `line_quality` 随之全空。实测每页约 95 个 `�`。**该缺陷在 v1 的同一份代码中即已存在**，并非本轮新增；v1 的 MTHv2 局部结果同样受此影响。
- **为何这不是数据问题**：`full decode == prediction` 为 True，行级目标本身正确（敦煌该页 12 行、`region_line_report.mapped_characters=190`、`window_fallbacks=0`、`token_fallbacks=0`）。坏的只是 token→字符的归属分解。
- **修正**：新增 `decode_pieces`，对每个前缀做 decode 并与完整 decode 取**最长公共前缀**，输出 `full[cursor:k]`。这样逐段拼接在构造上恰好等于完整 decode，且把"由多个 token 拼成的字符"归属于**完成它的那个 token**。中途试过的"前缀差集"写法同样错误（前缀停在半个字符时解码器输出 U+FFFD，补全后该占位符被**替换**而非追加，差集因此丢字），已在函数文档中记录以免回退。
- **验证（真实模型 tokenizer + v3b 真实产物，未用 GPU）**：旧法 0/256 页可对齐；新法 **128/128 MTHv2 + 128/128 敦煌 = 256/256（100%）**。残留 5 处 U+FFFD 为真实非 BMP 字形在两条路径中同样出现的替换符，因两侧一致故不影响判等。空 token 列表返回空列表。
- **新增回归测试** `tests/test_line_mask_v3_decode_pieces.py`（5 项）：用字节级 tokenizer 桩件钉住"逐段拼接等于完整解码"、每字符归属完成它的 token、`�` 不得进入任何分段，并显式记录旧法失败。
- **保留**：run 与快照重命名为 `..._v3_b`（`glmocr_line_mask_v3_stage_d_20260923_v3_b/`、`code_snapshots/..._v3_b/`）。其 **40 个臂级 summary 与全部 predictions 均保留**，CER/EOS/触顶/循环等**不依赖 token 对齐**的字段仍然有效，可作为"该抽样下四臂 CER 与生成行为"的过程证据；但其中 `alignment_coverage`、`reliable_spatial_positions`、`mean_continuous_line_iou` 等**空间指标一律无效，不得引用**。
- **协议字段**：`test_manifest_read=false`、`test_used_for_selection=false`；未读取 test。

## v3 启动（2026-09-23 16:02 Asia/Shanghai）

- **Run ID**：`glmocr_line_mask_v3_stage_d_20260923_v3`；tmux 同名 session；Slurm 不适用。**v1、v2a、v2b 三个目录均未被触碰、未被覆盖。**
- **不可变源码快照**：`.../line_mask_v3/code_snapshots/glmocr_line_mask_v3_stage_d_20260923_v3/ocrmodel`（57 文件）。快照内已核实三项修正均在位：`region_textline`（mask_targets）、`val_tune_pages` 守卫（diagnose）、`row_granularity` 已移除（merge）；`bash -n` 通过。run 内快照 fingerprint 由 launcher 启动后计算并追加。
- **启动前 GPU 准入**：物理 GPU0–4 瞬时均为 `0%`。
- **累计修正清单（相对 v1）**：① 新增 `line_source="region_textline"` 行级支持 + 锁定 `line-evidence.json`；② `line_source="auto"` 不再静默退化为逐 token 目标；③ 单框窗口凸包退化导致栅格全零已修（全部四角入包）；④ merge 的 `set` 序列化缺陷已修；⑤ line-evidence 覆盖性守卫比较对象已修。
- **关键里程碑（16:06，v1 与 v2b 的死亡点）**：两域 manifest 均锁定——敦煌/地方志以 `region_textline`/`line_regions` 通过（`diagnostic_pages=32`，抽样 manifest SHA `82998b196bdb516de9a79d642aa5557430546fe7f7df010808e1e59a14b7cdc8`，与 v1 相同）；MTHv2 以 `annotation`/`character_boxes` 通过（SHA `20221a809232596d04a1c36c28d07adae8151e0d36a6c1dbf021fb50f69b65a9`，与 v1 相同）。GPU 准入 `admission.json`=`0:0,1:0,2:0,3:0,4:0`。五个 MTHv2 worker 全部**越过 v1/v2b 的失败点**并开始加载权重，错误扫描为空。v3 已解决的正是两轮失败各自的直接原因。
- **启动后另行发现并修正的两处同类缺陷（已写入工作区，将在下一次快照生效）**：⑥ full_val_tune 的 evaluator/merge 会把证据文件的 `status` 与固定的 `locked_before_candidate_inference` 比对，而 full_val_tune 的 prepare 写入的是 `locked_after_32_page_screen_before_full_inference`——若不修，32 页筛查通过后会在完整 240/80 页阶段整体失败；现改为按 `expected_status` 逐阶段比对，并额外校验 `evaluation_stage`。⑦ 增加"本阶段页数不得超过证据覆盖页数"的边界检查。与 ⑤ 同类：均为新校验的比较基准写错，都是**在浪费 GPU 之前**由代码复查或（⑥）离线复算发现。注意 v3 的不可变快照含 ①②③④⑤，**不含 ⑥⑦**，故完整 val_tune 阶段届时需另起新快照与 run ID；本阶段 32 页筛查不受影响。

## v2b 启动记录（2026-09-23 15:53–15:57 Asia/Shanghai）

- **Run ID**：`glmocr_line_mask_v3_stage_d_20260923_v2`（v2a 中止后重建，run 目录与快照均为新目录）；tmux 同名 session；Slurm 不适用。**v1 目录未被触碰、未被覆盖**。
- **不可变源码快照**：`.../line_mask_v3/code_snapshots/glmocr_line_mask_v3_stage_d_20260923_v2/ocrmodel`（57 文件）。launcher 自行解包并计算 run 内快照 fingerprint，启动后追加。与 v1 的 `9ebd6017…ada7c85` 不同，符合"新增 region-textline 与 line-evidence 校验、修复 merge 序列化与单框凸包退化"的预期差异；相对 v2a 亦不同（含 merge 修正，快照内 `grep -c row_granularity` 为 0，`locked_granularity` 在位）。远端 `bash -n` 通过，脚本为 LF 行尾。
- **模型与 LoRA（与 v1 同起点，已核定）**：raw GLM-OCR revision `ca5d8b3e287e52589e37c28385d9655ee4372f9d`，aggregate SHA `cde6532e869801998c91ea7ac209f349e281d89859d0ed3eba59066a27f1eabe`（1 个权重文件）；冻结 decoder LoRA `ada4cdd3bb1f2f417d1c0a68054dc61e70ab59f215b3bc2018074b4f2fdcd5d5`；冻结 head epoch8/step3456 `d56023f270fd51812b58272141f8903bbd2ca1b437b8565b20ab5d12c7f59cac`。三者均与 v1 及预注册一致。
- **启动前 GPU 准入**：物理 GPU0–4 瞬时 `utilization.gpu` 均为 `0%`（v2a、v2b、v3 三次启动均为 0/0/0/0/0%，由 launcher 自身的 `gpu_admission` 阶段再次查询并记录到 `admission.json`）。
- **首轮健康检查（v2a，15:48:54）**：tmux 存活；`launcher_status={"status":"running","phase":"source_snapshot"}`；`base-model-fingerprint.json`、`decoder-lora-fingerprint.json`、`model-fingerprints.log` 已写出且数值与 v1 相同（raw aggregate `cde6532e…1eabe`，LoRA `ada4cdd3…cd5d5`）；MTHv2 图像审计进行中。NaN/Inf/CUDA/OOM/Traceback/Error 扫描为空。

## v4 启动（2026-09-23 16:44 Asia/Shanghai）

- **Run ID**：`glmocr_line_mask_v3_stage_d_20260923_v4`。**v1、v2a、v2b、v3b 四个目录全部保留未覆盖。**
- **不可变源码快照**：`.../code_snapshots/glmocr_line_mask_v3_stage_d_20260923_v4/ocrmodel`，run 内 fingerprint 启动后追加。
- **本快照相对 v1 的全部修正（7 项）**：① `line_source="region_textline"` 行级支持；② `line_source="auto"` 不再静默退化；③ 单框窗口凸包退化致栅格全零；④ merge 的 `set` 序列化；⑤ line-evidence 覆盖性守卫比较对象；⑥ full_val_tune 证据状态按阶段比对 + 页数边界；⑦ **逐 token 解码改为最长公共前缀法**（使 `alignment_coverage` 与 line-IoU 真正可用，这是 Stage D 判据的载体）。
- **启动前 GPU 准入**：物理 GPU0–4 瞬时利用率均为 0%。
- **相对 v3b 的差异**：仅 ⑥⑦ 两处（其余 ①–⑤ 已在 v3b 快照内）。①–⑤ 已被 v3b 证明可跑通（两域 manifest 均锁定、十片 worker 全部越过 v1/v2b 失败点、40/40 臂级 summary 完成、全程无 Traceback/NaN/OOM）。

## v4 失败于 merge，但十片 worker 全部完成（2026-09-23 17:19 Asia/Shanghai）

- **状态**：`launcher_status={"status":"failed","phase":"merge_and_preregistered_screen","exit_code":1}`。**两域十片（5 shard × 2 domain）worker 全部 `worker_status=complete`，40/40 个臂级 summary 写出，错误扫描为空**——失败发生在推理**之后**的 merge 阶段，GPU 未被浪费。
- **失败原因**：`merge_line_mask_v3_diagnostic.py:139` 抛 `RuntimeError: shard 0 is incomplete`。根因是 merge 自身的一处预存缺陷（此前从未被执行到，v1 未走到 merge）：该行把**单个分片**实际处理的页数与**整个 32 页集合**比较——`status["pages"] != len(expected_ids)`（如 7 != 32），故第一个分片即被误判。worker 本身无任何异常。
- **修正（已写入工作区，未用于本次结果的合并）**：改为与 `expected_ids[shard_index::5]` 比较，即该分片自身的分配页数。
- **结果获取方式与边界（重要）**：因 merge 未通过，下节数字由我**直接读取 40 个臂级 summary 与全部 predictions 汇总**得到，**未经预注册的 merge 流程**（该流程会校验五片身份字段一致、page_id 无漏无重、reference 一致，并产出 `summary.json`）。因此这些数字属**过程性汇总**，可作机制判读依据，但在 merge 通过前**不得登记为"筛查结论"或用于正式候选选择**。
- **协议字段**：`test_manifest_read=false`、`test_used_for_selection=false`；未读取 test；`val_verify` 未读取。

## v4 diagnostic32 结果（过程性汇总；32 页/臂 × 2 域，val_tune）

推理协议与预注册一致：seed 42、整页图像 + `Text Recognition:`、fast processor、max_pixels 4,000,000、greedy、1536 上限、BF16 backbone / FP32 head、`torch_sdpa_math`、`decoder_layers=16`、`mask_threshold=0.5`。冻结起点：raw aggregate `cde6532e…1eabe`、decoder LoRA `ada4cdd3…cd5d5`、head epoch8/step3456 `d56023f2…9cac`。分片耗时 601–1062 s/域。

### MTHv2（32 页，参考字符 12,634）

| arm | CER | I / D / S | vs D0 | vs D1 (相对) | 可靠对齐点 | coverage | line IoU |
|---|---:|---|---:|---:|---:|---:|---:|
| D0 无路由 | 0.318031 | 1344 / 710 / 1964 | — | — | 0 | n/a | n/a |
| D1 全层 bias 1.0 | 0.420294 | 2208 / 523 / 2579 | +0.102264 | — | 8346 | 0.7125 | 0.33730 |
| D2 bias 0.5 | **0.313519** | 1174 / 741 / 2046 | **−0.004512** | −0.254048 | 8606 | 0.7448 | 0.33928 |
| D3 后半层 | 0.384835 | 2098 / 1082 / 1682 | +0.066804 | −0.084367 | 8507 | 0.7306 | 0.33089 |

### 敦煌／地方志（32 页，参考字符 10,574）

| arm | CER | I / D / S | vs D0 | vs D1 (相对) | 可靠对齐点 | coverage | line IoU |
|---|---:|---|---:|---:|---:|---:|---:|
| D0 无路由 | 0.111973 | 190 / 184 / 810 | — | — | 0 | n/a | n/a |
| D1 全层 bias 1.0 | 0.118593 | 284 / 157 / 813 | +0.006620 | — | 8792 | 0.8052 | 0.39933 |
| D2 bias 0.5 | **0.111500** | 191 / 183 / 805 | **−0.000473** | −0.059810 | 8772 | 0.8108 | 0.40281 |
| D3 后半层 | 0.113391 | 206 / 183 / 810 | +0.001418 | −0.043864 | 8763 | 0.8086 | 0.39853 |

### 生成行为（两域各 32 页）

| arm | MTHv2 EOS / 触顶 / 循环 | 敦煌 EOS / 触顶 / 循环 |
|---|---|---|
| D0 | 30 / 2 / 2 | 32 / 0 / 0 |
| D1 | 29 / 3 / 3 | 32 / 0 / 0 |
| D2 | 30 / 2 / 2 | 32 / 0 / 0 |
| D3 | 30 / 2 / 2 | 32 / 0 / 0 |

### 可读出的判读（过程性，非筛查结论）

1. **D1（历史全强度路由）在两域都退化，且以插入为主**：MTHv2 CER +0.102、插入 1344→2208；敦煌 +0.0066、插入 190→284。与既有 v2 记录"收益/退化主要走 I/D 而非替换"的模式一致，提示固定全层 bias=1.0 导致**多吐字**而非认错字。
2. **D2（bias 减半）是唯一两域都不劣于 D0 的臂**（MTHv2 −0.0045、敦煌 −0.0005）；D3 在 MTHv2 明显劣于 D0（+0.0668）。
3. **按预注册候选规则，D2 与 D3 均合格**（两域 CER 均不高于 D1）；按"等域平均相对 CER 变化最小、并列优先 D2"排序，**D2 以 −0.1569 胜出**（D3 为 −0.0641）。——此为规则算术结果，**在 merge 正式通过前不构成锁定的候选选择**。
4. **mask 定位本身有效且偏离 D0 预期**：D0 无路由故无对齐点与 IoU；D1–D3 覆盖率 0.71–0.81、行级 IoU 0.33–0.40。**这是"逐 token 解码"修正生效的直接证据**——同一份 harness 在 v3b 时这些指标全为 0。
5. **granularity 与预注册一致**：MTHv2 `character_boxes`/`character`，敦煌 `line_regions`/`textline`（`pages_with_unavailable_line_mapping=0`）。**敦煌的 IoU 是行级、无行内几何，不得与 MTHv2 的字符级 IoU 直接比较。**
6. 敦煌四臂的 CER 差都在 ±0.007 内、参考字符仅 10,574，**该域 32 页不足以支撑任何方向的结论**；MTHv2 的 D1/D3 退化幅度远超噪声，但这仍只是 32 页筛查。

### 尚未完成 / 不得宣称

- 未跑完整 240 / 80 页 val_tune；未做跨域正式 merge 与候选锁定；**没有独立 val_verify**；结果仍全部属 development / val_tune。
- 未做来源分组 bootstrap；无显著性声明。32 页筛查按预注册只用于淘汰明显失败项。
- `test` 全程未读，`test_manifest_read=false`、`test_used_for_selection=false`。
