# 偏置校正的置信度：149 页实测

**日期**：2026-09-21
**前序**：`docs/LAYOUT_TRACKING_OFFLINE_DIAGNOSIS.md`（离线重放预测 +1.1 点定位）、
`docs/LAYOUT_LINE_DETECTOR_AND_PREDMAP_RESULT.md` §8（污染实测）、§9.1（门槛与剂量耦合）
**运行**：`glmocr_layout_corr_pilot_20260921_v1`（名字里的 `pilot` 是历史遗留：
原计划先在 28 页子集上试接线，但 `GLMOCR_MTHV2_SPARSE_Q32_ROOT` 当时没被 tmux 重启重新导出，
覆盖被静默丢掉，实际跑的是全 149 页。已在 launcher 里修，记录按全量读。）

**状态**：**运行中**。本文件先落判据，数值待填。

---

## 一、设置

| 项 | 值 |
|---|---|
| checkpoint | `glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000` |
| 数据 | MTHv2 sparse24 **validation 全 149 页**（不读 test） |
| 分辨率 | `max_pixels=4000000` |
| 探针 | 第 8 层 × 头 `2,3,8,10,11,12,14,15`（阶段 0 登记配置） |
| 臂 | `noroute` / `line100` / `raw6` / `corr05` / `corr1` / `corr2` |

`corr*` 与 `raw6` 的唯一差别是**校正开关**：`corr*` 把路由加在读数上的偏置除掉再送进门控，
`raw6` 不除（现行为）。门槛分别 0.5 / 1.0 / 2.0 与 6.0，所以这一轮同时也在扫「校正后的置信度
该配什么门槛」。

**接线已核**：探针报告的 `bias_corrected_steps` 在 `corr*` 上是 235/254（首两页），
在 `raw6` 上是 0；`corr*` 的报告带 `top_line_mass_raw` 而 `raw6` 不带。
launcher 的 summarize 已把这条写成硬判据。

## 二、判据（预登记，跑之前写下）

- `noroute` 应复现 **0.169924**、`line100` 应复现 **0.124694**；不复现则装置不可比，先修。
- **主比较**：`corr*` 对 `raw6`，CI 下界 > 0 才算成立。离线预测 ~+1.1 点定位、~+0.8 点 CER。
- 若 `corr*` 全部不优于 `raw6`，则离线那条定位改善**没有传导到 CER**，
  这是一个与离线结论相反的负结果，必须原样记录，不能用定位指标替代。
- 任何新臂 CER 高于 `raw6` 都要明说更差，无论显著性。

## 三、结果

（待填：CER / S-I-D / 触顶 / 逐页配对 bootstrap，按页与按卷号分组两档）

## 四、依据

1. 工具：`tools/replay_tracker_offline.py`、`tools/analyze_cer_significance.py`。
2. 实现：`src/layout_ocr/attention_probe.py`（`_bias_added_per_token`、逐头校正）、
   `src/layout_ocr/train_screen.py`（`--layout-tracking-corrected-confidence`）。
3. 离线预测：`docs/LAYOUT_TRACKING_OFFLINE_DIAGNOSIS.md` §1。
