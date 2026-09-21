# win50_g4 锁定 test 的预登记确认

**日期**：2026-09-21
**状态**：预登记（先于 test 数据提交；结果另文）
**前序**：`docs/LAYOUT_TRACKING_WINSHARE_RESULT.md`（§3.1 把这条确认列为方向①）

## 要确认的是什么

`win50_g4`（跟踪行 + 下一行份额 0.5 + 门控 4.0，真值行框地图）在 149 页 validation 上把 CER 从
0.139722 降到 **0.133313**（删除 771→552），且按页/按卷两种配对 bootstrap 都不含零。
它是 5 臂网格（份额 0.25/0.5/0.75/1.0 × 门控 4/6/8）里选出来的，区间偏乐观。
确认的必要性、以及「换 checkpoint 不可得」的说明见 winshare 结果 §3.1——该 source run 只有
`seed42/checkpoint-3000` 一个 checkpoint，确定性重跑不是独立复现，所以唯一有效的确认是上锁定 test。

## 预登记口径（锁定，不在 test 上改）

| 项 | 取值 |
|---|---|
| 页集 | MTHv2 sparse24 锁定 test，**509 页**，`split=test` |
| 分辨率 | 4M、`num_queries=32`、`max_eval_new_tokens=1536` —— 与 validation 上 0.133313 的记录同分辨率 |
| checkpoint | `...boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000` |
| 协议 | `...boxeq820_3000_from_boxeq58_a100_260916_v1.test_locked.json`（`test_manifest_read=true`，`test_used_for_selection=false`） |
| 臂 | `noroute`（无路由）、`raw6`（跟踪行、门控 6、无份额）、`win50_g4`（跟踪行、门控 4、份额 0.5） |

**主比较**：`win50_g4` vs `noroute` —— 路由干预的整体效果。
**次比较**：`win50_g4` vs `raw6` —— `raw6` 与 `win50_g4` 只差「份额 0→0.5、门控 6→4」，隔离边际效果。

**判定口径**：CER 点估计 + 按页/按卷两档配对 bootstrap（10000 次，seed 0），
区间方向与项目文档一致（正 = 臂更好）；报告最坏情况下界，不做 p 阈值二值化、不调阈值、不做后处理。

## 边界

- 地图仍是**真值行框** ⇒ 即便确认，也只是**诊断臂**的确认，不改变「可部署形态（预测地图）仍无改善」。
- `raw6` 本身（跟踪行、无份额）相对 `noroute` 的差异不是本轮问题；本轮只看它作为 `win50_g4` 的对照。
- test 不参与任何选点（AGENTS.md 第 10 条）；本次不读 validation 之外任何用于调参的信息。

## 执行

- launcher：`tools/training/run_glmocr_layout_oracle_test_confirm_a100.sh`
- 实现：`train_screen.py` 新增 `--eval-split test` / `--test-manifest`，eval-only 时以 test 记录计分，
  预测写 `test_predictions.jsonl`、summary 键为 `test`；`test_manifest_read` 从协议读、`test_used_for_selection=false`。
- 产物：`oracle_line_eval/glmocr_layout_test_confirm_20260921_v1/`，汇总 `test_confirm_summary.json`。
