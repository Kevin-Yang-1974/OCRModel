# 敦煌／地方志 GT行mask Oracle：典型错误注意力回放

## 标识与协议

- Run ID：`glmocr_dunhuang_gtline_attention_diag_20260923`；日期：2026-09-23；短时事后推理诊断，launcher complete、exit 0；无 Slurm、无 tmux。
- 源分支/commit：`glm-ocr-layout-mask-routing` / `ea73f182bc7ebe69f7c7386024b3b97bf89e392a`。沿用不可变代码快照 `/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/code/line_mask_20260922_v4/ocrmodel`；没有改动计划代码或训练代码。
- 入口脚本（run-owned）：`D:/yangky/glm-ocr-assets/dunhuang-new77-20260923/oracle-gtline-rawglm-b1/attention-diagnostic-20260923/capture_gt_oracle_attention.py`；脚本 SHA256 `c68788201674f5177c71100fd336af0a0451200d23890593b9d0766f953c3dee`。渲染器为同目录 `render_gt_oracle_attention.py`。
- 远端产物：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/diagnostics/glmocr_dunhuang_gtline_attention_diag_20260923`；本地：`D:/yangky/glm-ocr-assets/dunhuang-new77-20260923/oracle-gtline-rawglm-b1/attention-diagnostic-20260923`。
- 数据为之前全量59页 GT Oracle locked-test 中事后选出的4页；新建4页manifest SHA256 `61fdee622074d2b88de5c63287e546ba3fe948a671320173e95739b843cdf4c2`。覆盖2个新增插入（简单稀疏1、高复杂度1）、1个修正替换（高复杂度）、1个修正漏字（中等）；这是错误类型可视化案例，不是独立抽样评测。seed42。Validation不适用。
- 协议：`test_manifest_read=true`；`test_used_for_selection=false`；test历史已暴露；只事后解释，不据此调bias、阈值、后处理或选权重。

## 权重、组别与执行

- Baseline与Oracle都重用原始GLM-OCR revision `ca5d8b3e287e52589e37c28385d9655ee4372f9d`；config SHA256 `4e1daf0d8a3f63e58960ac14bcb58b7be96758cad231fb7a1e5fec60f42dcd8c`，raw `model.safetensors` SHA256 `a16eb0de98d199293371c560f95f83130d2a2c9612449df16839f08ff9498815`。无LoRA、无微调或layout checkpoint，可训练参数0。
- Baseline关闭路由；Oracle仅新增GT textline bbox、GT转写同步pointer及bias=1.0。全页输入、Text Recognition:、fast processor、4M像素、1536 max tokens、greedy、BF16、use_cache、SDPA math均与已完成原run一致；两臂配对、相同4页、相同预算。训练/优化器/LR/schedule/loss/训练步数/DDP均不适用。
- 物理GPU0，启动前 utilization 0%；启动后单卡完成。未查询GPU0–4之外设备。远端临时目录仅用run内 `/data3/.../tmp`。

## 结果

原59页全量run（见同名 rawglm-b1 实验日志）：baseline CER 0.1368791828，1,943 edits，I/D/S 519/404/1,020；Oracle CER 0.1383585770，1,964 edits，I/D/S 548/398/1,018。该全量点估计 Oracle 多21 edits；按域分层 page bootstrap 95% CI跨0。该诊断没有重算全量 CER。

| 事后案例 | Baseline → Oracle I/D/S | 页级 edits变化 | GT行框内视觉条件注意力 | 全部视觉token注意力质量 |
| --- | --- | ---: | ---: | ---: |
| `oracle-added-insertion-dunhuang` | 4 / 1 / 23 → 21 / 1 / 24 | 62.43% → 80.62% | 41.28% → 44.68% |
| `oracle-added-insertion-gazetteer` | 13 / 14 / 20 → 34 / 14 / 22 | 40.74% → 61.09% | 26.81% → 29.22% |
| `oracle-fixed-substitution` | 10 / 11 / 40 → 10 / 11 / 25 | 42.44% → 58.56% | 34.51% → 14.79% |
| `oracle-fixed-deletion` | 17 / 2 / 22 → 17 / 0 / 23 | 69.38% → 48.51% | 27.26% → 7.64% |

采集的是每个目标输出字符所对应生成step的最终decoder layer、head平均注意力。图中 patch 概率只在视觉键内重新归一化；同时记录非条件化的视觉总量。每个示例记录各自生成step并使用其真实的 applied line bbox。Prefill最后query预测首个输出字符；其后每个cached decode query对应下一输出token。四页 baseline 与 Oracle 输出均逐字重现归档（4/4 pages、8/8 arms），捕获token/字符映射和行框mask核验通过；所有attention数组有限。

案例层面，增加目标行注意力与字符变准不等价：两例插入错误变多时行内质量仍上升；漏字得到修正的案例行内质量反而下降。因此该图显示Oracle bias改变空间证据分配，但不能把“注意力更聚焦”直接解释为识别改善。GT框/同步文本是oracle机制诊断，不代表可部署路由。无IoU/MAE（该run不训练或预测mask）；loss曲线不适用。没有新checkpoint；原始权重加载成功。日志错误扫描无NaN/Inf、CUDA/OOM、Traceback、NCCL或异常退出。

## 文件与校验

- 汇总图：`D:/yangky/glm-ocr-assets/dunhuang-new77-20260923/oracle-gtline-rawglm-b1/attention-diagnostic-20260923/gt-oracle-typical-error-attention.png`
- 指标JSON：同目录 `attention_metrics.json`；capture summary `captures/capture_summary.json`。
- attention脚本SHA256：`c68788201674f5177c71100fd336af0a0451200d23890593b9d0766f953c3dee`；4页manifest SHA256：`61fdee622074d2b88de5c63287e546ba3fe948a671320173e95739b843cdf4c2`。
- `launcher_status.json=status complete`；远端capture log已归档；四份NPZ和四张原图已下载。
