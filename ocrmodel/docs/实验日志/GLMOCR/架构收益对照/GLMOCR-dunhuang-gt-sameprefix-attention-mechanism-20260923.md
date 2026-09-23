# GT 强制前缀下的行 mask 注意力注入复核

**结论：注入代码路径确实在目标 query 上把注意力推向指定 GT 行。** 四个事后选出的典型错例在 16 个 decoder layer 中全部出现同方向变化；逐头的“GT 行 / 其他 key”注意力赔率，加入 bias=1.0 后乘数与 e¹=2.71828 一致，最坏相对误差 9.56e-6。这验证了注意力 mask 的局部注入机制，不代表字符识别因此变好，也没有测量开启 bias 前后的最终 token logit 或 CER。

## 实验标识与协议

| 字段 | 内容 |
| --- | --- |
| Run ID / 状态 | glmocr_dunhuang_gtline_sameprefix_mechanism_20260923 首轮失败于诊断器的半 token 前缀判定（2/4 案例已写结果）；修订为 glmocr_dunhuang_gtline_sameprefix_mechanism_v2_20260923，完成 4/4 |
| 日期 / 分支 / commit | 2026-09-23；glm-ocr-layout-mask-routing；ea73f182bc7ebe69f7c7386024b3b97bf89e392a |
| 代码来源 | 不可变快照 code/line_mask_20260922_v4/ocrmodel，source tar SHA256 838c06d66617a001210731043ac735dc5d240cb6cc46148168c3fb8841ff4c26；远端实际 attention_routing.py SHA256 7aa073b628154f62555e5cab1e0cb69e082f4d6552ecf07217c9dc22b3db8a81 |
| 入口 / 产物 | 本地脚本 [run_gt_forced_sameprefix.py](D:/yangky/glm-ocr-assets/dunhuang-new77-20260923/oracle-gtline-rawglm-b1/attention-diagnostic-20260923/gt-forced-mechanism/v2/run_gt_forced_sameprefix.py)；远端 /data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/diagnostics/glmocr_dunhuang_gtline_sameprefix_mechanism_v2_20260923；无 Slurm、无 tmux |
| 数据 | 已历史暴露的 dunhuang_local_gazetteer_q32_v1 test 59 页池中，按插入、替换、删除错误类型事后抽取 4 页（敦煌 2、地方志 2）；此 4 页 manifest SHA256 61fdee622074d2b88de5c63287e546ba3fe948a671320173e95739b843cdf4c2。此为机制图示样本，不是有代表性的统计抽样。seed 42 |
| 模型 / 训练 | 原始 GLM-OCR revision ca5d8b3e287e52589e37c28385d9655ee4372f9d；无 LoRA、无微调权重；训练关闭，可训练参数 0。LR、warmup、schedule、总步数、loss、梯度累积、effective batch 均不适用 |
| 推理设置 | 整页、Text Recognition prompt、4M 像素上限、BF16、SDPA；强制 GT token 到目标 token（0-based 步为 2/238/51/77），目标 query 才开一次 bias=1.0，其余 query 均关闭；物理 GPU0 单卡、batch 1、DDP=false |
| 协议字段 | test_manifest_read=true；test_used_for_selection=false；test 数据历史已暴露。本诊断只作事后机制解释，不选参数、checkpoint 或后处理 |

## 同状态注意力测量

先用 GT token 固定目标字符之前的生成历史，且此前所有 query 都关闭 bias。在产生目标 token 的 cached query 上，对每层读取真实运行中的 Q/K/V 和 attention mask；“bias on”用该层实际 mask 计算，“bias off”从同一个 mask 中移除当层的 GT 行 bias，再用完全相同的该层 Q/K 计算 softmax。因此每层比较不受两个自由生成分支在此前错误后走向不同上下文的干扰。深层 query 本身仍包含同一目标 forward 中较浅层已加 bias 的影响；这里验证的是逐层 mask 的直接作用，不是两个完整模型 forward 的最终 logits 对照。

v2 按实际 target-token decode step 触发，并把该一步的路由索引显式设到目标字符的 GT 索引，以隔离 mask 几何与 pointer 漂移。触发前自然同步指针的位置及其行号也一并记录：四例都落在正确目标行，第四例字符指针为 71、GT 字符索引为 72，但两者仍在同一第 4 行。四个注入 mask 均与 GT bbox 的 patch-center 栅格化结果、此前 Oracle capture 的 mask 完全一致。

## 结果

“行质量”是所有 key 上的注意力概率总和中，落在该 GT 行视觉 patch 的部分；“视觉总量”是全部视觉 patch 的总概率。热图使用视觉 patch 内条件归一化，并在图内同时标出上述总体行质量与视觉总量。

| 案例 | GT 字符 / 索引 / 行 | 行 patch / 全视觉 patch | 行质量 bias off → +1 | 视觉总量 bias off → +1 | 自然指针位置 / 行 |
| --- | --- | ---: | --- | --- | --- |
| 敦煌插入错例 | U+4EBA / 2 / 0 | 188 / 2494 (7.54%) | 22.13% → 35.91% (+13.78 pp) | 35.37% → 44.57% (+9.19 pp) | 2 / 0 |
| 地方志插入错例 | U+4E2D / 203 / 17 | 31 / 2520 (1.23%) | 10.72% → 21.02% (+10.30 pp) | 27.04% → 33.82% (+6.78 pp) | 203 / 17 |
| 替换修正错例 | U+7232 / 42 / 6 | 64 / 2508 (2.55%) | 3.84% → 9.37% (+5.53 pp) | 11.07% → 15.99% (+4.92 pp) | 42 / 6 |
| 删除修正错例 | U+616E / 72 / 4 | 76 / 2494 (3.05%) | 1.55% → 4.01% (+2.45 pp) | 5.95% → 8.22% (+2.27 pp) | 71 / 4 |

对每层、每个 head，odds(line vs. every other key) 的 post/pre 比值理论上恰为 exp(1)，实测跨 4 案例×16 层的最大相对误差为 9.56e-6。attention map 全部 finite，4/4 GT mask 几何核验通过；v2 launcher exit 0，NaN/Inf、OOM、Traceback、NCCL、CUDA error 扫描为空。

首轮 v1 不是模型或 GPU 失败：第三例的 byte-level tokenizer 半 token 解码不能形成稳定的字符前缀，旧诊断器错误地将其判为 alignment ambiguous。已保留 v1 的失败状态和部分输出；v2 不再对不稳定的半 token 解码做字符前缀断言，而直接按 target token step 触发。

## 可视化与解释边界

- [四例 GT 前缀同状态注意力图](D:/yangky/glm-ocr-assets/dunhuang-new77-20260923/oracle-gtline-rawglm-b1/attention-diagnostic-20260923/gt-forced-mechanism/v2/sameprefix_gt_attention_grid.png)
- [16 层目标行注意力质量](D:/yangky/glm-ocr-assets/dunhuang-new77-20260923/oracle-gtline-rawglm-b1/attention-diagnostic-20260923/gt-forced-mechanism/v2/sameprefix_line_mass_by_layer.png)
- [逐层赔率理论值误差](D:/yangky/glm-ocr-assets/dunhuang-new77-20260923/oracle-gtline-rawglm-b1/attention-diagnostic-20260923/gt-forced-mechanism/v2/sameprefix_odds_error_by_layer.png)
- 指标、有限性与 mask 几何核验：[verification_summary.json](D:/yangky/glm-ocr-assets/dunhuang-new77-20260923/oracle-gtline-rawglm-b1/attention-diagnostic-20260923/gt-forced-mechanism/v2/verification_summary.json)；[逐例 CSV](D:/yangky/glm-ocr-assets/dunhuang-new77-20260923/oracle-gtline-rawglm-b1/attention-diagnostic-20260923/gt-forced-mechanism/v2/case_metrics.csv)

这次实验终于把“注入是否生效”回答清楚了：生效，而且确实把概率从行外移向行内；注意力并未被硬裁剪，bias 同时提高了总体视觉 token 占比。它不回答目标字的 logit 有没有因此变好。之前完整 59 页自由生成 Oracle 的 CER 为 0.1383586，baseline 为 0.1368792，Oracle 多 21 个编辑，见[原始 Oracle 实验日志](GLMOCR-dunhuang-oracle-gtline-rawglm-b1-20260923.md)。所以不能把“行注意力上升”当作“识别受益”。

下一步应在固定 GT 前缀下，对相同目标 query 分别完整运行 bias-off 与 bias-on，比较 GT 下一 token 的 log-prob、rank 和 top-1 是否翻转；这才判断空间注入是否把正确字变得更可预测。当前四页来自历史 test，只能作为事后机制样本，不能据此调 bias。若该信号值得继续，需在可调集合上完成参数比较，再用与所有调参页不交叉、且此前未暴露的独立 val_verify 验收；不能把旧 149 页重新命名为独立验证。
