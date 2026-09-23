# 行级 mask epoch8 与无路由 baseline：MTHv2 full locked test

## 预注册协议与启动前核验

- **run ID**：`glmocr_line_mask_v2_full_20260922_1940_test_20260923`
- **状态**：协议已锁定，五卡正式评估待启动/进行中；本节在推理前记录。
- **日期 / 分支 / commit**：2026-09-23；`glm-ocr-layout-mask-routing`；当前 HEAD `452ea59d9c91f7ca8fbc093b51a9ce15f79f9d18`。本轮新增评测脚本位于工作树，未提交。
- **入口**：`ocrmodel/tools/evaluation/run_line_mask_locked_test_a100.sh` → `evaluate_line_mask_locked_test.py` → `merge_line_mask_locked_test.py`。
- **远端产物根**：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/locked_tests/glmocr_line_mask_v2_full_20260922_1940_test_20260923`。
- **会话**：`a100-yky` 上 tmux `glmocr_line_mask_v2_test_20260923`；无 Slurm job。
- **训练来源**：完整行 mask run `glmocr_line_mask_v2_full_20260922_1940`，训练集2159页、dev validation149页、seed42，8 epoch / 3456更新；epoch8/step3456由既定validation选中，validation CER `0.12846305276804149`。训练与选点在 test 读取前完成。
- **训练源码快照**：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/code/line_mask_20260922_v4/ocrmodel`，source tar SHA256 `838c06d66617a001210731043ac735dc5d240cb6cc46148168c3fb8841ff4c26`。
- **选择来源**：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_line_mask_v2_full_20260922_1940/selection.json`，epoch8/step3456；文件记录 `test_manifest_read=false`、`test_used_for_selection=false`。
- **行 mask checkpoint**：同训练目录 `epoch-8.pt`，SHA256 `d56023f270fd51812b58272141f8903bbd2ca1b437b8565b20ab5d12c7f59cac`；训练记录 finite。锁定来源为validation选点，不以test调阈值或选点。
- **共享 backbone 起点**：模型 revision `ca5d8b3e287e52589e37c28385d9655ee4372f9d`；冻结 `glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000` decoder LoRA，SHA256 `ada4cdd3bb1f2f417d1c0a68054dc61e70ab59f215b3bc2018074b4f2fdcd5d5`。
- **数据协议**：官方完整 MTHv2 test manifest `/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1/test/manifest.jsonl`，800页、800个唯一 page_id、split 全为 test、图片缺失0；SHA256 `2904bdaf155a4d1b162d4e4f5fc378cc2263f7d9ea9990c1e3e1020775911962`。全量训练/validation/test官方计数为2159/240/800；本 mask run 依用户既定协议使用149页 validation选点。
- **两条评估臂**：A baseline = 共享 checkpoint-3000 backbone LoRA，关闭行 mask routing；B = 同一 backbone LoRA + 已选 epoch8/step3456行 mask head。唯一区别为 learned line-mask routing开/关。无训练、无优化器、学习率/warmup/loss权重不适用；预算一致，每臂各800页。
- **推理固定项**：全页图像 + 固定 `Text Recognition:` prompt；不向模型输入参考文字、bbox、行ID或test标注；fast processor、max_pixels `4000000`、max_new_tokens `1536`、greedy (`do_sample=false`)、BF16、SDPA、seed42。EOS、触顶和重复循环逐页记录。字符频次诊断未注册为目标；仅报告全量 micro CER、I/D/S、reference字符数、exact page rate和生成健康指标。
- **五卡方式**：物理GPU `0,1,2,3,4`，各一个独立单卡worker，Round-robin五片各160页；每个worker在同一页片上依次评估两臂，因此每张卡仅加载一个模型实例，两臂各覆盖完整test。不是训练DDP。GPU admission仅查询这五张卡的瞬时 utilization.gpu，要求全部严格低于50%；首次核对读数为 `0/0/0/0/0`，启动器启动前会重新核验。
- **协议字段**：test manifest已读取以锁定hash、分片和覆盖；`test_manifest_read=true`；`test_used_for_selection=false`。此后test只作一次性报告，不用于调整任何配置。
- **结果**：待五worker结束并完成无重无漏合并；不得以部分分片或validation结果代替全量test。
- **异常字段**：待记录CUDA/OOM/Traceback/NaN/Inf、worker退出、checkpoint finite、CER finite及EOS/触顶/循环。每个worker日志和分片保留在本run根目录。
- **归档**：完成后将protocol、完整summary、分片预测、运行日志和checkpoint/manifest指纹下载至 `D:/yangky/glm-ocr-assets/line-mask-ddp-20260922/locked-test-20260923/`，再更新本日志与 `EXPERIMENT_REGISTER.md`。test永不回流至selection。

## 全量结果（已完成）

两臂均完成官方test的800页全页自由生成。项目 `aggregate_ocr_metrics` 按去空白字符统计 micro CER：

| 评估臂 | CER | 字符错误 | I / D / S | 参考字符 | exact page | EOS | 触顶 | 循环页 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 无路由 baseline | 0.32235647940700685 | 84,890 | 35,454 / 16,700 / 32,736 | 263,342 | 0/800 | 765/800 (95.625%) | 35/800 (4.375%) | 23/800 (2.875%) |
| 行级 mask epoch8/step3456 | **0.3189312756795346** | **83,988** | 38,549 / 12,015 / 33,424 | 263,342 | 0/800 | 762/800 (95.25%) | 38/800 (4.75%) | 32/800 (4.0%) |

mask 比 baseline 绝对降低 CER `0.003425203727472237`（0.3425个百分点），相对降低约`1.06%`。总错误减少902，但组成发生变化：删除减少4,685，插入增加3,095，替换增加688。平均生成token数为425.53 vs 435.975。test上的收益明显小于validation上的差距（validation选中CER 0.12846305）；该test结果仅作一次性泛化报告，不回流调整head、阈值、生成参数或选点。

**核验**：两臂各800条、page_id唯一且与官方manifest次序和参考文本完全一致；分片为160×5，无重无漏，summary显示`coverage_verified=true`。独立重读原始manifest和两份合并预测后，用项目metrics实现复算的micro CER、I/D/S与summary逐项相同；两臂`I+D+S`分别等于总字符错误，EOS+触顶均为800。有限性检查通过：epoch8 checkpoint 29个head tensor全finite；共享decoder LoRA 192个tensor全finite；指标无NaN/Inf。五个worker均`complete`，launcher exit code 0；日志错误扫描未见CUDA/OOM/NaN/Inf/Traceback/NCCL/异常退出。

**协议与产物**：官方test manifest SHA256 `2904bdaf155a4d1b162d4e4f5fc378cc2263f7d9ea9990c1e3e1020775911962`；epoch8 checkpoint SHA256 `d56023f270fd51812b58272141f8903bbd2ca1b437b8565b20ab5d12c7f59cac`；backbone LoRA SHA256 `ada4cdd3bb1f2f417d1c0a68054dc61e70ab59f215b3bc2018074b4f2fdcd5d5`。`test_manifest_read=true`（本次锁定test评估），`test_used_for_selection=false`；selection.json仍指向epoch8/step3456。原始两臂逐页预测及日志已下载至 `D:/yangky/glm-ocr-assets/line-mask-ddp-20260922/locked-test-20260923/`，本地protocol、summary和两份预测SHA256与远端一致（summary `e780a6dbdb61c70c188d602e239bec09cb6ef197ae17b503ecdf75a5c13cae5a`）。

**结论边界**：在当前整页Greedy协议下，预测行mask相对无路由baseline仅小幅改善micro CER，同时插入、替换和循环页略增、删除明显减少；不支持把validation上约0.0415的改善外推为test增益。该locked test没有参与任何选择或后处理调整。低频字符召回未计算。
