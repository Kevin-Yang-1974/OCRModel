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

## 全量结果

待运行完成后追加：baseline与epoch8/step3456各800页micro CER、I/D/S、EOS/触顶/循环、覆盖核验、finite与异常扫描，以及baseline差值。原始预测保留在本轮归档目录。
