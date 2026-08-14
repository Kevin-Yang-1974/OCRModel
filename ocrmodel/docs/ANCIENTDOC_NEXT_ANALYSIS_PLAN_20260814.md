# AncientDoc 下一步分析计划

## 目标

当前已完成 `baseline`、`C4`、`C5`、`C6` 在 AncientDoc test split 上的统一验证。下一步不再重复训练，也不继续用 test 选择新配置，而是先对现有 `C4` 与 `C6` 预测结果做逐页配对错误分析，确认 C6 的收益来自哪些页面、哪些书籍或哪些错误类型。

本阶段只做离线分析：

- 不加载模型；
- 不启动训练；
- 不占用 GPU；
- 不读取大型训练日志；
- 不改变已有 validation 结果。

## 输入

默认输入为服务器上的已完成验证目录：

```text
/data3/yky/yangky_ocr_models/evaluation_runs/GOT/ancientdoc_validation_20260814
```

脚本会读取：

- `c4/layout_validation_metrics.json`
- `c6/layout_validation_metrics.json`
- 上述 metrics 中 `predictions` 字段指向的预测 JSONL
- metrics 中 `manifest` 字段指向的 AncientDoc test manifest

如果 metrics 中缺少 manifest，可通过 `--manifest` 显式传入。

## 输出

默认输出目录：

```text
/data3/yky/yangky_ocr_models/evaluation_runs/GOT/ancientdoc_validation_20260814/analysis/c4_vs_c6
```

输出文件：

- `summary.json`：机器可读总摘要；
- `analysis_summary.md`：可直接阅读的分析摘要；
- `page_comparison.csv`：逐页 C4/C6 配对差异；
- `group_comparison.csv`：按 split、类别、书名、文本长度等分组；
- `error_categories.json`：空输出、过短、过长、CER>=1、重复字符等错误类别统计；
- `worst_pages.md`：C6 改善最大、退化最大、C6 最差、C4 最差页面列表。

## 推荐执行指令

本地同步：

```powershell
Set-Location 'D:\yangky\学推计划\ocrmodel'
.\tools\sync\sync_to_server.ps1 `
  -RemoteHost a100-yky `
  -RemoteRoot /data3/yky/yangky_ocr_models/ocrmodel
```

服务器运行：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/evaluation/run_ancientdoc_paired_analysis.sh \
  --suite-root /data3/yky/yangky_ocr_models/evaluation_runs/GOT/ancientdoc_validation_20260814
```

运行完成后只回传最后一行 JSON，或用下面的紧凑命令查看核心结果：

```bash
jq -c '{event:.status,left:.left_label,right:.right_label,overview:.overview,bootstrap:.bootstrap_delta_cer,outputs:{summary:"summary.json",analysis:"analysis_summary.md",page_csv:"page_comparison.csv",group_csv:"group_comparison.csv"}}' \
  /data3/yky/yangky_ocr_models/evaluation_runs/GOT/ancientdoc_validation_20260814/analysis/c4_vs_c6/summary.json
```

## 结果使用口径

本分析只能回答“在当前 AncientDoc 页面级 split 上，C6 相对 C4 改善、持平或退化发生在哪里”。它不能证明跨书籍、跨版本、跨馆藏泛化，也不能单独证明收益来自 VLQA 布局结构。

若 C6 的改善集中于少数长页或特定书籍，后续应优先做来源泄漏审计和分组隔离划分。若改善较均匀，再进入 validation-only replay 比例筛选与等参数 adaptor、无布局监督 queries 的结构归因对照。

## 下一步：来源泄漏审计

逐页分析完成后，继续审计当前 AncientDoc train/validation/test 是否存在跨 split 来源重叠。该命令仍然只做离线数据检查，不训练、不推理、不占 GPU。

服务器运行：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/preprocessing/run_ancientdoc_split_leakage_audit.sh \
  --dataset-id ancientdoc_layout_260707
```

默认输出：

```text
/data3/yky/yangky_ocr_models/training_data/got_layout_pages/ancientdoc_layout_260707/audit/ancientdoc_split_leakage
```

核心文件：

- `split_leakage_audit.json`
- `split_leakage_audit.md`

若要额外做图像感知哈希近重复检查，使用：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/preprocessing/run_ancientdoc_split_leakage_audit.sh \
  --dataset-id ancientdoc_layout_260707 \
  --enable-perceptual-hash
```

感知哈希检查会读取全部图片并做跨 split 近重复比较，耗时会高于默认精确哈希审计。默认先运行不带 `--enable-perceptual-hash` 的版本。

## 重建 group-isolated AncientDoc split

当前审计若返回 `book_key_cross_split > 0`，说明同一 `category/book` 跨 train/validation/test。此时应保留旧数据集作为页面级适配基线，另建一个按 `category/book` 隔离的新数据集。

服务器运行：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/preprocessing/run_ancientdoc_group_isolated_prepare.sh
```

默认读取：

```text
/data4/hyf/backup/GOT-OCR2.0/reference-260707/AncientDoc
```

默认输出：

```text
/data3/yky/yangky_ocr_models/training_data/got_layout_pages/ancientdoc_layout_260707_group_isolated_seed20260814
```

该入口默认使用 `train/validation/test=0.6/0.2/0.2` 的页面数近似比例，并把同一个 `category/book` 的全部页面固定到同一个 split。图片默认使用 symlink 指向只读原始 AncientDoc 图片，避免重复占用空间；如果必须复制图片，可加 `--copy-images`。

生成后必须立即对新数据集重新运行泄漏审计：

```bash
cd /data3/yky/yangky_ocr_models/ocrmodel
source config/paths.env
bash tools/preprocessing/run_ancientdoc_split_leakage_audit.sh \
  --dataset-id ancientdoc_layout_260707_group_isolated_seed20260814
```

只有当 `book_key_cross_split=0`、`original_image_cross_split=0`、`normalized_text_cross_split=0` 且 `image_sha256_cross_split=0` 时，才把该数据集作为后续 AncientDoc 小样本与 replay 比例筛选的基础。
