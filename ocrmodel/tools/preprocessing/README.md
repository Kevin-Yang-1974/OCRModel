# Whole-page 数据准备

主线数据始终以整页图像为样本。页面区域的 bbox、方向和阅读顺序写入 manifest，供 PVLD 训练和离线评测使用，不进入推理输入。

## 合成页面

`generate_synthetic_layout.py` 生成带 DOM 几何真值的 whole-page 页面；`prepare_diverse_synthetic_layout.py` 生成 S3/S4 多样化页面；`audit_synthetic_layout.py` 在 train/validation/test manifest 一起运行，检查页面哈希、source/content ID、近重复、字体和跨 split 泄漏。

正式数据必须先按来源组、书手、版本、馆藏或内容 ID 划分 split，再在 split 内生成模板和退化版本。同一源页、近重复页或同一内容不得跨 split。输出目录由外部数据根指定，不覆盖已有数据。

## MTHv2

`prepare_mthv2_layout_dataset.py` 将官方 whole-page 页面与 `label_textline` 转为有序区域候选。该字段不是严格 column ground truth。官方 `train/validation/test` split 在转换时固定并按源页面继承：`train` 供 P1 replay 和 P3 训练，`validation` 不进入训练，`test` 只在 P3 validation selection 后执行 selection-locked test。

MTHv2 不提供完整书手/版本/馆藏分组信息，报告中必须注明这一限制。不得先生成 oracle chunk 再随机划分。Chunk 数据和分块脚本只在归档分支复现，不能与主线 whole-page 结果直接比较。

## Validation lock

`prepare_time_constrained_validation.py` 从 S3/S4 validation 固定 400 页：S3 与 S4 各 200 页，并按 region-count bucket 与 complexity tertile 分层。输出 lock 文件记录来源 manifest SHA-256、固定清单 SHA-256、分层计数以及 `test_used_for_selection=false`。P1、P2、P3 共用该清单。

## 本地检查

```bash
python tools/preprocessing/generate_synthetic_layout.py --help
python tools/preprocessing/audit_synthetic_layout.py --help
python tools/preprocessing/prepare_mthv2_layout_dataset.py --help
```

服务器上的数据、权重和日志不属于源码同步范围。同步规则见 [`docs/SYNC_AND_RUN.md`](../../docs/SYNC_AND_RUN.md)；分支与数据协议见 [`docs/BRANCH_AND_DATA_LAYOUT.md`](../../docs/BRANCH_AND_DATA_LAYOUT.md)。

历史 AncientDoc、BSCC、旧 MTHv2 Chunk/布局准备和 server synthesis wrapper 已移至 `archive/legacy-vlqa-chunk-20260829`。
