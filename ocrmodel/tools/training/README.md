# A100 训练入口

`run_layout_a100.py` 是 GOT2 整页训练与验证的统一编排器。它只读训练数据，不生成页面，也不改写远端原始数据。

## 统一入口

- `--mode pretrain`：从原始 GOT2 做 P1。
- `--mode joint-train`：从已完成的 P1 VLQA checkpoint 做 P2。
- `--mode adapt`：从已完成的 P2 VLQA checkpoint 做真实域或 replay 适配。
- `--mode validate`：只做 prompt-only 整页验证，不训练。
- `--mode smoke`、`--mode overfit`、`--mode pilot`：只保留给工程诊断。

## 当前可直接跑的 baseline

- `adapt` + AncientDoc OCR-only。
- `adapt` + AncientDoc + synthetic OCR replay。
- `adapt` + AncientDoc + synthetic layout replay。
- `joint-train` loss 消融：`full`、`no-direction`、`no-bbox`、`object-only`、`ocr-only-adapter`。

## AncientDoc 数据

远端原始数据位于 `/data4/hyf/backup/GOT-OCR2.0/reference-260707/AncientDoc`，内容是：

- `label_for_got.json`
- `label_for_got_split1.json` 到 `label_for_got_split5.json`
- `imgs/`

训练前先把它转换成 `layout-page-jsonl` 格式，再让 `run_layout_a100.py` 读取转换后的 manifest。转换脚本是 `tools/preprocessing/prepare_ancientdoc_layout_dataset.py`。

## 训练与验证

- 训练总入口：`tools/training/run_ancientdoc_baseline_suite.sh`
- 验证总入口：`tools/evaluation/run_ancientdoc_validation_suite.sh`
- 验证完成后：服务器结果目录同时包含机器可读的 `summary.json` 和可直接阅读的 `report.md`。
- 首次结果记录：`docs/ANCIENTDOC_BASELINE_REPORT_20260814.md`。

训练入口支持 `--start-from c5` 或 `--start-from c6`。跳过 C4 时必须通过 `--c4-model` 指定已完成的 C4 `p2/model`；从 C4 开始时，入口会自动把新生成的 C4 checkpoint 传给 C5 和 C6，不需要重复填写。

## 还未实现的结构 baseline

- 等参数量普通视觉 adaptor。
- 无布局监督 queries。
- oracle/pseudo-layout region-token adapter。
- 全量解冻上限与更细粒度 decoder 解冻策略。
