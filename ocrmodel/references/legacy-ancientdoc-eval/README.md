# AncientDoc 历史兼容评估快照

本目录保存项目早期共享环境中的四个评估脚本快照，仅用于复现和核对 AncientDoc split5 的历史解码与指标口径：

- `GOT/eval/myeval.py`
- `scripts/4.batch_calculate.py`
- `scripts/eval.sh`
- `eval-reference.sh`

除本说明外，快照文件保持原样。它们可能包含旧路径、旧模型选择、历史注释或不适合当前服务器策略的并发设置，不得直接作为活动入口。可执行包装器位于 `tools/evaluation/run_legacy_ancientdoc_eval.sh`。

AncientDoc 旧 split 存在书籍级重叠，因此该评估只能作为页面兼容诊断，不能支持小样本、跨书手、跨版本或 VLQA 性能结论。当前正式协议见 `docs/EXPERIMENT_PROTOCOL.md`。
