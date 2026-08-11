# GOT-OCR2.0 活动源码

本目录保留 GOT-OCR2.0 上游包结构，并包含本项目的整页 VLQA、页面级数据集和训练改动。它是仓库中唯一的活动 GOT2 源码树。

在锁定的 GOT 服务器环境中执行：

```bash
python -m pip install --no-deps -e .
```

模型、数据、checkpoint、日志和结果均位于仓库外，通过 `../../config/paths.env` 或命令行参数选择。不要在本目录建立指向服务器资产的软链接。

当前正式输入是原始整页图像。`scripts` 中的 line-level 与 AncientDoc 入口只用于诊断和历史兼容；VLQA 状态与运行门槛见 `../../docs/PROJECT_STATUS.md`，上游来源见 `../../NOTICE.md`。
