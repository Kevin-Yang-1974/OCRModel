# GOT-OCR2.0 活动源码

本目录保留 GOT-OCR2.0 上游包结构，并承载本项目当前的整页 VLQA、页面级数据集和训练改动。它是仓库中唯一的活动 GOT2 源码树。

模型、数据、checkpoint、日志和结果均放在仓库外，通过 `../../config/paths.env` 或命令行参数选择。不要在本目录下建立指向服务器资产的软链接。

当前主线是原始整页图像输入，不再以 line-level 作为正式训练入口。`scripts` 里的 AncientDoc 入口只用于适配、诊断和历史兼容；正式训练与验证入口见 `../../tools/training/README.md` 和 `../../docs/PROJECT_STATUS.md`。
