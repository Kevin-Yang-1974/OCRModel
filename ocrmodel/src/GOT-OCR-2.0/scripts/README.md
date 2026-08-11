# 项目训练入口

本目录包含活动 GOT 数据与训练代码。服务器外部路径由仓库根目录的 `config/paths.env` 提供；不要在这里建立数据或权重软链接。

## 既有 line-level 诊断入口

- `linelevel_dataset.py`：读取 line-level 单行或单列图像及转写。
- `preflight_linelevel_dataset.py`：不加载模型权重的数据链路预检。
- `train_GOT_linelevel.py`：line-level 训练实现。
- `run_linelevel_smoke.sh`：只运行 1 个 optimizer step 的工程 smoke。
- `verify_linelevel_checkpoint.py`：检查训练前后权重键与目标模块变化。

上述入口保留用于 AnandaSky 兼容对照、字符识别诊断和既有工程 smoke。当前 `run_linelevel_smoke.sh` 只验证工程链路，不是正式训练配置，也不包含整页布局 queries。

## 当前主方案状态

当前正式主方案是整页 GOT2＋端到端 Visual Layout Query Adapter（VLQA），详见 `../../../docs/GOT2_LAYOUT_ENCODING_PLAN.md`。首版活动代码包括：

- `layout_page_dataset.py`：整页 manifest dataset 和布局监督 collator；
- `train_GOT_layout.py`：`P1` 布局预热与 `P2` 联合训练的开发入口；
- `verify_layout_checkpoint.py`：检查 VLQA/projector 权重、P1 零门控和最终模型重载；
- `../GOT/model/layout_query.py`：有序 queries、object/bbox/direction 头、零门控写回与损失；
- `../../../tools/preprocessing`：HTML 页面生成和无泄漏审计。

`../../../tools/training/run_layout_a100.py` 已把环境、数据、GPU、P1/P2 和 checkpoint 检查串成受限入口，并已在 A100 打通 CUDA forward/backward、整页 batch、P1→P2 加载和最终模型重载。首次 2 页长跑没有通过 P1 可拟合性检查，当前代码已增加 FP32 分项日志和固定 P1 两样本 `--mode overfit`；该诊断通过前不启动新 pilot。validation loader 和统一指标仍未实现，当前不得报告训练效果。最短上传与运行命令见 `../../../docs/SYNC_AND_RUN.md`。

## 页面兼容入口

- `audit_ancientdoc_dataset.py`：审计 AncientDoc 页面和五折标签。
- `ancientdoc_dataset.py`：页面级兼容数据集。
- `preflight_ancientdoc_dataset.py`：最长页面 CPU 预检。
- `train_GOT_ancientdoc.py`：页面兼容训练实现。
- `run_ancientdoc_train.sh`：`smoke` 或 `reference` 启动器。

`reference` 默认训练 6 epochs，可通过 `ANCIENTDOC_EPOCHS` 显式覆盖。由于现有 AncientDoc split 存在书籍级泄漏，且该路径没有 VLQA，这条路径只能报告为历史页面兼容基线，不能作为当前小样本、跨书手泛化或布局查询结果。

完整同步、预检、训练和评估命令见 `../../../docs/SYNC_AND_RUN.md`。
