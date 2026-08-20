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
- `train_GOT_page_ocr.py`：不启用 VLQA 的原始 GOT2 整页 OCR-only 训练入口，用于 C1；
- `train_GOT_layout.py`：`P1` 布局预热与 `P2` 联合训练的开发入口；
- `verify_layout_checkpoint.py`：检查 VLQA/projector 权重、P1 零门控和最终模型重载；
- `../GOT/model/layout_query.py`：有序 queries、object/bbox/direction 头、零门控写回与损失；
- `../../../tools/preprocessing`：HTML 页面生成和无泄漏审计。

`../../../tools/training/run_layout_a100.py` 已把环境、数据、GPU、P1/P2、validation 和 checkpoint 检查串成受限入口，并已在 A100 打通 CUDA forward/backward、整页 batch、P1→P2 加载和最终模型重载。`layout_overfit_20260812_002747` 已通过固定 1000 steps 的 P1 两样本实现诊断；`layout_page_dataset.py`、`evaluate_GOT_layout.py` 和 `layout_validation_metrics.py` 已提供 prompt-only validation loader、整页 generation 与统一页面指标。tokenizer 预检修复后的 `layout_validate_20260812_014816` 已在两页 `train` split 上完成链路验证，但使用的是同页 P1 overfit checkpoint；正式 split、held-out 效果验证仍未完成，当前不得报告泛化性能。最短上传与运行命令见 `../../../docs/SYNC_AND_RUN.md`。

C1 通过 `../../../tools/training/run_got2_page_ocr_a100.py` 调用 `train_GOT_page_ocr.py`，只接收原始整页图像、OCR prompt 和页面文本真值，不把 layout metadata 送入模型。其原生 `decoder_projector` 策略与 C4 的 VLQA adapter 策略不等参数，且上游 checkpoint 历史不同，因此只作为完整适配路线对照。

## 页面兼容入口

- `audit_ancientdoc_dataset.py`：审计 AncientDoc 页面和五折标签。
- `ancientdoc_dataset.py`：页面级兼容数据集。
- `preflight_ancientdoc_dataset.py`：最长页面 CPU 预检。
- `train_GOT_ancientdoc.py`：页面兼容训练实现。
- `run_ancientdoc_train.sh`：`smoke` 或 `reference` 启动器。

`reference` 默认训练 6 epochs，可通过 `ANCIENTDOC_EPOCHS` 显式覆盖。由于现有 AncientDoc split 存在书籍级泄漏，且该路径没有 VLQA，这条路径只能报告为历史页面兼容基线，不能作为当前小样本、跨书手泛化或布局查询结果。

完整同步、预检、训练和评估命令见 `../../../docs/SYNC_AND_RUN.md`。
## A0-A5 整页消融

`train_GOT_layout.py --ablation_id ... --layout_loss_preset ...` 是统一训练底层入口。A1 只训练 `mm_projector_vary`；A2 训练 projector 与独立 `GenericVisualTransformerAdapter`；A3/A4 训练 projector 与相同 VLQA；A5 保留原 P1→P2。未指定 `--ablation_id` 时保留原 P1/P2 默认语义。

`evaluate_GOT_layout.py` 的 `--model-kind` 支持 `baseline`、`generic` 和 `vlqa`，所有类型都只接收 `whole_page_image` 与 OCR prompt；布局 metadata 不进入模型。

## PVLD-32 独立原型

原有 `layout_query.py` 是 **Fixed-Slot VLQA baseline**：K16 使用 `max_regions=16`，K32 仅把固定槽位容量提高到 32，二者都按训练期 query 顺序对应区域；超过上限时使用 oracle chunk 或截断。K32 只是容量对照，不是变量长度创新。

`../GOT/model/layout_prompt_decoder.py`、`serialize_layout_targets.py`、`train_GOT_variable_layout.py` 和 `evaluate_GOT_variable_layout.py` 提供独立的 **Prompted Variable-Length Layout Decoder（PVLD-32）** 原型。PVLD 的 32 个 prompt token 不绑定区域编号，区域数量由结构 token 序列中的 `REGION` 记录和 `EOS` 决定；decoder 保留对完整页面视觉 token 的访问。`train_GOT_variable_layout.py` 在没有预计算 `visual_features` 时只执行 manifest/config preflight，不伪造 GOT2 训练或效果；正式接入 GOT2 视觉塔和 `GOTQwenModel.forward` 仍是后续任务。MTHv2 ordered textline 与 oracle-chunk 结果必须和 whole-page 结果分开。
