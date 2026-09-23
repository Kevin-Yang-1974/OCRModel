# GLM-OCR 分支协作规则

本文件适用于本 worktree 及全部子目录。当前分支为 `glm-ocr-layout-ot`，与原 GOT2 工作区隔离。

1. 研究对象是小样本条件下的多场景通用符号识别；古籍和谱面是验证场景，不限定模型适用范围。
2. 当前主线是 GLM-OCR 核心模型及其视觉内容合并前的布局适配。布局条件化半松弛最优传输是待验证主候选，必须与 content-only、普通 attention 和几何融合在统一协议下比较。
3. 正式推理只接收整页图像和 prompt。bbox、阅读顺序、书写方向及 token-region 对应只可用于训练监督、评测真值或可选解释性输出，不得作为推理输入。
4. 首轮机制筛选使用 128 页、seed 42。R1 表示领域级少样本，R2 表示稀有符号级 K-shot；必须按书手、版本、馆藏或符号类型做隔离，并阻断同页与近重复泄漏。
5. 当前是工程实施阶段。文档要区分已实现、正在实施和待验证内容，不虚构实验数据或把候选方法写成已验证结论。
6. `ocrmodel/` 是唯一代码编辑区。不从原 GOT2 实验目录导入模块；需要复用时提取独立组件，在文档中记录来源和改动。
7. 模型、数据、checkpoint、日志和实验产物必须位于源码树外；共享 `/data4/hyf` 只读，个人产物写入 `/data3/yky/yangky_ocr_models` 下的本分支隔离目录。
8. 本地默认只做静态检查、CPU 单元测试和小型张量测试。不自动同步、推送、提交或启动正式训练。
9. A100 入口须显式限定允许的物理 GPU，仅查询该集合的瞬时 `utilization.gpu`。所有目标卡都严格低于 50% 才启动；否则整体退出，不等待、不抢占、不查询集合外 GPU。
10. 正式训练前必须锁定 train/val_tune/val_verify/test 协议。`val_tune`用于参数优选、架构/损失/阈值/后处理比较、early stopping和checkpoint选点；`val_verify`仅在方案、参数和checkpoint全部锁定后用于最终验证。参数优选与最终验证的Val不得重叠或交叉使用，按page_id、图像内容及来源组（书手/版本/馆藏等适用隔离单位）审计，并阻断同页和近重复泄漏；保存各自manifest、SHA256、用途与交集报告。用于任何调参或选点的数据不得再作为独立最终验证集，历史已暴露的Val不能通过重新命名或事后拆分宣称独立。不得根据val_verify结果继续调参或挑checkpoint；如据其结果改方案，该集合转为development，下一次独立验证必须另锁未暴露集合。锁定后再执行selection-locked test；test不参与训练、选点、阈值或后处理调整。
11. A100/BSCC 任务监控：`PENDING`/排队阶段每 30 分钟查看一次；首次进入 `RUNNING` 后只做最多两次、间隔 5 分钟的健康检查。健康检查必须确认会话仍在、结构化状态为运行态、metrics/日志/输出有实际进展，且无 NaN/Inf、CUDA/OOM、Traceback、异常退出或资源异常。连续两次健康后必须立即降频：256-step 诊断、短任务或单次验证为每 30 分钟一次；正式全量训练、长任务或大规模 array 为每 1 小时一次。不得继续按 5 分钟频率检查。
12. 定时监控实现：桌面 Codex 使用绑定当前线程的 automation/heartbeat；初始可设 5 分钟，但两次健康检查后必须通过 `automation_update` 更新同一个 automation 的完整字段和 `rrule`，切换到 30 分钟或 1 小时，保留正确的线程、job/run ID、只读约束和 test 隔离规则，不得创建重复 automation。阶段切换时立即检查并重新设置阶段频率；用户要求每次触发可见回报时，降频后仍须简短回报。任务完成、失败、异常退出、NaN/Inf、CUDA/OOM、Traceback 或需要决策时立即回报；任务完成并完成结果核验后删除或停用旧监控，避免 stale automation。
13. 实验记录：每个 run（含机制探针、离线诊断、消融、按用户要求中止或失败的 run）都必须在 `ocrmodel/docs/EXPERIMENT_REGISTER.md` 登记，需要展开的写入 `ocrmodel/docs/实验日志/`。记录详细程度以「可直接用于论文」为准，缺少下列任一项即视为不完整，不得据此下结论或写入文档：
    - 标识与产物：run ID、日期、分支/commit、入口脚本或 config 路径、远端产物根目录、Slurm job/array ID。
    - 数据协议：manifest 指纹、隔离单元、页数与 train/validation/test 划分、seed。
    - 训练配置：学习率（主 adapter 与 decoder LoRA 分别写）、warmup 与 schedule、总步数、per-device batch size、梯度累积、GPU 数与 DDP 方式、effective global batch、精度、可训练参数范围与秩、loss 权重（`auxiliary_weight`、gate 初值、各项 layout 权重）。
    - 组别设置：每条对照臂的 run ID、唯一改动变量、共享起点 checkpoint，并说明各臂预算是否一致。
    - 结果：validation 选点 step 与全部被比较指标、locked test 指标（IoU/MAE/CER、I/D/S）、EOS/触顶/循环率、loss 曲线关键点、checkpoint 是否 finite、有无 NaN/Inf/OOM/Traceback。
    - 协议字段：`test_manifest_read`、`test_used_for_selection`；未跑 test 也要显式写「未读取」，不得留空。
    - 失败与中止 run 同样登记，注明原因、产物是否保留和可否复用；不删除、不覆盖既有记录，修正以追加说明的方式写入。
14. 任何训练或评测子集都必须从完整候选池按适用的来源组和任务复杂度分层抽取，明确报告简单稀疏、中等及高密度/高复杂度样本的覆盖；不得只选容易的 sparse 样本。候选池本身缺少复杂度跨度时，只能报告其实际覆盖范围，不得宣称覆盖了未出现的复杂样本。
