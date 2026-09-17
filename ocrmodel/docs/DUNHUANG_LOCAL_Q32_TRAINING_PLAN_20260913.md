# 敦煌／地方志 q32 训练计划（2026-09-13）

## 一、当前问题判断

当前 2000-step 任务与短程参数优选并不是同一优化轨迹。虽然峰值学习率、`auxiliary_weight` 和 warmup 相同，但当前任务使用 2000-step cosine horizon；step 500 仍处于较高学习率区间，而短程优选的 step 600 已到达 schedule 末端。继续训练到 step 1000–2000 后，decoder LoRA 出现自由生成漂移、自循环、EOS 命中下降和 1536-token 触顶，属于后程退化，不能只由训练 loss 判断。

此外，A100 使用五卡同步 DDP，BSCC 使用四卡同步 DDP，global batch 和每步样本组成不同；两平台结果只能作为趋势参照。当前明确保留 1536-token，因此也不把历史 768-token CER `0.198717` 当作本轮必须精确复现的目标。

## 二、阶段 A：600-step 参考对照

同时运行 geometry 与官方 GLMOCR content-only baseline，使用当前代码、当前 BSCC 四卡环境和同一数据协议。

| 项目 | Geometry | Baseline |
| --- | --- | --- |
| 训练页／validation 页 | 240／80 | 240／80 |
| Seed、queries | 42、32 | 42、32 |
| Optimizer steps／LR horizon | 600／600 | 600／600 |
| Checkpoints | `200, 400, 600` | `200, 400, 600` |
| Adapter peak LR | `2.5e-5` | `2.5e-5` |
| Decoder LoRA LR | `5e-6` | `5e-6` |
| LoRA rank／alpha／dropout | 8／8／0 | 8／8／0 |
| Auxiliary weight | `0.4` | `0` |
| Warmup／grad clip | 216／1 | 216／1 |
| Assignment／loss | Hungarian／full | content-only |
| Validation generation | plain、1536-token | plain、1536-token |
| 数据边界 | validation-only；不读取 test | validation-only；不读取 test |

每个任务先完成独立 8-step smoke，再进入 600-step 正式训练。训练期间保存 `checkpoint-200`、`checkpoint-400`、`checkpoint-600`；阶段 A 对这三个 checkpoint 做统一 validation 选点，避免把 held-out test 用于调参。

## 三、阶段 A 验收

必须同时检查：

1. checkpoint 与全部训练参数 finite，无 CUDA、OOM、NaN/Inf 或异常退出；
2. 80 页 validation 完整覆盖，预测文件可逐行解析且页 ID 唯一；
3. micro CER、去空白 CER、macro CER、mean NED、exact match、平均编辑距离；
4. 插入／删除／替换、EOS 命中率、1536-token 触顶率、repeated trigram、repeated cycle 和 loop page rate；
5. geometry 与 baseline 使用相同 manifest、模型 revision、processor、generation 和训练预算。

循环页率或触顶率超过 5% 时，不进入 test；先进入阶段 B。CER 只在协议完全相同时比较，不把历史 768-token 数值直接混入本轮表格。

## 四、阶段 B：只解决后程退化

若阶段 A 的 geometry 稳定但长程任务退化，保持网络结构、数据和解码不变，仅测试一个新增变量：`auxiliary_weight=0.4 + warmup_steps=500`，仍训练 600 steps。若该组合未优于 `warmup=216`，立即停止该方向。

若阶段 A 在 600 steps 已出现循环，则不延长训练。先对 checkpoint 做页级错误归因，区分密集页、敦煌页和地方志页，并核对 EOS、label truncation、预测长度与 decoder LoRA 参数漂移；在这些证据完成前不加入 repetition penalty、强制 EOS 或新损失，以免改变比较协议。

## 五、阶段 C：正式比较

阶段 A/B 选出的唯一配置用于 geometry 与 baseline 的正式比较。训练保存 200／400／600 三个 checkpoint，完整 validation 选点后只锁定一个 checkpoint；然后对 59 页 held-out test 各执行一次 selection-locked test。最终与其他方法统一汇总 micro CER、去空白 CER、macro CER、mean NED、exact match、平均编辑距离及循环相关指标。

若需要统计稳定性，再扩展 seed 43/44；不得在看到 test 后修改 checkpoint、解码参数或后处理。
