# Line-mask validation 注意力对照图（2026-09-23）

## 目的与数据边界

从正式训练 `glmocr_line_mask_v2_full_20260922_1940` 的 149 页 validation 归档中，筛选 baseline 发生插入/删除、而 epoch-8 line-mask 在对齐位置读对目标字的样例；生成 baseline 与行级 mask-routing 的末层 patch 注意力并叠加在原图上。样例选择、字符框、两臂预测与注意力回放都只使用 validation。`test_manifest_read=false`，`test_used_for_selection=false`；locked test 的 manifest、图像和预测均未用于本诊断。

原 validation manifest 指纹：`36ec845875e1ea18a48d4a523b6c5a3f007b46e2a07de0cafd2c26139929a348`，149 页。训练协议的 train split 为 2159 页、指纹 `1016198040944e39329712eb2a7bdfe6db7526134b91d750c83afa91d854f9b3`；原实验 test split 为 800 页，本诊断未读取。划分沿用正式训练协议，seed 42；该协议没有声称按书手、版本或馆藏隔离。

正式训练执行源码快照为 `/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/code/line_mask_20260922_v4/ocrmodel`，tar SHA256 `838c06d66617a001210731043ac735dc5d240cb6cc46148168c3fb8841ff4c26`。本诊断新增 capture/runner 脚本在本地分支，未改动共享源快照。验证预测归档来自 `D:/yangky/glm-ocr-assets/line-mask-ddp-20260922/baseline-pred/` 和 `D:/yangky/glm-ocr-assets/line-mask-ddp-20260922/formal-20260923/validation-epoch8/`；本次筛选后的两份10页 JSONL SHA256 分别为 `ada447b620b226195694f7cfc8fa8dc8f91b45fc1c3b899eb6ef578d636dc02` 与 `7cbb875ea16aedc27d487dde65de4c423c5200ed3c045646a90a7fc0fd39ab1f`。

## 运行记录

| 字段 | 失败的初版 | 最终完成版 |
|---|---|---|
| run ID | `glmocr_line_mask_v2_attention_val_20260923` | `glmocr_line_mask_v2_attention_val_20260923_v2` |
| 日期、分支/commit | 2026-09-23；`glm-ocr-layout-mask-routing` / `e40a3a5dbbd65adee3cd8afcd809825f30369109` | 同左 |
| 入口 | `ocrmodel/tools/evaluation/capture_line_mask_attention_cases.py` 与 `run_line_mask_attention_capture_a100.sh`；初版 capture SHA256 `4d42d52cb17076bbb51b9c8e328f0a8e48fd02176849eda3ef271fb522c2d776` | 同左；capture SHA256 `03450d6c65f13d6ced27028d5c276f7d895c792c003dc2b737ef76b7bd8cabe4`；runner SHA256 `1825a6d1f84d71912fd7b0ad653e60db0fff9c7aa6dc7917b8697b7c13cd56cb`；renderer SHA256 `9014e494a88cad8e4d385fab26d9b06531e629d98b13c9c14801539abb5a2ec2` |
| 远端产物根 | `/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/diagnostics/glmocr_line_mask_v2_attention_val_20260923` | `/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/diagnostics/glmocr_line_mask_v2_attention_val_20260923_v2` |
| GPU / 调度器 | 物理 GPU 2；启动时 `utilization.gpu=40%`；Slurm job/array 不适用，前台 SSH 执行 | 物理 GPU 2；启动器重新检查并记录 `utilization.gpu=40%`；Slurm job/array 不适用，前台 SSH 执行 |
| 状态与原因 | `failed`。为了拿 attention 将末层切至 eager，`mthv2_mth1200_JX_260_1_224` 的 baseline 在目标前缀第 7 个字与归档预测分叉，保护性校验中止；前两页留下的 NPZ 保留但未用于最终图。无训练、无 NaN/Inf、无 OOM；该脚本报出的 `RuntimeError` 是前缀不匹配。 | `complete`。10 个不同 validation 页、baseline 与 line-mask 各一轮，共 20 条前缀均与归档完全一致；10 组 attention NPZ 与 summary 完整。无训练、无 NaN/Inf/OOM/Traceback。 |

两次运行都不训练、不选 checkpoint、不调阈值或后处理。GPU admission 仅指定并查询物理 GPU 2；`CUDA_VISIBLE_DEVICES=2`，容器内使用 `cuda:0`。没有读取其他 GPU 指标。

## 上游正式训练配置与对照组

本次只回放既有 validation 预测。为满足配置溯源，下面列出产生所比较归档的正式训练设置；它们不是本次诊断新启动的训练参数。

| 项目 | 正式训练记录 |
|---|---|
| 上游 run / 选点 | `glmocr_line_mask_v2_full_20260922_1940`；epoch 8 / step 3456；149 页 validation 选点 |
| 共享起点 | 模型 `/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d`；冻结 decoder LoRA `/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/training_runs/glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000`，SHA256 `ada4cdd3bb1f2f417d1c0a68054dc61e70ab59f215b3bc2018074b4f2fdcd5d5` |
| 行级 head | `/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_line_mask_v2_full_20260922_1940/epoch-8.pt`，SHA256 `d56023f270fd51812b58272141f8903bbd2ca1b437b8565b20ab5d12c7f59cac`；501892 个可训练参数；配置 hidden 1536、dim 128、bias 1.0、threshold 0.5、detach_every 32 |
| 优化 | head LR `2e-4`，decoder LR `0`；warmup 172 steps；3456 updates；完整周期 cosine 到初值的 0.1 倍；AdamW、weight decay 0.01、gradient clip 1 |
| batch / 并行 | 5 卡 DDP，per-device batch 1、gradient accumulation 1、effective global batch 5；seed 42 |
| 精度 / 可训练范围 | 冻结骨干 BF16；head FP32；decoder LoRA rank 8 / alpha 8 / dropout 0 且冻结；仅训练行级 head |
| loss | BCE 1、Dice 1、location 0.2、area 0.2、transition 0.2、stop 0.05、empty 0.2；auxiliary weight 0、layout weight 0。更新门为两层 MLP（PyTorch 默认参数初始化）；空的首个 mask 强制 update=1；无 residual geometry gate |
| 空间路由 | hard mask；阈值 0.5；bias 1.0；所有 decoder 层；prefill 不注入；target shift 2 |
| 正式生成协议 | fast processor、max pixels 4,000,000、max new tokens 1,536、greedy、BF16、math-SDPA |

| 对照臂 | 预测来源 | 唯一推理差异 | 预算/起点 |
|---|---|---|---|
| Baseline | `line-mask-ddp-20260922/baseline-pred/` | 不启用 line-mask 路由 | 共用 checkpoint-3000 LoRA、同 validation 图像/prompt、greedy 与原正式生成协议 |
| 行级 mask-routing | `formal-20260923/validation-epoch8/` | 启用由 validation 选定的 epoch-8 head | 共用上述 LoRA、同 validation 图像/prompt、greedy 与原正式生成协议 |

上游 full-run 是一个训练 head 的单臂结果，没有等预算的独立训练对照；图仅展示已归档 baseline 与已选 epoch-8 方法的例子，不将收益归因于特定 loss 项或 head 子模块。

## 例子选择与图像生成

对 baseline 和 epoch-8 预测按项目字符编辑距离做 Levenshtein 对齐，优先级为 substitution、deletion、insertion；筛选 baseline 在目标处缺字/多字且方法在对应目标位置输出正确字符的不同页面。保留 5 个 deletion 与 5 个 insertion。全页 validation 归档指标用于筛例背景，未在本诊断中重算 CER：baseline CER `0.16992365679166466`，I/D/S=`1948/1082/4048`；epoch-8 CER `0.12846305276804149`，I/D/S=`796/531/4024`。

| Case | 错误类型 | Page ID | 目标字 | Baseline query / 对齐位置字符 | Mask-routing 输出 | 页面编辑数 baseline → method |
|---:|---|---|---|---|---|---:|
| 01 | deletion | `mthv2_mth1000_114` | 一 | 乃 | 一 | 179 → 39 |
| 02 | deletion | `mthv2_mth1000_009` | 生 | 老 | 生 | 104 → 19 |
| 03 | deletion | `mthv2_tkh_0001_022_29_04` | 四 | 佛 | 四 | 81 → 55 |
| 04 | deletion | `mthv2_tkh_0001_013_26_09` | 有 | 智 | 有 | 30 → 12 |
| 05 | deletion | `mthv2_tkh_0001_038_29_11` | 苦 | 空 | 苦 | 23 → 9 |
| 06 | insertion | `mthv2_tkh_0001_021_28_11` | 無 | 世（多字序列的首个额外字） | 無 | 1008 → 66 |
| 07 | insertion | `mthv2_tkh_0001_021_28_01` | 不 | 也 | 不 | 97 → 48 |
| 08 | insertion | `mthv2_tkh_0001_016_26_04` | 不 | 也 | 不 | 27 → 3 |
| 09 | insertion | `mthv2_mth1200_JX_260_1_224` | 生 | 。 | 生 | 29 → 22 |
| 10 | insertion | `mthv2_tkh_0001_030_25_02` | 在 | 外 | 在 | 32 → 26 |

每个 attention query 取所显示 token 对应的 decoder forward；deletion 的 baseline 没有缺失字本身的 token，因此图中显示 Levenshtein 对齐边界处的 baseline query，并框出未输出的目标字。模型生成只收到整页图像和固定 OCR prompt；标注字符框只用于事后框选与筛例，不输入模型。重放按目标位置动态设置 `max_new_tokens`，上限 512，足以覆盖所选字符，不用于全页指标比较。

attention 定义：末 decoder 层、对 heads 求均值、最后一个 query 对 image patch keys 的原始概率；NPZ 同时存原始 patch mass 与归一化空间分布，并记录 query token、生成 step、visual mass。为了保持归档预测不变，末层仍通过原 SDPA 计算模型输出；分析 wrapper 仅额外重算该 query 的 QK softmax 并返回诊断权重，不把它送回模型。每张图内左右 panel 使用同一 raw score 上限；不同样本不共用色标。patch map 双线性插值回原图，绿框使用 validation 字符标注的 normalized xyxy 框。10/10 框、字位置与图像原始尺寸已逐页核验。

这些热力图是注意力读数的可视化，不能单独证明因果定位或预测正确的机制；判读时同时看原图、目标框、输出字符和整页错误变化。

## 训练结果与异常核验

本诊断没有优化器步骤、没有新 loss 曲线。上游 epoch-8 checkpoint 已载入并 finite，501892 个参数；正式训练末 100 步平均 loss 1.5327、BCE 0.1148、Dice 0.4539、location 4.3116、area 0.2617、transition 0.2248、stop 0.0359、IoU 0.6325、mass ratio 2.5306。正式 baseline EOS 148/149、触顶 1、循环 1；epoch-8 EOS 149/149、触顶 0、循环 0；两者 exact page rate 均 0。它们是上游 149 页 validation 记录，不是本次 10 页注意力诊断重算值。

初版 eager 回放因前缀不匹配中止，并保留失败日志和两页临时 NPZ；其临时 NPZ不在最终可筛图目录中。最终 v2 launcher status 为 complete；20/20 重放前缀与各自 archive 一致，所有 raw/conditional 数值 finite 且 visual mass 为正。诊断期间无 NaN/Inf/OOM；v2 无 Traceback。`test_manifest_read=false`、`test_used_for_selection=false`。

## 产物位置

- 本地筛选目录：`D:/yangky/glmocr_mask_viz/glmocr_line_mask_v2_full_20260922_1940_validation_20260923/`
- 总览：`contact_sheet.png`；单张原分辨率并排图：`individual/comparison-01-...png` 至 `comparison-10-...png`
- 选择/框/预测索引：`comparison_index.json`、`comparison_index.csv`；原始注意力：`attention/*.npz`；原图：`images/*.jpg`
- 远端 v2 原始回放：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/diagnostics/glmocr_line_mask_v2_attention_val_20260923_v2/`
- 远端 eager 失败回放保留于对应无 `_v2` 的 diagnostics 目录。

## 图例补充（2026-09-23）

按筛图需求，对上述10张并排图及 `contact_sheet.png` 使用已归档的 attention NPZ 重新渲染，增加数值色条；没有重放模型、重算 attention 或更改样例。每组左右共用本组动态范围 `0–max(baseline_raw, routed_raw)`，色条标出 0、25%、50%、75% 和组内最大值的 raw attention 数值；颜色使用 Inferno，图例在浅色底上合成并注明透明度规则 `alpha = sqrt(value / max) × 150/255`。不同样例仍各自按组内最大值缩放，故颜色不用于跨样例绝对比较。更新后的图保存在相同本地筛选目录；renderer SHA256 `ED5473E1771B9D362F9C9806157B28A55149643DCEE327DBE7D6D2C91CA59752`。`test_manifest_read=false`、`test_used_for_selection=false`。
