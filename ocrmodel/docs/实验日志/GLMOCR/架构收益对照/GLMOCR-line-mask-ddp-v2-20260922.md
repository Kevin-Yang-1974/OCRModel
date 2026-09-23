# 行级 patch mask v2 五卡 DDP

## 固定配置与初始登记

- 日期2026-09-22；分支glm-ocr-layout-mask-routing；base commit 452ea59d9c91f7ca8fbc093b51a9ce15f79f9d18，加本轮未提交新文件，不提交/推送。方案与全部loss定义见[方案](../../../LINE_MASK_DDP_V2.md)。
- smoke run ID `glmocr_line_mask_v2_smoke_20260922_1940`，正式run `glmocr_line_mask_v2_full_20260922_1940`；正式训练已完成并通过预注册 CER<=0.14 validation 目标，最终结果见文末。
- 入口tools/training/run_line_mask_ddp_a100.sh → train_line_mask_ddp.py；a100-yky、物理GPU0–4；tmux同run ID，Slurm job/array不适用。
- 源码快照`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/code/line_mask_20260922_v1/ocrmodel`；tar SHA256=`7a81ffbaa e0a5967bc72a9e147e0d8906db912d0e1ed6110fd6815b8865d1467`（去空格即完整指纹，后附机器核验值）。
- 正式run实际使用修复后的`code/line_mask_20260922_v4/ocrmodel`，source tar SHA256=`838c06d66617a001210731043ac735dc5d240cb6cc46148168c3fb8841ff4c26`；v1条目保留为早期smoke快照，不作为正式结果源码指纹。
- 产物根`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/<run ID>`；独立缓存、日志、checkpoint，既有产物不覆盖。
- train manifest full2159 SHA256=`1016198040944e39329712eb2a7bdfe6db7526134b91d750c83afa91d854f9b3`；validation sparse24的149页 SHA256=`36ec845875e1ea18a48d4a523b6c5a3f007b46e2a07de0cafd2c26139929a348`。seed42、官方页面划分，未声称书手级隔离；未读取test。smoke固定取前10train/5validation，仅工程验证。
- 起点为trained decoder LoRA checkpoint-3000：`/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/training_runs/glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000`。官方model revision ca5d8b3e287e52589e37c28385d9655ee4372f9d。
- 单训练臂、新head随机初始化；baseline同冻结起点、不加路由，非另一个等预算训练臂，不把本轮变化归因到单个loss。没有VAE或旧MLP同期消融；不得声明特定组件的独立贡献。
- 正式：8epoch/3456更新，5rank真DDP、per-device1、累积1、global5；LR head2e-4/decoder0，warmup172、完整cosine到0.1×，AdamW wd.01、clip1；headFP32/冻结骨干BF16；LoRA rank8 alpha8冻结；trainable仅head，参数数以protocol.json为准。smoke1epoch/2更新，warmup1。
- loss: BCE1+Dice1+location.2+area.2+transition.2+stop.05+empty.2；auxiliary_weight0/layout0；update门随机Linear初始化、首步强制更新，无残差geometry gate。
- fast/4M/math-SDPA、1536 greedy；hard mask threshold.5、bias1、所有decoder层、prefill不注入；最后层hidden预测下一forward空间区域，label shift2。
- epoch2/4/8完整149页validation选CER最小；第7epoch前由epoch6预测head刷新训练侧routed特征，GT文字teacher forcing，mask无GT反馈。
- test_manifest_read=false；test_used_for_selection=false；locked test未读取。validation选点、CER/I/D/S、EOS/触顶/循环、loss关键点、checkpoint finite及异常结果均在文末追加。

## 已完成检查

本地compileall、git diff --check通过；本地无torch，未声称本地PyTorch测试通过。A100环境CPU张量测试3项通过：scan/step等价及无未来泄漏、梯度有限、小型行目标可优化。首次pytest因未配置libcudnn.so.9加载路径而收集失败，补项目NVIDIA库路径后通过；没有GPU实验产物，该环境错误不计训练失败。

磁盘启动前可用183GiB；缓存预计几十GiB，具体以实际写入为准，不删除其他run。GPU0–4观察均0%，正式启动仍重新admission。

## 启动失败与修复（追加）

初始tar机器指纹为`7a81ffbaae0a5967bc72a9e147e0d8906db912d0e1ed6110fd6815b8865d1467`。

1. `glmocr_line_mask_v2_smoke_20260922_1940`：GPU admission通过，torchrun导入PyTorch因旧anandasky CUPTI优先加载报`undefined symbol: cuptiActivityEnableDriverApi`；退出1，未进入DDP/训练、无checkpoint/metrics/validation。旧日志和launcher_status.json保留，不可作训练恢复点。修复为系统CUDA库优先、其余组件库追加。
2. `glmocr_line_mask_v2_smoke_20260922_1940_r2`：同配置，仅库搜索路径修复；启动因默认临时目录无可用空间/写入位置报`No usable temporary directory`，退出1，仍未训练、无checkpoint/validation。旧产物保留不可复用；不清理他人磁盘，后续TMPDIR显式写入个人/data3本run.tmp。
3. `glmocr_line_mask_v2_smoke_20260922_1940_r3`：应用上述两项环境修复，重新admission后启动。数据、模型、loss、DDP和预算与初始smoke相同；实际源码为初始快照叠加run_line_mask_ddp_a100.sh修复，最终launch脚本SHA另行追加。状态待核验，test仍未读取。

heartbeat ID `mask-ddp` 绑定当前任务；初始5分钟。所有失败未产生可用训练结果，不能解读为head数值崩溃。

## r3 smoke 完成与闭环 smoke 扩展

`glmocr_line_mask_v2_smoke_20260922_1940_r3` complete：10train/5validation，5卡真DDP，2更新；五rank同step的同步后grad_norm完全一致（step1=12.59042835、step2=4.10696650，clip前），loss有限，checkpoint已产生。5页CER=0.0648078372，仅smoke小集合，**绝非149页验收**。两步仍有低IoU/偏大mask面积，不能把工程通过说成定位收敛。

新增`glmocr_line_mask_v2_smoke_20260922_1940_r4`：同10/5、相同loss/学习率/起点，扩为2epochs/4更新，在epoch2前加入预测mask+KV teacher-text训练特征刷新，以覆盖正式第7epoch的闭环路径；补source_group/image_path隔离检查及checkpoint保存再载入逐tensor一致性检查。预算不同于r3，是额外工程验证，不是对照收益实验。独立code/line_mask_20260922_v2/ocrmodel，source-v2 tar指纹另附；产物/log名称用r4，状态待完成后补。test仍未读取。

## r4闭环故障及r5修复

- r3完整smoke指标补齐：reference chars1327，I/D/S=51/3/32，EOS5/5、触顶0、循环0；没有locked test、没有正式selection资格。
- r4源码tar SHA256=`af8f592ffb9747a16f69562c44c77cc4f95c2dc222a9c0095c267d3ded71a11f`。r4在epoch1训练/validation结束后，epoch2 routed_cache阶段rank4 CUDA OOM（申请2.96GiB，已用36.72GiB），退出1；epoch1 checkpoint与日志保留，不能作正式结果。根因证据：Transformers5.3 `GenerationMixin.prepare_inputs_for_generation`只在传next_sequence_length时裁切；手动KV循环未传该参数，继续送入整个prefix，KV追加/attention显存膨胀。修复为prefill传prompt长度，decode明确传1并检查shape，logits_to_keep=1。未调低4M分辨率、未删页、未改变1536验收预算。
- 新run `glmocr_line_mask_v2_smoke_20260922_1940_r5`，独立快照`code/line_mask_20260922_v4/ocrmodel`；tar SHA256=`838c06d66617a001210731043ac735dc5d240cb6cc46148168c3fb8841ff4c26`。2epochs/4更新+epoch2闭环缓存；其余训练/数据配置同r4。另将patch key归一化移到encode每页一次，并以归一化key池化反馈；这是head的最终定义，CPU三项测试重跑通过，不声称与旧smoke数值等价。显式init_process_group device_id以消除rank设备映射警告。
- r5运行结果待追加；仅r5完整闭环smoke通过后，才以相同v4代码启动正式run。代码v3仅做CPU测试、没有GPU训练run。
- 全量图像内容SHA256审计：train2159条/2158个唯一图像hash（train内部一对重复，保留全量并披露）；validation149条/149唯一hash；train与validation图像hash交集0，train2159页均有line_index。没有读test，也没有据此声称近重复或书手级隔离已证实。

## r5完整通过；正式全量任务启动

r5 complete：五卡DDP执行4更新，10页无路由缓存与10页预测mask闭环缓存均完成；2个epoch checkpoint通过finite及保存重载逐tensor一致检查。epoch2的5页生成CER=0.06556141672946496，I/D/S=51/4/32、reference chars1327、EOS5/5、触顶0、循环0。无NaN/Inf/新OOM/Traceback；这是工程smoke，未达到正式149页验收证据标准，不能以此声称达标。没有test，所有产物保留。

正式run `glmocr_line_mask_v2_full_20260922_1940` 已于约19:48派发；2026-09-22 19:49 Asia/Shanghai启动核验：tmux仍在，status=running/phase=cache，五rank的progress均前进，缓存已过约50页。物理GPU0–4 admission全部<50%；使用已通过r5的v4代码与全新随机head，不续接smoke权重。

正式protocol.json核对：train2159/validation149、world_size5、DDP=true、head参数501892、3456更新、warmup172；冻结LoRA SHA256=`ada4cdd3bb1f2f417d1c0a68054dc61e70ab59f215b3bc2018074b4f2fdcd5d5`；launcher SHA256=`832489632edd38baca28baa4c2a75353eb2147f26006d0746b5747a86f841700`。每个Python源文件的SHA256也记入protocol。torch2.8.0+cu128/transformers5.3.0；seed、loss、LR、调度与本日志固定配置一致。

正式log为`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/line_mask_full_20260922_1940.log`；产物根为本节正式run对应training_runs子目录；tmux同run，Slurm不适用。本地已下载`D:/yangky/glm-ocr-assets/line-mask-ddp-20260922/full-protocol.json`。

本次启动核验作为健康检查1；heartbeat `mask-ddp`已切换正式run，5分钟后做健康检查2，若五rank继续进展且无异常，立即更新同一个heartbeat为每1小时。当前阶段为训练特征缓存，尚无正式优化/validation结果；后续epoch2/4/8选点及149页CER/I/D/S/EOS/触顶/循环仍待生成。test_manifest_read=false/test_used_for_selection=false，locked test未读取。

## 健康检查2与降频（2026-09-22 19:53）

19:53:09 CST核验：tmux session `glmocr_line_mask_v2_full_20260922_1940`存在（19:48:25创建）；`status.json`=running/phase=cache（19:48写入，属stale字段，与log尾部一致故不构成不一致）；`launcher_status.json`不存在即在跑；五rank progress均为cache且持续前进（rank0/1/2/3/4 index=455/461/462/463/469，时间戳19:53同步）；`cache-unbiased`已落盘485个文件共4.7GiB，log行数持续增长。cache进度口径：train 2159页按world=5分片，每rank约432页，故index≈455/461/462/463/469表示该rank分片已接近完成、正在进行尾部（存在尾部重试/回退，非停滞）。log尾部为逐页JSON，无预填异常关键词命中；正式错误扫描（NaN/Inf/OOM/Traceback/NCCL/异常退出）为空。

物理GPU0–4（仅查询允许集合）瞬时utilization=100/89/100/0/100%，这是本run五进程各自的稳态计算负载，不是与其他任务争卡的空闲判据；admission发生在19:48启动时且通过。`/data3`可用178GiB（98%已用），本run缓存预计数十GiB，未删除其他文件。

结论：健康检查1与2均通过，无NaN/Inf/OOM/Traceback/NCCL/异常退出，无停滞证据。按规则自此降频为每1小时一次，绑定同一监控任务，不创建重复automation。仍处于训练特征缓存阶段，尚无正式优化与validation结果；epoch2/4/8 149页选点与CER/I/D/S/EOS/触顶/循环待生成。test_manifest_read=false/test_used_for_selection=false。

## cache完成、baseline 149页跑完、训练启动（2026-09-22 20:18–20:19）

阶段流转（按入口实际顺序，`status.json`不写`empty_validation`中间态，故该事实由产物推断）：
1. 20:08 训练特征缓存完成：`cache-unbiased` 2159文件/20GiB，五rank最终index=2155–2158（覆盖0–2158全量），progress停写属正常。
2. 20:08 进入 `baseline_validation`（入口第3阶段）：同一起点、不加路由，完整149页自由生成。
3. 20:19 `baseline/summary.json`写出=complete，status转为`training`/epoch1。

**baseline 149页结果（不加路由，同起点，非oracle）**：pages149、reference chars41654、CER=**0.16992365679166466**、I/D/S=1948/1082/4048、exact_page_rate0、EOS148/149、触顶1页、循环1页（loop_rate0.0067114）、无token budget变化、无删页。coverage由入口自身校验（`validation coverage mismatch`）通过，合并后149条即全部验证页。作为参考锚点：本轮head必须显著优于它并向0.14靠近才谈得上路由有收益。

**训练已启动**：`status.json`=running/training/epoch1（20:19:01写入，心跳更新正常）；`metrics-rank{0..4}.jsonl`均存在并每步追加，已到step7；五rank同step的`grad_norm`完全一致（step6=8.672424316453934、step7=7.255163669586182，clip前），确认DDP同步梯度；loss≈4.23–4.50有限。起点阶段`iou≈0.0`且`mass_ratio`偏大（5–28），是随机初始化head的预期早期状态，不作定位结论。

其中`location`项≈8.0–8.2，恰等于`log(patch数)`（patch约3200–3600），即预测分布≈均匀、该交叉熵项尚未提供定位信息；`transition≈0.693=log2`（二分类无信息）、`dice≈0.96`（面积近似下的退化值）同属早期信号，需在后续checkpoint复核是否下降。

全文件风险扫描（`nan|inf|oom|traceback|error|nccl|killed|exception|abort|fail`，排除progress行）为空；`launcher_status.json`不存在即在跑。GPU0–4瞬时40/100/39/100/37%，属本run五进程自身负载；`/data3`可用163GiB。本地已归档`D:/yangky/glm-ocr-assets/line-mask-ddp-20260922/baseline-summary.json`。

**监控说明（更正）**：Claude Code CLI没有桌面Codex式的持久`automation/heartbeat`，只有会话内定时任务（本会话CronCreate ID `5e39ac7e`，每小时07分），不写盘、会话结束即失效、无`automation_update`接口（改频率只能删除重建）。跨会话持久监控需外部计划任务，否则训练期间需保持本会话存活。此更正不影响上文任何训练事实。

## epoch2选点：149页CER 0.13533（首个达标点，非最终结论）

21:15核验：`validation-epoch2/summary.json`写出complete、`selection.json`同时写出，status=running/training/epoch3。选点step=864、checkpoint=`epoch-2.pt`、`validation_sha256=36ec8458...`与protocol一致。

**epoch2 完整149页（预测mask，无reference同步）**：pages149、reference chars41654、CER=**0.1353291400585778**、I/D/S=**868/786/3983**、exact_page_rate0、EOS149/149、**触顶0、循环0**（相对baseline的触顶1/循环1已改善）、无删页、无token budget变化、coverage校验通过。对不路由baseline（CER0.16992365679166466）为**−0.0346（相对降20.4%）**，字面达到`acceptance.criterion`的CER<=0.14且`eligible=true/passed=true`。

必须同时记录的限制：①这是**中间epoch2**，epoch4/8仍在跑，最终头部按各epoch CER最小重选，此点不等于最终验收成绩；②149页validation已被历史多次使用，是development validation，不是locked test，不得据此下独立结论；③single arm无等预算消融，不能把降幅单独归因给某个loss项；④`acceptance.passed`是脚本按当前best自动写的字面字段，最终验收仍需epoch8后按同一协议复核。

下载：`selection-epoch2.json`、`validation-epoch2-summary.json`、`metrics-rank0.jsonl`已归档到`D:/yangky/glm-ocr-assets/line-mask-ddp-20260922/`。

## loss配比与dice/IoU下降慢的原因（metrics-rank0逐行分析，step1157）

名义权重：BCE1 / Dice1 / location0.2 / area0.2 / transition0.2 / stop0.05 / empty0.2，即**名义占比BCE35.1%、Dice35.1%**，两者是最大项。

但按实际加权贡献，**晚期loss已被location主导**（step1157 raw dice0.5762/location4.0637/bce0.1198/area0.2533）：加权占比约为 **location50.4%、dice35.7%、bce7.4%、area3.1%、transition2.9%**，stop+empty合计0.3%。这与早期（step1：location37%、dice22%、bce32%、area5%）相比，bce与area迅速让位，location成为唯一大项。

**IoU单调上升、dice并非"降不下来"**：iou在step1=0→step101=0.066→step201=0.373→step1101=0.540，近300步均值0.499（max0.850），当前step1157=0.551；dice loss由0.96降到约0.576（口径为逐token平均，含空/EOS行；均值被拉高恰与`mass_ratio`正相关，近300步corr=0.59，对应掩码面积仍偏大）。

因此"dice下降慢"的直接原因是**两项都没有独立驱动力**：唯一学习K的空间项的location被BCE/Dice淹没(dilution)。这属于设计时已披露的权衡（方案文档原话：候选logits始终有直接监督以降低保持门阻断新行学习的风险），不是运行故障；两项仍在下降，只是斜率被location盖过。

按"若确有问题才提"的约束，下一轮若要提升空间收敛，候选旋钮是升location权重或对BCE/Dice做行内归一化，但**本轮不自动改动、不另开昂贵实验**，须由用户决定。

## 巡检（21:18，epoch3训练中，无异常）

status=running/training/epoch3/step1220；tmux在；`launcher_status.json`不存在；五rank `metrics-rank*.jsonl`均为1220行、同step `grad_norm`一致（step1216=1.482832431793213）→ DDP同步正常；loss 1.32–1.95有限；风险扫描为空。`epoch-1.pt`/`epoch-2.pt`已各6076759字节，finite与保存重载一致性检查由入口内部强制。

空间项继续收敛：step1216五rank `iou`=0.423/0.700/0.601/0.742/0.576，`dice`=0.609/0.361/0.493/0.421/0.538，`mass_ratio`=2.25/1.74/1.46/1.60/2.43；相较epoch2均值（iou≈0.499、mass≈3.53）IoU上行、面积比接近1–2.4，与上一节诊断（IoU单调上升、dice受空行口径与面积偏大影响）一致。

GPU0–4 utilization仅11–18%属**预期**：本阶段是head在已缓存特征上训练、骨干前向未参与，显存19–32GiB为已加载模型常驻，不是训练空转或卡死判据；`/data3`可用163GiB。下一节点为epoch4的149页选点（预计约21:40）。

## epoch4选点：149页CER 0.134177（当前选中点）

21:48写出`validation-epoch4/summary.json`并更新`selection.json`；22:18巡检确认status=running/training/**epoch6**/step2510，`epoch-{1..5}.pt`均在（各6076759字节），五rank `metrics-rank*.jsonl`同步2506行、同step grad_norm一致（step2504=1.4060918092727661），风险扫描为空。

| 点 | 149页 CER | reference chars | I/D/S | EOS | 触顶 | 循环 | 选中 |
|---|---|---|---|---|---|---|---|
| baseline（无路由） | 0.16992365679166466 | 41654 | 1948/1082/4048 | 148 | 1 | 1 | — |
| epoch2 (step864) | 0.1353291400585778 | 41654 | 868/786/3983 | 149 | 0 | 0 | 曾选中 |
| **epoch4 (step1728)** | **0.1341767897440822** | 41654 | 889/666/4034 | 149 | 0 | 0 | **当前** |

`selection.json`=epoch4/`epoch-4.pt`、`validation_sha256=36ec8458...`、`eligible=true`、`passed=true`、`test_manifest_read=false`、`test_used_for_selection=false`。相对baseline为**−0.0357（相对−21.0%）**。

如实判断：epoch2→epoch4仅改善−0.00115，边际很小且两epoch都贴着0.14上沿；substitution仍高达4034（baseline4048，几乎未降），本轮降低主要来自insertion（1948→889）与deletion（1082→666）。同源历史参照（同validation149/checkpoint-3000/4M/1536/fast）：GT整行CER0.12469、GT 3–5字窗口0.13694——当前预测mask的0.13418**优于窗口参照的0.13694，但仍未达到GT整行的0.12469**，即"预测定位"尚未追平"完美定位"，这是定位损失仍有空间的量化线索，不是已达上限的证明。

下载：`selection-latest.json`、`validation-epoch4-summary.json`、`metrics-rank0.jsonl`已归档`D:/yangky/glm-ocr-assets/line-mask-ddp-20260922/`。

## 下一阶段预告

epoch6将于step2592结束，随后立即进入 **epoch7前的`routed_cache`闭环特征刷新**：用epoch6的预测head、在生成KV路径中以teacher-forced文字对全部2159页重算特征（mask全部来自预测，无GT反馈）。该阶段是纯缓存、期间无优化metrics，**不得据此判定停滞**，应看`cache-routed-epoch7`文件数与progress行。之后epoch7/8训练，最后epoch8再做149页选点作为最终验收点。

## 阶段变化：进入epoch7前routed_cache闭环刷新（22:21起，23:18核验）

epoch6于step2592正常结束（五rank metrics均为2592行），`epoch-6.pt`（6076759字节）22:21落盘；`status.json`=running/**routed_cache**/epoch7（22:21写入）。23:18核验：tmux在、`launcher_status.json`不存在、风险扫描为空、五rank progress全为`routed_cache`且持续推进、`cache-routed-epoch7`已617个文件。

进度量化（按progress文件计数，非估计）：rank0–4已处理118/122/125/127/125页，合计**617/2159=28.6%**，历时约57分钟；按当前速率约**10.8页/分钟**，剩余1542页预计还需约143分钟，即routed cache约**01:40前后完成**（外推估计，非承诺）。该阶段比无路由缓存（20分钟）慢约9倍，原因是每页要走生成KV路径并对每个文字token施加head预测mask，属预期开销，不是异常。

**该阶段没有新的优化metrics属正常**：`metrics-rank*.jsonl`停在2592行（epoch6结束值），不得据此判定停滞；判据是`cache-routed-epoch7`文件数与progress行。磁盘需盯：可用空间由163GiB降至157GiB（routed cache预计再写入约20GiB）；GPU0–4瞬时38/34/39/39/100%。若之后epoch7/8训练约各17分钟、epoch8验证约22分钟，整体预计**约02:40前后完成**（外推估计）。

已按规则在阶段变化时复核频率：该阶段为纯缓存、耗时长且无metrics，维持每1小时一次；下一次重点确认617→更大的文件数与progress行是否继续单调前进。

**巡检00:18（routed_cache进行中）**：无异常。`cache-routed-epoch7`=1412文件/14GiB（65.4%），rank0–4 index=1385/1401/1417/1443/1399；近一小时795页≈13.25页/分钟，剩余747页约需56分钟，**完成时间外推修正为约01:15**（前次01:40偏保守；仍属外推非承诺）。metrics仍停2592行（epoch6结束值）、status仍为22:21写入的routed_cache/epoch7，两者都是该阶段正常表现，不是停滞。风险扫描为空；`/data3`可用150GiB（本阶段约10MB/页，剩余约需7.5GiB）；GPU0–4瞬时34–39%均匀负载。tmux在、`launcher_status.json`不存在。

## 阶段变化：routed_cache完成，epoch7训练已启动（01:17）

01:18核验：`cache-routed-epoch7`=**2159/2159文件**（0–2158全量覆盖），该阶段自22:21起历时**176分钟**（无路由缓存时为20分钟，约9倍），与逐token施加预测mask的每页开销一致。status随即由routed_cache/epoch7翻为 **running/training/epoch7**（01:17写入，心跳恢复更新）；五rank `metrics-rank*.jsonl`由2592推进到2612，step2608同step `grad_norm`一致（1.9467642307281494）→ 闭环缓存后的DDP仍正常；loss1.50–1.87有限；风险扫描为空。

闭环特征已生效的初步迹象（epoch7 step2608，五rank）：`iou`=0.6676/0.5539/0.3631/0.7504（rank1/4/3/2），`mass_ratio`=1.41–2.45，`dice`=0.368–0.615，均优于epoch6末（iou≈0.55、mass≈2.3–5.3）的同口径读数——**但这是训练侧teacher-forced特征上的读数，不是自由生成CER**，不能据此预判最终成绩，只有epoch8的149页自由生成才作数。

时间外推（非承诺）：epoch7按历史27分钟/epoch，约01:44结束；epoch8约02:11结束；随后epoch8完整149页验证约22分钟，**整体预计约02:33完成**。`/data3`可用143GiB。GPU0–4瞬时7–25%属head-only训练期预期。下一节点为epoch8选点与最终验收核验。

## 正式run完成与最终验收核验（2026-09-23 02:01完成，02:18核验）

`status.json`=**complete/complete**、`best_validation_cer=0.12846305276804149`、`test_manifest_read=false`；tmux session已正常退出（五卡显存0MiB/利用率0%），`launcher_status.json`不存在即**无异常退出**。epoch-7.pt(01:34)、epoch-8.pt(01:51)均在，`epoch-{1..8}.pt`各6076759字节；五rank `metrics-rank*.jsonl`均3456行、覆盖step1→3456、epoch1→8；全文件风险扫描为空。

**最终选中点：epoch8 / step3456 / `epoch-8.pt`**

| 点 | 149页 CER | reference chars | I/D/S | EOS | 触顶 | 循环 | exact |
|---|---|---|---|---|---|---|---|
| baseline（无路由，同起点） | 0.16992365679166466 | 41654 | 1948/1082/4048 | 148 | 1 | 1 | 0 |
| epoch2 (step864) | 0.1353291400585778 | 41654 | 868/786/3983 | 149 | 0 | 0 | 0 |
| epoch4 (step1728) | 0.1341767897440822 | 41654 | 889/666/4034 | 149 | 0 | 0 | 0 |
| **epoch8 (step3456)** | **0.12846305276804149** | 41654 | **796/531/4024** | 149 | 0 | 0 | 0 |

**独立复核（不是引用训练脚本的摘要）**：把 `validation-epoch8/predictions-rank{0..4}.jsonl`下到本地，用项目自身的 `layout_ocr.metrics.levenshtein_alignment`+`levenshtein_error_counts`重算，得到 pages149、reference chars41654、character_errors5351、I/D/S=796/531/4024、**micro CER=0.12846305276804149**，与summary逐位一致；149条page_id唯一无重复、无缺失，rank分片30/30/30/30/29合计149与protocol一致。`epoch-{2,4,8}.pt`载入检查：29 tensor、**501892参数**、`all_finite=True`、epoch/step字段正确。`selection.validation_sha256`=`protocol.validation_sha256`=validation manifest磁盘实测sha256=`36ec845875e1ea18a48d4a523b6c5a3f007b46e2a07de0cafd2c26139929a348`（三者一致）。`test_manifest_read=false`、`test_used_for_selection=false`。

**验收判定：通过（有保留）**。相对无路由baseline为 **−0.04146（相对−24.4%）**，CER=0.12846 ≤ 0.14，`acceptance.eligible=true/passed=true`。保留四条：①149页validation已被历史多次使用，是**development validation，不是locked test**，不得据此下独立结论；②单训练臂 + 单次baseline，无等预算对照，**不能把降幅归因给某个loss或组件**；③`substitution`=4024几乎未动（baseline4048→4024，两轮仅−24且epoch4曾回弹到4034），全部改善来自 `insertion`1948→796 与 `deletion`1082→531，即**漏字/多字被修掉，字级替换未改善**；④同源参照下 GT整行CER=0.12469 仍优于本预测mask的0.12846（但已优于GT窗口的0.13694）——预测定位尚未追平完美定位。

**训练侧收尾指标（metrics-rank0，末100步均值）**：loss1.5327、bce0.1148、dice0.4539、location4.3116、area0.2617、transition0.2248、stop0.0359、iou0.6325、mass_ratio2.5306；末步step3456 iou0.7251、mass_ratio1.901、lr2.0000041181857276e-05（cosine已收到≈0.1×）。无任何非有限值。与早前诊断一致：iou自0持续上行（末100步min0.184/max0.866），location仍是loss大头（约50%），dice受空行口径与残留面积偏大（mass_ratio约2.5）拖累。

下载归档`D:/yangky/glm-ocr-assets/line-mask-ddp-20260922/`：`selection-final-epoch8.json`、`status-final.json`、`validation-epoch8/`（summary+5个predictions，合计149条）、`metrics-rank0.jsonl`（3456行）。

**下一步建议（不自动执行）**：剩余误差主体是字级替换（4024/5351≈75%），且定位已接近但未达GT整行水平。若继续推进，性价比最高的方向是：①在**同一冻结骨干**下做等预算对照，隔离"闭环特征刷新(routed_cache)"与"head结构"各自的贡献；②针对替换误差检查是否为低频/异体字问题（本轮summary的 `low_frequency_k*`/`r2_k*`字段因 `train_character_counts` 传空Counter而全为0/null，**不可用于低频分析**，需补传真实训练字符频次）；③若要locked test，必须先固定selection再执行，不得据test回调。上述均需用户决策，本轮不自动另开昂贵实验。

## 补真实训练字符频次并做离线频率诊断（2026-09-23，只读、不训练）

**起因**：正式run所有summary里的 `low_frequency_k{1,3,5}_*`、`r2_k*` 字段全为0/null，因为 `aggregate_ocr_metrics` 被传了空 `Counter()`（`train_line_mask_ddp.py`第117行等5处调用点均如此，其中4处在旧脚本/锁定测试工具内）。这是**记录缺陷，不是指标本身不可用**；本次补真实频次后按同一validation149页重算，未重新推理、未读test、未参与任何选点。

**数据**：train manifest `mthv2_layout_page_v1/train/manifest.char.jsonl`（2159页，SHA256 `10161980...4f9b3`，138MB）字段含 `page_text`；实际训练字符数 **722207**、唯一字符 **6072**。

**逐字替换(micro CER/recall)按训练频次分桶**（分母为该桶在149页中出现的参考字符数）：

| 点 | CER | 编辑总数 | I/D/S | k≤1 recall | k≤3 recall | k≤5 recall | 高频(>5) recall | 训练中未出现 recall |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.16992 | 7078 | 1948/1082/4048 | 0.5938 (64) | 0.5621 (153) | 0.6124 (209) | 0.8787 (41363) | 0.6341 (82) |
| epoch2 | 0.13533 | 5637 | 868/786/3983 | 0.5938 | 0.5556 | 0.5981 | 0.8875 | 0.6220 |
| epoch4 | 0.13418 | 5589 | 889/666/4034 | 0.5781 | 0.5490 | 0.5933 | 0.8892 | 0.6341 |
| **epoch8** | **0.12846** | **5351** | **796/531/4024** | **0.6094** | **0.5556** | **0.5981** | **0.8926** | **0.6341** |

**结论1（稀有字不是本轮瓶颈）**：低频/未见字的样本量极小——149页中k≤1仅64字、k≤3仅153字、k≤5仅209字、训练中完全未出现仅82字，四类合计不到参考字符的1%。其recall在0.55–0.63之间、且**从baseline到epoch8几乎没有变化**（k≤3甚至0.5621→0.5556）；同期高频字recall 0.8787→0.8926。即路由带来的−0.0415主要来自高频字，**不是稀有符号识别问题**；R2类稀有符号结论需专门协议，不在此149页口径内。

**结论2（替换误差≈异体字/字形变体归一化，而非学习失败）**：epoch8 top20混淆对占全部编辑的 **40.7%**、占全部替换的 **54.2%**；典型为 `増→增`、`縁→緣`、`䖏→處`、`閒→間`、`爲→為`、`逺→遠`、`𫝹→念`、`𢙉→惱`、`眀→明`、`㑹→會`——**参考是异体/古字、预测是通行正体**。仅在等长对齐页面上统计：epoch8有40页等长，落在"训练中频次为0的参考字"上的替换只有**6个**（baseline为2个），说明这些替换的参考字在训练集中**是见过的、只是登记为异体**。需要明确的是：`増`、`縁`、`䖏` 等字在epoch8里比baseline错得**更多或持平**（如 `増→增` 524/541、`縁→緣` 178/179），因此结论限于"该部分误差与定位无关、更像词表/字形归一化问题"，**不能断言训练把模型推得更差**。

**结论3（异体归一化的量级）**：若仅从编辑计数中扣除epoch8 top10混淆对（这是上限式假设、**不是可达成绩**，因为其中含 `薩→蓮` 这类真错），CER从0.12846降至0.0830；对baseline同口径为0.16992→0.12561。即这些变体对占了路由前误差的很大一块，是**数据/词表口径**问题而非掩码路由问题。

**结论4（长页触顶这一具体failure已被路由修掉）**：baseline最差页 `mthv2_tkh_0001_021_28_11` 为 edits1008、触顶1536 token（insertion922）；epoch8同页降到 edits66、424 token、13 deleted。epoch8全部149页**触顶0页**，故再无"触顶后over-generation"贡献编辑。编辑集中度：epoch8 top1页占4.0%、top5占13.1%、top10占19.6%、top20占31.4%、top30占41.6%——仍处长尾，但最大单页贡献已由14.2%降到4.0%。

**代码改动（只加能力、不改机制）**：`train_line_mask_ddp.py` 增加 `train_character_counts()` 并在baseline与各epoch evaluate中传入真实频次，后续新run的summary将自动带有效低频字段；`evaluate_line_mask_locked_test.py`/`merge_line_mask_locked_test.py` 与locked test launcher增加可选 `--train-manifest`（锁定测试merger将据此填有效频次）。**既有已完成run的各summary保持原样不回写**，历史0值字段按"当时未传频次"解读，本次修正以本文档追加记录。新增离线诊断工具 `ocrmodel/tools/evaluation/diagnose_line_mask_frequency.py`（只读既有predictions，不推理、不读test），输出归档 `frequency-diagnosis.json`。

## 正式run健康检查2与当前状态（2026-09-23）

第二次健康检查通过后按规则将同一 heartbeat `mask-ddp` 从每5分钟切换为每1小时；没有创建重复监控。正式run仍在 `routed_cache/epoch7`，这是 epoch6 head 对完整2159页做预测mask+KV teacher-text 特征刷新，期间 metrics 保持2592行是预期行为。

只读核验结果：tmux会话存在；训练父进程和五个torchrun worker仍在；`cache-routed-epoch7`文件数由前次约1412继续到约2101/2159（97.3%），progress rank最近 index约2070/2101/2107/2128/2069，文件时间持续更新；无 `launcher_status.json` 失败标记。step2592五个rank最后记录均finite，IoU约0.46–0.74、mass_ratio约1.2–4.1；日志风险扫描未见NaN/Inf/OOM/Traceback/WorkNCCL/ChildFailed/CUDA error。GPU查询严格限于0–4，均有训练进程和显存占用。

`status.json` 的写入时间仍是旧的 epoch7 cache 标记，不能单独作为进度依据；本次以进程、cache文件更新时间、rank progress和log尾部联合判定为健康。缓存结束后应立即检查 status 阶段是否转为训练，再在 validation 阶段读取 epoch8/最终 summary；在正式149页成绩出现前不得宣称 CER 通过。test 仍未读取。

## 正式结果：epoch8 selection-locked validation 完成（2026-09-23）

正式run `glmocr_line_mask_v2_full_20260922_1940` 已完成，`status.json`=`complete`，tmux与五卡进程正常退出；没有launcher失败标记，最终日志风险扫描未见NaN/Inf/OOM/Traceback/WorkNCCL/ChildFailed/CUDA error。训练五个rank各写满3456行（step 1–3456），epoch-8 checkpoint保存并可加载。

固定协议核对：完整train=2159，既定validation=149，validation SHA256=`36ec845875e1ea18a48d4a523b6c5a3f007b46e2a07de0cafd2c26139929a348`；五卡真DDP、global batch=5、8 epochs/3456 updates、epoch7前预测mask闭环特征刷新；test_manifest_read=false、test_used_for_selection=false，test未读取。validation predictions本地核对149条、149个唯一page_id、无重复无遗漏。

| 点 | step | CER | reference chars | I/D/S | EOS | 触顶 | 循环 |
|---|---:|---:|---:|---|---:|---:|---:|
| baseline，无路由 | — | 0.16992365679166466 | 41654 | 1948/1082/4048 | 148/149 | 1 | 1 |
| epoch2 | 864 | 0.1353291400585778 | 41654 | 868/786/3983 | 149/149 | 0 | 0 |
| epoch4 | 1728 | 0.1341767897440822 | 41654 | 889/666/4034 | 149/149 | 0 | 0 |
| **epoch8，最终选中** | **3456** | **0.12846305276804149** | **41654** | **796/531/4024** | **149/149** | **0** | **0** |

最终 `selection.json` 选中 `/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_line_mask_v2_full_20260922_1940/epoch-8.pt`，`acceptance.eligible=true`、`passed=true`。相对无路由baseline绝对下降`0.0414606040`（约24.4%）；相对GT整行oracle `0.1246939069` 仍高`0.00376915`，因此预测mask已达到目标但尚未追平真值定位上限。替换错误仍为4024，几乎未降于baseline的4048；主要收益来自插入`1948→796`和删除`1082→531`，后续若继续研究应优先看替换/视觉内容错误与行切换，而非继续把oracle结果当成可部署上限。

有限性核验：epoch-8 checkpoint含29个tensor、501892个参数，远端`torch.isfinite`全通过；五份metrics JSON均3456行且非有限值为0。最终149页生成全部EOS，`generation_limit_hits=0`，`loop_pages=0`、`loop_rate=0.0`。本地归档目录 `D:/yangky/glm-ocr-assets/line-mask-ddp-20260922/formal-20260923/`，包含protocol、selection、epoch-8 checkpoint、baseline/epoch2/epoch4/epoch8 summaries、149页分片预测与五份训练metrics；关键文件逐一与远端SHA256一致。

本轮验收为development validation，不是独立test泛化证明；没有新增VAE、旧MLP或loss单变量同期对照，不能把收益拆分归因给某一损失项。正式结果核验完成后，`mask-ddp` heartbeat已停用，避免留下过期监控。
