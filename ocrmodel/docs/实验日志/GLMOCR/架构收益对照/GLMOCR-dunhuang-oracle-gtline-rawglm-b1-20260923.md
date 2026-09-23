# 敦煌／地方志 GT 行 mask bias=1.0 oracle 对照

## 标识、数据与协议

- Run ID glmocr_dunhuang_oracle_gtline_rawglm_b1_20260923；日期2026-09-23；完成，launcher exit 0。
- 分支 glm-ocr-layout-mask-routing；启动时HEAD ea73f182bc7ebe69f7c7386024b3b97bf89e392a。
- 外置入口 D:\yangky\glm-ocr-assets\dunhuang-new77-20260923\oracle-gtline-rawglm-b1\driver\run_oracle_gt_line.sh；SHA256 1eb34535d7726e014e3f560b4a916fcc9f7672e08881ca2f6552c876d5f78908。推理和合并脚本SHA分别5cb8d321f275fb0551b1704ad6d93dd28d37a5266a48e86a3c56d4b20aa8aded、3f8d3f5f8bbeb1e9610157cc5fb97b3a8359918014623a6926be3ba929e2db3b。
- 远端产物 /data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/locked_tests/glmocr_dunhuang_oracle_gtline_rawglm_b1_20260923；tmux glmocr_dh_oracle_raw_20260923；Slurm不适用。本地归档 D:\yangky\glm-ocr-assets\dunhuang-new77-20260923\oracle-gtline-rawglm-b1。
- 数据 dunhuang_local_gazetteer_q32_v1 全量59页有OCR真值，敦煌19、地方志40、5来源组；test manifest SHA256 75003905009b49513843f1396326aebf9b02f81429d655530fdef19707a085b8，canonical source SHA256 da94dbd7e372e5e206c7c340be881e40f11eabff1b12df26668f9a983e7f782d。新77张无转写图全部排除；59页此前已在扩展test评估。
- 复杂度按参考字符数×行区域数分三分位：简单稀疏20（DH10/LG10）、中等20（DH9/LG11）、高复杂度19（DH0/LG19）；此池无高复杂度敦煌页。
- test_manifest_read=true；test_used_for_selection=false。无训练、选点、checkpoint选择、阈值或后处理调整；这不是独立新test。

## 权重、实验臂与执行

两臂均使用原始GLM-OCR revision ca5d8b3e287e52589e37c28385d9655ee4372f9。config SHA256 4e1daf0d8a3f63e58960ac14bcb58b7be96758cad231fb7a1e5fec60f42dcd8c；model.safetensors SHA256 a16eb0de98d199293371c560f95f83130d2a2c9612449df16839f08ff9498815。推理脚本调用AutoModelForImageTextToText.from_pretrained并明确不注入LoRA；5个worker权重加载进度均达100%。没有训练checkpoint，可训练参数0。

Baseline baseline_raw_glm关闭路由。Oracle oracle_gt_line_bias1_raw_glm只新增GT行框mask及GT转写同步器，单调lookahead=16；cached decode全部decoder层对当前行视觉token的attention logits加1.0 bias，prefill不加。Backbone输入仍为整页图像和prompt，但路由器读取GT文本，故oracle不可部署。

共用设置：整页图像、Text Recognition: prompt、fast processor、4,000,000 max pixels、1536 max new tokens、greedy、BF16、math-SDPA、seed42。5个单卡worker分别使用物理GPU0–4，round-robin分片；两臂页池、预算相同。按用户要求豁免GPU准入。
训练配置均不适用：eval-only，无train/validation、优化器、LR、warmup/schedule、步数、batch/梯度累积、loss权重或loss曲线。无checkpoint，finite checkpoint检查不适用。

## 结果

| 指标 | Baseline raw GLM-OCR | GT行mask bias=1.0 | 差 |
| --- | ---: | ---: | ---: |
| CER | 0.1368791828 | 0.1383585770 | +0.0014793942（+0.148个百分点） |
| 页数 / reference chars | 59 / 14,195 | 59 / 14,195 | — |
| 字符错误 | 1,943 | 1,964 | +21 |
| I / D / S | 519 / 404 / 1,020 | 548 / 398 / 1,018 | +29 / −6 / −2 |
| Exact page | 0/59 | 0/59 | — |
| EOS / max-token hit / loop | 59 / 0 / 0 | 59 / 0 / 0 | — |
| 平均生成token | 298.339 | 299.220 | — |

按域：敦煌19页/3995字，baseline CER 0.166959、I/D/S 253/10/404，oracle 0.168711、265/7/402，+7 edits；地方志40页/10200字，baseline 0.125098、266/394/616，oracle 0.126471、283/391/616，+14 edits。
来源组 edits 差（oracle−baseline）：敦煌BD01015 17页−10；敦煌BD01028 2页+17；地方志4505 12页−12；地方志4530 8页+24；地方志5405 20页+2。
复杂度结果：简单稀疏20页，CER 0.188045/0.193441、edits 453/466（+13）；中等20页，CER 0.148183/0.147233、edits 624/620（−4）；高复杂度19页，CER 0.114323/0.115908、edits 866/878（+12）。

30,000次按domain分层配对page bootstrap、seed42，CER差95%区间[-0.0029172909,+0.0066933374]，跨0。只有5个来源组（2敦煌、3地方志），该区间不是来源级泛化保证。
路由诊断：17,466/17,595 decode steps施加bias；missing box 0；129步pointer越过GT标注文本。预测mask IoU/MAE不适用。59页两臂唯一完整；五分片12/12、12/12、12/12、12/12、11/11。Launcher exit 0；错误扫描未发现NaN/Inf、CUDA/OOM、Traceback、NCCL或异常退出；输出指标有限。

## 结论与边界

点估计未显示GT行mask+bias=1.0有收益，CER略高、编辑多21，增量主要为插入。两个域都略差，复杂度和来源组方向混合，区间跨0：这是负面机制信号，不足以认定普遍退化。GT框和GT转写同步不可部署，也不是学习mask head精度上限的直接等价物；不能据此断言学习行mask head必然失败。

本次raw baseline与D:\yangky\glm-ocr-assets\dunhuang-new77-20260923\locked-test\predictions-baseline.jsonl在59个重叠页文本逐页相同。此前扩展test protocol标注baseline使用checkpoint-3000 decoder LoRA，而本次脚本和protocol明确是raw GLM-OCR。预测相同不证明权重相同；该观察应留作跨run权重核查点。

## 归档校验

protocol.json SHA256 052ecb500360ded590cf891b863c7ad4527330e084a05803115e7e3b51bcc920；manifest SHA256 75003905009b49513843f1396326aebf9b02f81429d655530fdef19707a085b8；summary SHA256 5f1147dc00d0664a05c7ab22be4eb01170aed228827c87a1a2da1a738456b2fb。baseline predictions SHA256 984c1730da1e8e3d901afff3ccbf080822eaec5fc3306a3d93789be9f8fd0e84；oracle predictions SHA256 82109b212cca0c9cbdb3fccdf90efcf5b9527181c14ac0d653e82f967bd9a48e。分片预测、status、driver、launcher/worker/merge日志和协议副本留存于归档目录。此次新建实验日志并只向登记表末尾追加，没有修改计划代码或计划文件。
