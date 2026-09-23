# 敦煌／地方志扩展 test：baseline vs 行级 mask

## 预注册协议

- **Run ID**：`glmocr_dunhuang_local_plus_new77_line_mask_20260923`；日期 2026-09-23；分支 `glm-ocr-layout-mask-routing`，启动时 HEAD `ea73f182bc7ebe69f7c7386024b3b97bf89e392a`。无 Slurm job ID。
- **入口**：`ocrmodel/tools/evaluation/run_line_mask_dunhuang_extended_test_a100.sh`；数据准备、worker、merge 脚本同目录。远端代码驱动副本位于 run 根 `code/`。
- **远端产物根**：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/locked_tests/glmocr_dunhuang_local_plus_new77_line_mask_20260923`；tmux `glmocr_dunhuang_new77_mask_20260923`。
- **新增来源**：清华云盘共享包 `2026.08.20敦煌古籍样本-新增77/20260820`，77 张 JPG 与77个同名 `.RGN` 区域坐标文件，压缩包 SHA256 `3c0c34824c1e7d90593d6c47b5a4e441ac2e26a8115620e32ba55459c979dd39`。RGN 不包含 OCR 转写，因此新增77页纳入扩展推理清单，但不计 CER/I-D-S。
- **既有协议**：`dunhuang_local_gazetteer_q32_v1_portable`，train/validation/original test=`240/80/59`，seed42、Q32；原 test 含敦煌19页、地方志40页。新增77页的 source group 与现有 train/validation/test 均无重叠。原始三划分文件未改写；新 run 使用独立的136页扩展 test manifest，其中59页有参考文本、77页无转写。
- **test 使用边界**：test manifest 为本次推理已读取（`test_manifest_read=true`），不用于选点、阈值或后处理（`test_used_for_selection=false`）。既有59页曾在早期 Q32 实验中作为测试使用，本结果标注为已暴露测试集上的锁定比较，不视为首次独立外推。
- **两臂**：baseline（共享 checkpoint-3000 decoder LoRA，关闭 line-mask runtime）与 `epoch8/step3456`（相同 backbone/LoRA，加上由 MTHv2 validation 锁定的行 mask head）。两臂使用相同页序、prompt 和生成预算；不训练、不调参。
- **固定推理**：whole-page image + `Text Recognition:` prompt；fast processor；4M max pixels；greedy、1536 max new tokens、BF16、SDPA；seed42。五个指定物理 GPU 0–4各运行一个单卡 worker，按轮转方式均分136页（每片27或28页），同一 worker 对两臂逐页推理。
- **模型起点**：GLM-OCR revision `ca5d8b3e287e52589e37c28385d9655ee4372f9d`；backbone LoRA 来自 `/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/training_runs/glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000`；mask head 来自 `/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_line_mask_v2_full_20260922_1940/epoch-8.pt`。epoch8/step3456 在 MTHv2 149页 validation 上已锁定，CER `0.12846305276804149`。
- **协议文件**：远端 `protocol.json`、`inputs/expanded-test-manifest.jsonl`；执行中状态 `launcher_status.json`、每片 `progress.json` 与 `worker_status.json`、逐页预测 JSONL 和 logs。

## 结果

正式评测已完成。两臂在136页扩展 test manifest 上均完成推理；CER/I-D-S只统计原有59个有转写页面，新增77页仅报告生成行为。覆盖、指标、finite、日志扫描、SHA及本地归档见下文。


## 完成核验（2026-09-23）

- **状态**：launcher complete/exit0；5个单卡worker各完成（shard 28/27/27/27/27页）。五卡并行推理，不是DDP训练。
- **覆盖**：manifest 136个唯一page_id；两臂各136条，重复/缺失0/0。59页有转写，77页无转写；本地归档复核通过。
- **总体59页 / 14,195参考字符**：baseline CER 0.136879182811，I/D/S 519/404/1020；行mask CER 0.138429024304，I/D/S 553/396/1016。CER变化+0.001549841493，mask略差。
- **分域**：敦煌19页 baseline CER 0.166958698373 (253/10/404)，mask 0.167709637046 (259/6/405)；地方志40页 baseline 0.125098039216 (266/394/616)，mask 0.126960784314 (294/390/611)。括号为I/D/S。
- **新增77页**：77 JPG+77 RGN，RGN只有区域坐标、没有OCR转写；两臂EOS 77/77，触顶0、循环0，故不报告准确率。
- **协议**：新增source-group与train/validation/原test重叠为空；canonical splits未改。test_manifest_read=true、test_used_for_selection=false。原59页在早期Q32实验已作为test使用，故不宣称首次独立test。
- **完整性**：所有shard两臂head_finite=true，summary有限；launcher/worker/merge异常扫描空。
- **指纹**：新样本ZIP 3c0c34824c1e7d90593d6c47b5a4e441ac2e26a8115620e32ba55459c979dd39；扩展manifest b2e78dfe61d37f7110abb44ce8e9c230c00edace4a0978358ad26ed4f22b2c99；protocol fb474362954536d799ee38db81cffeb6db7c26196d1d64622a68ec336b2bf6f7；summary ffa80b01e81d20964913bc0105f3a36b4d64bfd17f80d0281c17c06f3b1d17b5；baseline预测 2d0377a78c2c3a602dd3ea6a90a027520d106efd971cd23ff9a89187a318a2dc；mask预测 c4311f63e14cd1d5dbf05d7dac2f9e5759dfeaec557adf56b6c5e2f2e5c5609f。
- **本地归档**：D:/yangky/glm-ocr-assets/dunhuang-new77-20260923/locked-test/。新增77页目前只能评估生成行为；需有对应转写才可评价识别准确率。


## 完整清单与驱动指纹补充

- train manifest SHA256: 00ae8c30fc12046586cce836897af26b7a701a749fa10ff805bb4ae8022fb29d
- validation manifest SHA256: e20d2f9b07e535ccfab03c95a8f81222f75ee4f5a793e27dabb28338f48316e2
- 原 Q32 test manifest SHA256: da94dbd7e372e5e206c7c340be881e40f11eabff1b12df26668f9a983e7f782d
- 有标注59页扩展子清单 SHA256: 75003905009b49513843f1396326aebf9b02f81429d655530fdef19707a085b8
- 新增77页无转写子清单 SHA256: efc17b98bb9e1b9744667ad454140c856845fb883334ba863edd521dcbd17e7b
- 扩展136页清单 SHA256: b2e78dfe61d37f7110abb44ce8e9c230c00edace4a0978358ad26ed4f22b2c99
- 预处理脚本 SHA256: add0dd478f22a11907d7a2cba1a81a120ddd21747c29feb0fff2aea23d555e60
- 推理脚本 SHA256: 3f5a49cea9fadae7f829d94f4fe39c16b702aa27958e117550a356c56a210680
- 合并脚本 SHA256: f762bb7ce342936cda90be501bfd7e89aa1afda94f551ae56360392e36087147
- A100 launcher SHA256: 794b42e5fc1d28d8c18d930889218fe50d5e88e1c3d9f4372e072c51cac60cc2
- 原始共享目录链接：https://cloud.tsinghua.edu.cn/d/d4d46c99faec44e2bfa4/


## 事后差异归因（仅解释已锁定结果，不用于选点或调参）

59页有转写子集上，CER从baseline 0.13687918到mask 0.13842902，绝对变化 +0.00154984（约+0.155个百分点，相对+1.13%）；净多22个编辑。I/D/S差为 +34/-8/-4：插入增加34、删除减少8、替换减少4。两臂均59/59 EOS、触顶0、循环0；平均生成长度从298.34增至299.49 tokens，按去空白后的预测字符平均每页约多0.71字符。故表现是轻微插入代价超过少量删除/替换收益，不是发散、触顶或循环崩溃。

逐页按与CER相同的空白剥离和Levenshtein口径重算：18页编辑数下降、15页上升、26页不变；46/59去空白预测串不同。净增22个编辑全部来自一个地方志页（localgazetteer_4530_00000005_00002，+22），其余58页编辑数净和为0；该来源组8页净增22。敦煌19页净增3，地方志40页净增19，整体回归集中在少数样本/来源，不是所有页一致变差。

不确定性很大：按domain分层做30,000次配对page bootstrap，mask-baseline CER差95% percentile区间为[-0.00251,+0.00617]；按5个source-group（敦煌2组、地方志3组）分层重采样，区间为[-0.00084,+0.00483]，都跨0。来源组只有5个，cluster区间只是粗略描述；再考虑旧59页曾在早期Q32实验中作为test使用，本次只能说该已暴露小测试上baseline点估计略优，不能认定mask有稳定真实退化。新加77页没有转写，不进入这些统计。

机制上的可检验解释：训练监督是MTHv2 teacher-forced文本前缀下每个字符对应的整行mask；自由生成时，上一生成前缀驱动递归mask，再将概率图以0.5阈值二值化，向所有decoder层的命中视觉key统一加1.0 attention-logit bias（prefill无bias，首个输出token不受控）。这可能让模型更愿意持续读取行内视觉证据，帮助少删字，同时在个别版式上多读/多写，造成插入。该解释与I/D/S及长度变化相符，但本次prediction JSONL没有保存逐步mask面积、阈值命中率、行定位IoU或attention变化，因此不能仅凭CER判定根因是分布偏移、mask定位还是bias剂量。MTHv2完整800页locked test曾小幅改善（baseline 0.32235648，mask 0.31893128），与Q32差值方向不同，提示效果依赖数据分布；两套数据绝对CER不可横向比较。

后续若要定位而不是直接调参，应先在未用于选点的独立且有转写页上记录每步预测mask的面积/行IoU/命中率，并按页面列出routing前后I/D/S；对新增77页，须先补OCR转写才有准确率证据。本次test结果不用于改head或重选checkpoint。


## 模型加载核验（2026-09-23，读取本次运行产物；未重跑）

- 五个 worker log 均显示模型权重加载进度到 510/510，随后各自完成27或28页；日志未见 missing/unexpected keys、shape mismatch、CUDA、Traceback等模型加载错误。
- run protocol 固定模型目录为revision ca5d8b3e287e52589e37c28385d9655ee4372f9d，config.json 的 model_type=glm_ocr、architectures=GlmOcrForConditionalGeneration。加载代码从该本地revision使用 local_files_only=True，BF16和SDPA。
- 五个worker使用同一 checkpoint-3000 decoder_lora.safetensors，文件SHA256 ada4cdd3bb1f2f417d1c0a68054dc61e70ab59f215b3bc2018074b4f2fdcd5d5；加载实现先注入rank-8模块，随后对权重键集合与张量形状逐一严格匹配，且先检查LoRA权重finite。五片均正常结束，未触发键/形状校验异常。
- mask selection与checkpoint均指向epoch8/step3456；实际远端epoch-8.pt SHA256 d56023f270fd51812b58272141f8903bbd2ca1b437b8565b20ab5d12c7f59cac 与protocol相同。worker先检查payload epoch/step，再以 strict=True 加载head state dict并逐张卡检查finite；5/5 shard summary均记录 head_finite=true。
- 每个worker只加载一次backbone/LoRA，然后在同一model对象上先运行baseline（routing disabled），再启用mask；故两臂共享同一底座与LoRA。launcher最终 complete/exit_code=0，两臂预测coverage均136页。
- 加载链路相关源码指纹：evaluate_window_mask_routing.py SHA256 383bfc30c4d1fa1095af01fc9a93c1d10c8fd9deafc358d9d37cd3787ac5c7da；line_mask_head.py SHA256 a8c3b8a45ab45ad96a96424e49d74825f1f2e051e986e0cf578c02edb49d4be9；line_mask_runtime.py SHA256 a16adb93abb94d91f181cc6c9369ef7fc0ded6d43eb73eea21b15f9f3f13176e。
- **结论**：运行证据支持模型、LoRA与mask head均按锁定路径正确加载；没有模型加载错权重或加载失败的迹象。此核验依据远端运行日志、协议/权重指纹和加载代码，不是运行结束后GPU内存快照。
