# GOT2 整页布局查询合成预训练方案

> 更新日期：2026 年 8 月 11 日
>
> 文档性质：Visual Layout Query Adapter（VLQA）的数据与训练执行方案
>
> 当前状态：整页生成与审计及 A100 工程链路已经打通；`layout_overfit_20260811_110317` 未通过 P1 两样本可拟合性检查，并暴露出首步极端 logit 与 bbox 饱和；显式初始化修复已在本地完成、待 A100 复测；以下参数和比例仍是首轮候选

## 1. 目标与边界

本方案使用 HTML/CSS 等可控排版工具生成带精确布局真值的整页图像，对 GOT2 内部的布局 queries 进行预训练。正式模型输入是整页截图，bbox、阅读顺序和书写方向只作为训练监督与评测标签，不作为推理输入。

合成预训练主要解决三个问题：

1. 真实页面缺少区域框、方向和阅读顺序标注；
2. 仅依赖 OCR loss 不能保证 learnable queries 形成明确布局语义；
3. 小样本真实页面不足以覆盖多列数、多方向和多模板组合。

合成数据不能单独证明真实场景收益。若提升只存在于合成页面、固定模板或浏览器字体域，必须判定为合成域差异，不能据此声称真实古籍或通用符号识别已经改善。

当前不再采用“合成整页后裁回 line-level，再显式输入 bbox”的主路径。现有 line crop 可以作为页面内容素材嵌入 HTML，但生成后的整页图像整体送入 GOT2。

## 2. HTML 页面生成器

### 2.1 推荐实现

首轮采用 Python 驱动 Playwright/Chromium：

```text
Python sampler
  -> HTML/CSS template
  -> Chromium deterministic rendering
  -> full-page screenshot
  -> DOM bounding boxes and reading-order labels
  -> page-level manifest
```

HTML/CSS 负责页面尺寸、边距、列数、栏间距、横排/竖排和区域组合；Pillow/OpenCV 负责截图后的纸张背景、透印、污渍、模糊、形变和扫描噪声。首轮应固定 Chromium 版本、字体包、viewport 和 `deviceScaleFactor=1`，并在读取 DOM bbox 前等待 `document.fonts.ready`。

竖排页面可以使用 `writing-mode: vertical-rl` 或 `vertical-lr`，但阅读顺序必须根据页面视觉坐标与模板规则显式生成，不能直接把 DOM 节点顺序当作真值。截图像素尺寸与 `getBoundingClientRect()` 的 CSS 坐标必须通过自动测试核对。

### 2.2 内容来源

合成页面分为三档：

| 数据档 | 页面内容 | 主要用途 |
|---|---|---|
| `S0-html-text` | 浏览器字体渲染的文字、符号和简单结构 | 验证 schema、query 槽位、方向与排序监督，扩大模板覆盖 |
| `S1-html-crop` | 将真实单行、单列或区域 crop 作为 `<img>` 嵌入 HTML 页面 | 保留真实字形和退化特征，是联合训练的主要合成内容 |
| `S2-hard` | 在 S0/S1 整页上加入背景、透印、污渍、模糊、形变、遮挡和区域边界扰动 | 鲁棒性训练与压力测试 |

纯 HTML 字体页面适合布局 query 预热，但不能独占主训练分布。真实 crop 重排页面用于缩小字体和扫描域差异。公式、表格、谱面或其他符号场景可在最小文字版页面生成器稳定后，再分别接入 MathJax/KaTeX、HTML table、Verovio/SMuFL 或对应领域渲染器；不得在首轮同时扩张所有场景。

### 2.3 页面与区域真值

设第 $i$ 个合成页面为：

$$P_i=\mathrm{Render}\left(T_i,B_i,\{C_{i,k}\}_{k=1}^{R_i}\right),$$

其中 $T_i$ 为模板，$B_i$ 为背景，$C_{i,k}$ 为第 $k$ 个内容区域。DOM 给出区域框：

$$\mathbf b_{i,k}=(x_{0i,k},y_{0i,k},x_{1i,k},y_{1i,k}).$$

所有 bbox 按截图宽高归一化。区域按模板定义的实际阅读规则生成 `reading_order`，而不是事后依据字符串或文件名猜测。页面级转写按真值顺序拼接：

$$\mathbf y_i=\mathrm{Concat}\left(\mathbf y_{i,\pi_i(1)},\ldots,\mathbf y_{i,\pi_i(R_i)}\right),$$

其中 $\pi_i$ 是页面阅读顺序。区域分隔符、换行和格式标记必须与 GOT2 的目标输出及评测规范统一，不能在训练与评估阶段分别处理。

每条页面记录至少包含：

```json
{
  "page_id": "page_xxx",
  "source_group_id": "book_or_corpus_xxx",
  "template_id": "vertical_rtl_08col",
  "image": "images/page_xxx.png",
  "page_size": [1024, 1024],
  "page_text": "按真值阅读顺序拼接的页面转写",
  "layout_source": "html_synthetic",
  "regions": [
    {
      "region_id": "region_000",
      "content_id": "content_xxx",
      "bbox": [x0, y0, x1, y1],
      "reading_order": 0,
      "writing_direction": "vertical_rtl",
      "text": "区域转写",
      "valid": true
    }
  ]
}
```

这些布局字段只用于训练辅助头和评测。正式推理调用不得读取 manifest 中的 bbox、顺序或方向。

## 3. 防泄漏与反捷径规则

必须先按 `source_group_id`、书籍、版本、馆藏或内容来源划分 train/validation/test，再生成页面。具体要求如下：

1. 同一 `content_id` 的所有字体、位置、模板和退化版本只能属于一个 split。
2. 同一真实页面拆出的全部 crop 只能属于一个 split。
3. 页面模板、背景、字体和内容独立采样，不能让特定书籍、符号类别或文本内容绑定固定列位。
4. 同一内容应在本 split 内出现在多个合法位置和模板中，降低“内容预测位置”的捷径。
5. `S-OOD` 必须保留未见模板、未见列数范围、未见背景和未见扰动组合。
6. 浏览器字体必须做字形覆盖检查，拒绝缺字方框、`.notdef` 或意外字体回退造成的伪标签。
7. 页面文本和区域文本必须经过一致的 Unicode 规范化；规范化规则写入配置并随结果保存。

模板随机化不能破坏语义合法性。例如，竖排从右到左的列序必须与页面转写一致；不能为了增加扰动而随机打乱标签后仍称为正确页面。

## 4. 分阶段训练

合成页面可动态生成，因此用 optimizer steps、有效 batch size、唯一 `content_id` 曝光次数和随机种子控制预算，不以 epoch 作为唯一比较口径。

| 阶段 | 数据 | 可训练参数 | 目的 |
|---|---|---|---|
| `P0` 原始基线 | 真实或固定合成整页 | 无新增模块 | 记录原 GOT2 页面 CER、顺序错误、显存和速度 |
| `P1` query 布局预热 | S0＋S1，使用区域真值 | layout queries、query cross-attention、辅助头 | 按阅读顺序分配区域槽位，学习区域存在性、bbox 和方向；写回门固定为 0 |
| `P2` 合成联合训练 | S0/S1/S2 整页图像与页面转写 | VLQA、辅助头、`mm_projector_vary` | 打开零门控残差，联合优化页面 OCR 和布局监督；ViT 与 Qwen 先冻结 |
| `P3` 真实页面适配 | 真实整页为主、少量合成页为辅 | VLQA、projector、Qwen LoRA 或低学习率 decoder | 减少合成到真实的域差异 |
| `P4` 分辨率扩展 | 密集整页与超长页面 | VLQA、高分辨率 query 读取路径、可选 multi-crop | 仅在 16×16 定位不足时验证更高分辨率 |

`P1` 不使用布局字段作为模型输入。区域真值只进入损失函数。若实现中为了调试临时加入 oracle metadata，必须置于独立对照配置，不能混入端到端主模型。

首轮超参数只做小网格筛选，候选值必须写入配置而不是写死在文档。筛选可先用一个随机种子；正式比较至少运行三个种子。所有消融保持相同数据曝光和优化器步数。

## 5. 目标函数

### 5.1 P1 布局预训练

布局预热使用：

$$\mathcal L_{\mathrm{layout}}=\lambda_{\mathrm{obj}}\mathcal L_{\mathrm{obj}}+\lambda_{\mathrm{box}}\mathcal L_{\mathrm{box}}+\lambda_{\mathrm{dir}}\mathcal L_{\mathrm{dir}}.$$

区域 query 按阅读顺序槽位监督；无区域槽位使用 `no-object` 标签并屏蔽 bbox 和方向损失。bbox 使用 L1 与 GIoU 的组合，方向使用交叉熵。阅读顺序由“第 $k$ 个 query 对应第 $k$ 个真值阅读区域”的目标分配监督，不增加可由 query ID 平凡完成的顺序分类头；成对顺序准确率和 Kendall's $\tau$ 仍用于评测。有独立顺序头的配置只用于无序 query 对照，并在区域匹配后计算损失。

P1 成功只说明 queries 能从页面视觉中定位和组织区域，不代表已经改善 OCR。

### 5.2 P2 以后联合训练

$$\mathcal L_{\mathrm{total}}=\mathcal L_{\mathrm{ocr}}+\mathcal L_{\mathrm{layout}}.$$

各损失权重在验证集上选择，同时记录 OCR 与辅助分支的损失、梯度量级和零门控参数。若布局损失持续下降而页面 OCR 不变，不能继续用更大权重强迫模型依赖布局，应先检查融合路径和任务相关性。

### 5.3 防止把普通视觉 adaptor 误判为布局模块

必须设置无布局监督但参数量相当的 query/resampler 对照。只有完整布局监督模型稳定优于以下对照，才能把收益归因于布局查询：

1. 原始 GOT2；
2. 等参数量普通视觉 adaptor；
3. 相同 queries、相同融合结构但令全部 $\lambda_{\mathrm{layout}}=0$ 的模型。

## 6. 验证集与指标

验证集至少固定为四组：

| 子集 | 输入 | 主要用途 |
|---|---|---|
| `S-ID` | 已见分布的合成整页 | 检查训练内布局与 OCR 学习 |
| `S-OOD` | 未见模板、列数、背景和扰动的合成整页 | 检查是否记忆固定排版 |
| `R-ID` | 与训练域相近但来源隔离的真实整页 | 检查合成到真实迁移 |
| `R-OOD` | 新书手、新版本、新馆藏或新符号场景的真实整页 | 检查小样本跨域泛化 |

主要 OCR 指标为页面 CER、页面编辑距离、完整页面精确匹配率和稀有符号召回率。布局指标为区域召回率、bbox IoU/mAP、方向准确率、成对阅读顺序准确率和 Kendall's $\tau$。同时报告可训练参数、峰值显存、吞吐量和单页延迟。

诊断实验包括：

1. 将 VLQA 残差门设为 0；
2. 去掉所有布局辅助损失；
3. 分别比较 bbox-only、bbox＋方向，以及在相同区域辅助监督下的有序与无序 query 目标分配；不设置没有区域损失支撑的 `order-only` 主配置；
4. 打乱 query 顺序 embedding；
5. 比较 $16\times16$ 与 $64\times64$ query 特征；
6. 可视化同一页面在正确模型和无监督 query 对照中的注意力区域；
7. 将同一内容放到不同合法版面，检查 OCR 与 query 定位是否稳定。

旧版“正确 metadata、全零 metadata、随机 metadata”测试只适用于显式 region-token adapter（旧 PCLA）对照，不适用于不接收 metadata 的 VLQA 主模型。

## 7. 小样本协议

领域级少样本按真实标注页面数或页面比例限制新书手、新版本、新馆藏或新场景数据；稀有符号级 K-shot 限制低频符号在训练页面中的实例数。合成页面不能绕过 K-shot 限制：若某个真实稀有符号只有 K 个独立来源实例，对其位置和模板做无限复制仍只算 K 个内容实例，并必须报告其增强曝光次数。

正式小样本比较必须满足：

1. 相同来源隔离划分；
2. 相同真实页面标注预算；
3. 相同合成内容来源和曝光次数；
4. 相同优化器步数和解码设置；
5. 至少三个随机种子；
6. 同时报告页面 OCR、布局指标和推理成本。

## 8. 判定与停止条件

出现以下任一情况时，不继续扩大合成数据规模：

1. `S-ID` 布局指标良好，但 `S-OOD` 明显下降，说明模型记忆模板或固定列位。
2. 合成页面 OCR 提升不能在 `R-ID/R-OOD` 复现，说明存在合成域差异。
3. 完整布局监督 queries 不优于无布局监督 queries，说明辅助布局目标没有有效贡献。
4. VLQA 不优于等参数量普通视觉 adaptor，不能把额外容量收益写成布局创新。
5. query attention 集中在背景、边框或固定模板装饰，而不是内容区域，说明出现视觉捷径。
6. 只有切换到高分辨率分支才能拟合训练集，但真实页收益仍不稳定，应停止增加分辨率和计算量。
7. 页面 OCR 因辅助损失显著恶化，且调低权重后仍不能恢复，应移除对应辅助任务。

## 9. 当前工程前置条件

截至 2026-08-11，以下内容已完成首版本地实现：

1. `generate_synthetic_layout.py`：`S0-html-text`、`S1-html-crop`、`S2-hard` 页面规划，Playwright DOM 渲染、整页截图、实际字体检查和 provenance；
2. `audit_synthetic_layout.py`：图片/HTML 路径、尺寸、哈希、bbox、顺序、页面转写、实际字体及跨 split 内容/来源泄漏检查；
3. `layout_page_dataset.py`：整页图像、页面 OCR 标签和定长 query 监督 collator；
4. `layout_query.py`：有序 queries、object/bbox/direction 头、二维位置编码、预测前 LayerNorm、零门控写回、显式完整初始化及以 FP32 计算的 BCE/L1/GIoU/方向损失；
5. GOT2 的 256-token 接线和 OCR/layout 总损失；
6. `train_GOT_layout.py`：`P1` 查询预热和 `P2` 联合训练的开发入口，以及原始/续训 safetensors 的 VLQA 键完整性审计；
7. `run_layout_a100.py` 与 `verify_layout_checkpoint.py`：环境、数据、GPU 0、初始 logit 尺度、P1/P2 单步训练、权重有限性和模型重载的受限 A100 smoke 编排，以及固定为 P1、2 条记录、200 steps 的 `overfit` 诊断。

本地已实际完成 `S0` 与 `S2-hard` 的 Edge 截图和全量 manifest 审计；`S1` 的文件 URI、orientation 与源图哈希通过 CPU 单元测试。生成器已使用 Chromium 平台字体接口拒绝未允许 fallback，但正式字体文件、版本和哈希仍待随数据环境锁定。当前仍未完成：

1. 正式字体包文件/版本/哈希锁定和真实 crop 内容清单；
2. 初始化修复版 P1 两样本 `overfit` 的 A100 复测；
3. validation loader，以及页面 CER、bbox、方向和阅读顺序的统一评测；
4. `A0/A1/A2/A4` 的同预算启动器；
5. `P3` 真实页适配、`P4` 64×64 特征路径及任何效果验证。

A100 run `layout_pilot_20260811_023528` 已经验证 CUDA、整页 batch、P1→P2 权重衔接、checkpoint 保存和完整模型重载。该 run 仅含 2 页且每阶段重复 500 epoch；P1 最后 5 步平均总损失为 11.4，P2 最后 5 步平均总损失为 18.8376，当时又没有分项损失和 validation。因此它是工程链路证据，不是有效 pilot，也不能支持布局学习或 OCR 改善结论。

A100 run `layout_overfit_20260811_110317` 已使用新版 FP32 分项日志执行 P1 两样本 200 steps，`overfit_assessment.status=fail`。第 1 步 object/direction logit 绝对值上限均约为 1680，bbox 已饱和到约 0/1，表明异常先于优化过程出现；末 20 步 object accuracy 为 0.5375、bbox mean IoU 为 0。当前修复在原始 checkpoint 没有 VLQA 键时强制完整重置适配器，在完整 P1 checkpoint 上保留已加载权重，部分 VLQA 状态则直接拒绝；同时增加预测前 LayerNorm、小尺度 bbox 输出层初始化和 query/bbox raw-logit 诊断。该根因判断仍需修复版 A100 首步复测确认。

现有 `scripts/linelevel_dataset.py`、`train_GOT_linelevel.py` 和 `run_linelevel_smoke.sh` 只保留为既有工程链路与 line-level 诊断入口，不能作为新版整页 VLQA 的正式运行命令。现有 `ancientdoc_dataset.py` 可用于核对整页数据读取方式，但其历史 split 存在书籍级重叠，不能直接充当无泄漏主实验。

正式实现顺序固定为：

```text
HTML page generator                      [local smoke passed]
-> page manifest and split audit       [local smoke passed]
-> page dataset/collator               [A100 smoke passed]
-> minimal 16x16 VLQA                  [A100 forward/backward passed]
-> P1/P2 save and reload chain         [A100 engineering check passed]
-> FP32 component logging              [A100 diagnostic completed]
-> P1 two-record overfit gate          [first A100 run failed]
-> explicit init and scale guard       [local fix; A100 rerun pending]
-> validation loader and metrics
-> A0/A1/A2/A4 single-step smoke
-> small-scale synthetic pilot
-> real-page adaptation
-> optional 64x64 query path
```

前一项通过后再进入下一项。当前首先要验证 P1 能否拟合两个固定模板页面；`overfit` 未通过时不得启动 P2 或扩大数据。即使 `overfit` 通过，它也只是实现正确性检查，仍需 validation loader、未见内容/模板/方向和真实页面评测，不能写成模型性能结果。

## 10. 当前结论

HTML 合成页不再只是为 line crop 制造外部 bbox，而是直接作为整页视觉输入，用 DOM 真值监督布局 queries。主训练内容采用“HTML 控制版面＋真实 crop 保留字形”的混合方式，纯浏览器字体页面用于几何预热与模板覆盖。推理阶段只输入页面图像；bbox、方向和阅读顺序是训练标签或可选输出，不是外部条件。

该方案已经完成首版 A100 工程链路验证，但首次受控 P1 两样本诊断明确失败；初始化与尺度保护修复尚待 A100 复测，因此仍属于待实现诊断、待 validation 和待实验判定的候选。只有完整布局监督模型在真实整页、小样本跨域和统一预算消融中稳定优于原 GOT2、普通视觉 adaptor 和无监督 queries，才能保留 VLQA 作为有效结构候选。
