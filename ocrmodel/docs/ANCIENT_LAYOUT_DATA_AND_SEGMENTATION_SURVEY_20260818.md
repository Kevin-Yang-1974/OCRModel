# AncientDoc 版面数据与列/文本行分割调研（2026-08-18）

## 1. 结论先行

后续 C4–C6 可以取消 synthetic replay，直接在真实 AncientDoc 页面上使用版面监督。但不建议把所有外部数据直接混成一个训练集：中文古籍、东亚竖排历史文档和西文历史文档解决的是不同层次的问题。

推荐的最小可行路线是：

1. 以 `MTHv2` 作为中文古籍版面监督的第一来源。它同时提供文本行位置、转写、字符框和栏间/边界线，最适合把现有页面 manifest 扩展为列或区域监督。
2. 用 `DB-SegHist` 的设计作为中文历史文本行 teacher 的方法参考；该方法在 `MTHv2`、CHDAC 和 HDRC 上报告了强结果，但当前没有核验到官方可直接运行的代码仓库，因此不能称为已经可下载的预训练模型。
3. 工程上先实现一个 Detectron2 `Mask R-CNN` teacher，在 MTHv2 的真实标注上微调，输出每个候选列/文本区域的 `mask + bbox + confidence`。Mask R-CNN 的实例输出比单一语义 mask 更适合相邻竖列拆分。
4. 用 `HJDataset` 预训练或做低权重混合布局先验，用 `YALTAi`/Kraken 做人工可修订的 PageXML 制备链，用 cBAD 和 DIVA-HisDB 做通用历史文档几何 sanity check；这些数据都不能替代中文古籍真值。
5. C4–C6 使用新的 run prefix 和新的 validation/frozen-test 协议。旧 C4/C5/C6 的 replay 结果保留为历史结果，不得与新协议直接比较。

外部分割器只用于离线标注制备、伪标签、质量审计或独立对照；GOT2 正式推理仍只接收 `whole_page_image + ocr_prompt`，不接收外部 bbox、mask 或列裁剪。

## 2. 数据集核验

| 数据集 | 与中文古籍的关系 | 可用版面标注 | 访问/许可状态 | 建议用途 |
|---|---|---|---|---|
| **MTHv2**（TKH + MTH） | 最贴近中文历史文档；MTH 含 2,200 张互联网复杂历史文档图像 | 文本行位置与按阅读顺序转写；字符类别和字符 bbox；栏间/边界线段 | 官方仓库提供下载入口；README 说明仅对学术界免费、仅研究使用；未发现标准开源许可证文件 | **中文古籍布局监督首选**；边界线转列分隔，文本行转区域/行框 |
| **SCUT-CAB** | 题目即“复杂版式中国古籍文档布局分析”基准 | 论文确认存在布局分析标注 | 论文和 DOI 已核验；截至本次检索未核验到官方公开下载页、仓库或许可证 | 优先联系作者；在取得授权前不纳入正式数据 |
| **HJDataset** | 历史日文，东亚竖排和复杂层级高度相关 | 超过 250,000 个版面元素；区域 bbox、mask、层级结构、阅读顺序 | 标注可下载；图像受版权限制，需表单申请；未发现标准 LICENSE | 竖排区域/层级/阅读顺序预训练或低权重混合监督 |
| **IACC2022 CHDAC** | 中国历史文档文本行检测竞赛数据，SegHist 使用 | 文本行检测标注 | Springer 论文脚注给出官方竞赛站点 `https://iacc.pazhoulab-huangpu.com/`；本次未确认稳定公开下载和许可证 | 取得官方数据后作为中文历史文本行 teacher 的补充 |
| **ICDAR 2019 cBAD** | 西文历史文档，版面和文本行几何可迁移但脚本不同 | 历史文档文本行/基线相关标注 | Zenodo DOI `10.5281/zenodo.3568023`，CC BY 4.0，约 4.67 GB | 通用历史文本行分割预训练；不直接当中文列真值 |
| **ICDAR 2017 cBAD** | 西文历史文档 | 文本行/基线相关标注 | Zenodo DOI `10.5281/zenodo.1491441`，CC BY-SA 4.0，约 4.19 GB | 与 2019 cBAD 做跨来源几何 sanity check |
| **DIVA-HisDB** | 中世纪手稿，脚本和版式不同 | 精细像素级历史文档区域/文本行标注 | Zenodo DOI `10.5281/zenodo.19127869`，CC BY 4.0，约 1.30 GB；该记录是 2026 年重新归档版本，不应当作论文首次发表年份 | 语义/像素分割预训练和模型 sanity check |
| **YALTAi 数据集** | 西文手稿与早期印刷书；适合区域类别和矩形检测 | Segmonto 区域类别、矩形框；配套 Kraken/YOLOv5 工具 | Zenodo 记录 `10.5281/zenodo.6814770` 与 `10.5281/zenodo.6814769` 已由论文页面列出；下载前需读取对应记录的许可条款 | 小数据 object-detection teacher、人工修订工作流 |

### 2.1 MTHv2 的项目转换

MTHv2 的原始标注不是本项目当前 manifest 的最终格式，转换时应保留来源证据：

```json
{
  "image": "relative/page/path.png",
  "page_id": "mthv2_xxx",
  "source_group_id": "book_or_collection_xxx",
  "layout_level": "column_or_textline",
  "regions": [
    {
      "region_id": "region_000",
      "bbox": [0.10, 0.08, 0.18, 0.91],
      "polygon": null,
      "reading_order": 0,
      "writing_direction": "vertical_rtl",
      "label_source": "mthv2_line_or_boundary",
      "label_status": "human",
      "confidence": 1.0,
      "text": "optional"
    }
  ]
}
```

- MTHv2 文本行位置：保留为 `textline`，不要未经几何合并直接命名为 `column`。
- 栏间/边界线：可生成 `column` 的分隔线、列中心和列框；边界线不等于完整列框，需用页面边界和相邻边界线推导，并记录 `label_source`。
- 书籍、版本、馆藏或原始文档必须先分组再划分 train/validation/test；官方随机 `3:1` 划分不能直接作为本项目泛化协议。
- 只有人工标注或经过人工复核的标签才能作为 `label_status=human`；teacher 输出应使用 `pseudo`，不能伪装成真值。

## 3. 分割与布局模型

| 方法 | 原生输出 | 优点 | 主要限制 | 项目定位 |
|---|---|---|---|---|
| **DB-SegHist / SegHist** | 语义分割式文本行检测，可处理任意形状和高长宽比文本行 | 针对中国历史文档；在 MTHv2、CHDAC、HDRC 上报告强结果，并强调竖排/旋转鲁棒性 | Springer 页面标为非开放获取；未核验到官方代码/权重 | 方法参考和复现对照，不宣称现成 teacher |
| **Mask R-CNN** | 每实例 mask、bbox、类别和置信度 | 实例输出适合相邻列分离；Detectron2 工程成熟；2024 历史文本行比较中优于若干 U-Net 系方法 | 需要实例框/掩码标注；固定分辨率下细长列可能被截断 | **第一版古籍列 teacher**；在 MTHv2 微调 |
| **Kraken + YALTAi** | 区域检测、layout analysis、reading order、PageXML/ALTO 等格式 | Apache-2.0 Kraken；有 eScriptorium 人工修订链；适合把预标注变成可审核 PageXML | 现成模型不保证中文古籍适用；需要微调和几何后处理 | 数据制备工程备选，不是 GOT2 推理前置模块 |
| **dhSegment** | 像素级语义分割及 baseline/page/layout extraction | GPL-3.0，历史文档领域成熟，适合作为语义分割基线 | 相邻竖列实例需要 connected components、watershed 或投影拆分 | Mask R-CNN 的语义分割对照 |
| **Doc-UFCN** | 轻量 fully-convolutional 文档布局/文本行分割 | 论文强调多数据集预训练和轻量化 | 本次未核验到官方代码仓库或权重 | 论文级 baseline，暂不作为可下载 teacher |
| **DocLayout-YOLO** | 快速区域 bbox 检测 | AGPL-3.0；提供 DocLayNet/D4LA/DocStructBench 等现代文档 checkpoint，适合快速出候选框 | 现代 `text/title/table` 类别不等于古籍列；DocSynth-300K 规模大且域差明显 | 快速 bbox 初始化/审计，不做零样本中文古籍真值 |

## 4. 推荐 teacher pipeline

### 4.1 第一版：MTHv2 → Mask R-CNN → AncientDoc 伪标签

1. **标签标准化**：将 MTHv2 line/boundary 转为统一坐标系；同时保存原始点、推导规则和来源哈希。
2. **分组划分**：按书籍/版本/馆藏划分 MTHv2；禁止同一来源页跨 split。
3. **微调 teacher**：Mask R-CNN 输出 `mask + bbox + score`。首版只做 `column` 或 `textline` 单类别，避免类别定义和中文古籍样式混在一起。
4. **AncientDoc 推理**：整页输入，不裁成 line-level；保存所有候选及置信度，不直接覆盖人工/公开真值。
5. **几何审计**：检查框是否越界、面积是否过小/过大、相邻框重叠、列中心单调性、竖排方向一致性、阅读顺序冲突。
6. **人工抽检**：按书籍、版本、页面模板和预测置信度分层抽检；抽检结果单独记录为 `audit_status`，不能只用平均置信度代替人工质量。
7. **转换为 GOT2 manifest**：仅将通过阈值和审计的候选写入 `label_status=pseudo`；不通过的页面只保留 OCR 监督或进入待修订队列。

### 4.2 备选：Kraken/eScriptorium 修订链

Kraken 的 layout analysis、reading order 和 PageXML/ALTO 支持，适合“模型预标注 → 人工修订 → 导出结构化标注”的流程。建议用于少量高价值 AncientDoc 页面，形成高质量校准集，再反哺 Mask R-CNN。它是数据制备工具链，不是本分支的“两阶段 GOT2”推理结构。

### 4.3 不推荐的直接用法

- 不能把 DocLayout-YOLO 的现代 `text` 框直接命名为古籍 `column`。
- 不能把西文 cBAD/DIVA 的 reading order 规则直接迁移为中文竖排从右到左。
- 不能把 SegHist 论文结果写成已经下载的 checkpoint；当前只确认论文和数据集使用范围。
- 不能把 teacher 的伪框与人工框混在一个 `human` 标签字段中。

## 5. C4–C6 无 replay 协议草案

旧 `C4/C5/C6 replay` 口径应废弃，采用新的 run prefix（例如 `ancientdoc_layout_real_v1_*`），并为新协议重新选择 validation best 和 frozen test。建议保留一个匹配预算的无布局基线，但不再把 synthetic replay 当作 C5/C6 的训练数据。

| 新分支 | 训练页面 | 版面监督 | 目的 |
|---|---|---|---|
| `C4-layout-box` | AncientDoc 真实页面；不混 synthetic replay | OCR + 区域存在性 + bbox L1/GIoU；优先使用人工/公开真值，伪标签单独统计 | 验证最小布局监督是否改善整页定位和 OCR |
| `C5-layout-order` | 同一真实页面和同一 optimizer-step 预算 | C4 + writing direction + 有序 query 槽位/reading order | 验证列方向和顺序监督的增益 |
| `C6-layout-pseudo` | 同一真实页面；加入通过审计的高置信伪标签，并记录标签来源 | C5 + confidence-weighted pseudo layout；可选 column/textline 双层消融 | 验证扩大布局覆盖后的收益及噪声风险 |

匹配的 `C4-ocr-only`（同页面、同起点、同预算）应作为独立基线，而不是偷偷使用 synthetic replay。所有分支使用相同 whole-page 输入、页面分组 split、prompt、有效 batch size 和 validation selection 规则。旧 frozen test 不用于新分支的调参。

### 5.1 损失接口

当前 GOT2 layout 计划已经定义了可接入的布局辅助损失，建议沿用而不是重新引入外部 bbox 输入：

```text
L_total = L_ocr
        + lambda_obj * L_obj
        + lambda_box * L_box
        + lambda_dir * L_dir
        + lambda_ord * L_ord   # 仅在无序 query 消融中启用
```

- `L_ocr`：页面级转写交叉熵，始终是主损失。
- `L_obj`：区域/列是否存在。
- `L_box`：归一化 bbox 的 L1 + GIoU。
- `L_dir`：`vertical_rtl`、`vertical_ltr`、`horizontal_ltr` 等方向类别。
- `L_ord`：只用于无序 query 的顺序消融；有序 query 主配置优先使用真实阅读顺序分配槽位，避免重复计算顺序损失。
- 伪标签项按 teacher 置信度或人工审计状态降权；不能用伪标签覆盖真实标签。

版面损失必须与页面 CER、编辑距离、整页 exact match、reading-order accuracy、bbox IoU/召回率同时报告。版面指标变好而 OCR 变差时，不能把结果描述成模型整体改进。

## 6. 风险与停止条件

1. **粒度错配**：MTHv2 文本行不是天然的列框。若目标是列监督，必须使用 boundary line 推导或人工校准，并在 manifest 中标注 `layout_level`。
2. **来源泄漏**：随机页划分会把同书手、同版本的版式带入 validation/test；正式结果必须按来源组隔离。
3. **伪标签偏差**：Mask R-CNN/YOLO 对细长列、边栏、印章和破损页面可能漏检；低置信页面应保留 OCR-only，而不是强行加入布局损失。
4. **跨脚本迁移**：HJDataset、cBAD、DIVA-HisDB 只提供几何先验，不能据此声称中文古籍泛化。
5. **许可证风险**：MTHv2 与 HJDataset 有研究/版权限制；下载、再分发和发布衍生标注前需逐项确认条款。
6. **模型许可风险**：dhSegment 为 GPL-3.0，DocLayout-YOLO 为 AGPL-3.0；若只在内部离线制备可行，发布整合代码前需做许可证审查。
7. **停止条件**：如果布局监督在来源隔离 validation 上不改善 OCR 或顺序指标，或收益只存在于已见页面模板，应停止扩大 query/损失复杂度，保留更简单的 OCR-only 或 bbox-only 分支。

## 7. 参考来源

1. Ma, W. et al. “Joint Layout Analysis, Character Detection and Recognition for Historical Document Digitization.” ICFHR 2020. DOI: [10.1109/ICFHR2020.2020.00017](https://doi.org/10.1109/ICFHR2020.2020.00017). MTHv2 release: [HCIILAB/MTHv2_Datasets_Release](https://github.com/HCIILAB/MTHv2_Datasets_Release).
2. Cheng, H., Jian, C., Wu, S., & Jin, L. “SCUT-CAB: A New Benchmark Dataset of Ancient Chinese Books with Complex Layouts for Document Layout Analysis.” 2022. DOI: [10.1007/978-3-031-21648-0_30](https://doi.org/10.1007/978-3-031-21648-0_30).
3. Shen, Z., Zhang, K., & Dell, M. “A Large Dataset of Historical Japanese Documents with Complex Layouts.” CVPR Workshops 2020. [HJDataset](https://github.com/dell-research-harvard/HJDataset), [arXiv:2004.08686](https://arxiv.org/abs/2004.08686).
4. Hu, X., Wei, B., Gao, L., & Wang, J. “SegHist: A General Segmentation-Based Framework for Chinese Historical Document Text Line Detection.” ICDAR 2024, pp. 391–410. DOI: [10.1007/978-3-031-70543-4_23](https://doi.org/10.1007/978-3-031-70543-4_23).
5. Clérice, T. “You Actually Look Twice At it (YALTAi): using an object detection approach instead of region segmentation within the Kraken engine.” arXiv:2207.11230 / JDMDH. DOI: [10.46298/jdmdh.9806](https://doi.org/10.46298/jdmdh.9806). Zenodo datasets: [10.5281/zenodo.6814770](https://doi.org/10.5281/zenodo.6814770), [10.5281/zenodo.6814769](https://doi.org/10.5281/zenodo.6814769).
6. Fizaine, F. C. et al. “Historical Text Line Segmentation Using Deep Learning Algorithms: Mask-RCNN against U-Net Networks.” *Journal of Imaging* 10(3), 65 (2024). DOI: [10.3390/jimaging10030065](https://doi.org/10.3390/jimaging10030065).
7. Oliveira, S. A. et al. “dhSegment: A Generic Deep-Learning Approach for Document Segmentation.” ICFHR 2018. DOI: [10.1109/ICFHR-2018.2018.00011](https://doi.org/10.1109/ICFHR-2018.2018.00011). Code: [dhlab-epfl/dhSegment](https://github.com/dhlab-epfl/dhSegment).
8. Boillet, M., Kermorvant, C., & Paquet, T. “Multiple Document Datasets Pre-training Improves Text Line Detection With Deep Neural Networks.” arXiv:2012.14163. Doc-UFCN is treated here as a paper-level baseline because an official repository was not verified in this search.
9. cBAD 2019 dataset, Zenodo DOI [10.5281/zenodo.3568023](https://doi.org/10.5281/zenodo.3568023), CC BY 4.0.
10. cBAD 2017 dataset, Zenodo DOI [10.5281/zenodo.1491441](https://doi.org/10.5281/zenodo.1491441), CC BY-SA 4.0.
11. DIVA-HisDB, Zenodo DOI [10.5281/zenodo.19127869](https://doi.org/10.5281/zenodo.19127869), CC BY 4.0.
12. DocLayout-YOLO: [opendatalab/DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO), arXiv: [2410.12628](https://arxiv.org/abs/2410.12628).
13. Kraken documentation and code: [kraken.re](https://kraken.re/main/index.html), [mittagessen/kraken](https://github.com/mittagessen/kraken).

## 8. 当前决策

本调研不下载大数据、不修改 C4–C6 训练代码、不启动服务器训练。下一步若确认协议，可先实现：

1. MTHv2 annotation → 项目 manifest 的转换和审计脚本；
2. Mask R-CNN teacher 的最小训练/推理入口；
3. `C4-layout-box/C5-layout-order/C6-layout-pseudo` 的配置模板和 validation selection；
4. 一小批人工抽检页面的标签质量报告。
