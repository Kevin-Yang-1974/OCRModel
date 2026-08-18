# 整页布局合成工具

本目录的活动入口用于生成 GOT2＋VLQA 的整页合成数据。模型输入始终是完整页面截图；DOM bbox、阅读顺序和书写方向只写入训练 manifest，不进入推理接口。

## 1. 依赖

页面规划和单元测试只需要 `numpy` 与 `Pillow`。真实 DOM 渲染使用锁定的 Playwright：

```powershell
python -m pip install -r tools/environment/requirements-layout-synthesis.lock.txt
# 本项目正式合成数据使用系统 Edge；不要安装或依赖 Playwright 捆绑 Chromium。
```

正式数据应使用系统 Edge。可通过 `--browser-channel msedge` 或显式 `--chromium-executable` 指向本机 Edge；生成结果会记录 Playwright 和浏览器版本。正式 manifest 当前使用 schema v2。

## 2. 内容清单

内容清单支持 JSON 数组、带 `records` 的 JSON object 或 JSONL。所有内容必须先划分 split。纯 HTML 文本记录为：

```json
{"content_id":"text_001","source_group_id":"book_001","split":"train","kind":"text","orientation":"any","language":"zh-Hant","text":"待排版文字或符号"}
```

真实 crop 记录为：

```json
{"content_id":"crop_001","source_group_id":"book_001","split":"train","kind":"image","orientation":"vertical","image":"crops/crop_001.png","text":"该列的真值转写"}
```

`image` 必须相对 `--content-root`，并且 `orientation` 必须显式为 `horizontal`、`vertical` 或更严格的 `horizontal_ltr`、`horizontal_rtl`、`vertical_ltr`、`vertical_rtl`。对文本方向已知的真实 crop 应使用严格值，防止把 LTR 图像作为 RTL 监督。同一 `source_group_id` 不得跨 split；相同 `content_id` 不得重复。

`language` 为可选 BCP-47 风格标签，缺省为 `und`。多语言正式数据必须由内容清单提供真实转写，可混合 `zh-Hans`、`zh-Hant`、`ja`、`ko`、`en` 或 `mixed-symbols`；生成器只负责采样和记录，不下载、不翻译也不虚构语料。

配置中的 `font_family` 控制 CSS 字体栈，`allowed_rendered_fonts` 列出允许实际绘制字形的字体族。正式渲染会通过浏览器 CDP 逐文本区域读取平台字体；没有实际 glyph、出现未声明 fallback，或审计时字体证据缺失都会报错。若确需符号字体 fallback，必须把其实际 family name 显式加入允许列表并保存同一字体包版本。

仓库中的 `config/synthetic_content.example.jsonl` 与 `config/synthetic_layout.example.json` 仅用于 schema 和本地 smoke，不是正式训练数据。

若本地 PDF 自带可靠文本层，可以从文本块同时生成浏览器文字记录和真实 PDF crop 记录。该入口按 PDF 文件哈希去重，并把整份 PDF 固定到单一 split；crop 标签只取自 PDF 文本层，不执行 OCR 猜测：

```powershell
python tools/preprocessing/prepare_pdf_layout_content.py `
  --pdf-root D:\corpus\pdf `
  --output-dir D:\layout_source\pdf_content_seed20260812 `
  --seed 20260812 `
  --validation-sources 4 `
  --test-sources 4
```

输出的 `content.jsonl` 同时包含 `kind=text` 与 `kind=image` 记录，`crops/` 保存真实 PDF 区域图像，`source_report.json` 保存重复 PDF、来源级 split、提取数量和 PyMuPDF 版本。扫描 PDF、损坏 PDF 和没有合格文本块的 PDF 会在报告中标记并跳过，不能用猜测转写补齐。

## 3. 规划与渲染

不启动浏览器的确定性规划：

```powershell
python tools/preprocessing/generate_synthetic_layout.py `
  --content-manifest config/synthetic_content.example.jsonl `
  --output-dir D:\layout_runs\plan_001 `
  --split train `
  --tier s0-html-text `
  --num-pages 8 `
  --seed 20260810 `
  --config config/synthetic_layout.example.json `
  --plan-only
```

正式 DOM 渲染去掉 `--plan-only`：

```powershell
python tools/preprocessing/generate_synthetic_layout.py `
  --content-manifest D:\layout_source\content.jsonl `
  --content-root D:\layout_source `
  --output-dir D:\layout_runs\train_s0_seed20260810 `
  --split train `
  --tier s0-html-text `
  --num-pages 1000 `
  --seed 20260810 `
  --config config/synthetic_layout.example.json
```

`--tier` 可以重复。`--num-pages` 表示每个 tier 的页面数；例如 `P2` 的联合合成集可在同一输出目录和同一 manifest 中生成：

```powershell
python tools/preprocessing/generate_synthetic_layout.py `
  --content-manifest D:\layout_source\content.jsonl `
  --content-root D:\layout_source `
  --output-dir D:\layout_runs\train_mixed_seed20260810 `
  --split train `
  --tier s0-html-text `
  --tier s1-html-crop `
  --tier s2-hard `
  --num-pages 1000 `
  --seed 20260810 `
  --config config/synthetic_layout.example.json
```

上述命令总计生成 3000 页。内容清单必须为每个所选 tier 提供足够且方向兼容的记录；联合 manifest 中的 `page_id` 包含 `s0`、`s1` 或 `s2` 标记。

面向古籍照片域的多样化 preset 为 `config/synthetic_layout.ancient_photo_diverse_v1.json`。它在不改变 bbox 几何真值的前提下覆盖：非均匀行/列宽、普通与 0–6 px 密集间距、18–52 px 字号、紧凑行高、负字距、逐区域 CJK/衬线字体栈、墨色/透明度、十种纸张底色，以及更宽的对比度、透印、污渍、模糊、高斯噪声和离散噪点范围。该 preset 只是更接近古籍照片退化分布的合成近似，不等同于真实古籍。

统一生成 train/validation/test 的入口如下；默认每个 tier 生成 `8000/1000/1000` 页，即三 tier 合计 30000 页。先用 `--plan-only` 小规模检查内容、字体与版式，再执行正式浏览器渲染：

```powershell
& .\.venv\Scripts\python.exe tools\preprocessing\prepare_diverse_synthetic_layout.py `
  --content-manifest D:\layout_source\content.jsonl `
  --content-root D:\layout_source `
  --output-root D:\layout_data\ancient_photo_diverse_v1_seed20260817 `
  --browser-channel msedge
```

正式字体文件必须成对传入 `--font-file` 和 `--expected-font-sha256`；所有实际渲染字体 family 仍必须出现在 `allowed_rendered_fonts`，否则立即失败。联合 `dataset_protocol.json` 记录 content/config 哈希、seed、split/tier 页数、输入粒度和 metadata 不进模型的协议。

输出目录必须不存在或为空，工具不会覆盖、删除或混入既有数据。输出包括：

```text
dataset_meta.json
manifest.jsonl
html/<page_id>.html
images/<page_id>.png
```

`s0-html-text` 只接受文本内容，`s1-html-crop` 只接受真实 crop，`s2-hard` 接受两者并增加保持几何不变的对比度、透印、污渍、模糊和噪声。当前 `s2-hard` 不做会改变 bbox 的几何形变。

## 4. 审计

单个 split：

```powershell
python tools/preprocessing/audit_synthetic_layout.py `
  --manifest D:\layout_runs\train_s0_seed20260810\manifest.jsonl `
  --summary-json D:\layout_runs\train_s0_seed20260810\audit_summary.json
```

正式划分应一次传入全部 manifest：

```powershell
python tools/preprocessing/audit_synthetic_layout.py `
  --manifest D:\layout_runs\train\manifest.jsonl `
  --manifest D:\layout_runs\validation\manifest.jsonl `
  --manifest D:\layout_runs\test\manifest.jsonl `
  --summary-json D:\layout_runs\split_audit.json
```

审计器检查页面和区域 schema、相对路径、截图尺寸及 SHA-256、HTML page ID、归一化 bbox、阅读顺序、页面转写、逐区域实际字体、`content_id`、`source_group_id`、源 crop 哈希和页面哈希的跨 split 泄漏。只有输出 `SYNTHETIC_LAYOUT_AUDIT_OK` 的 manifest 才能进入 dataset。

## 5. 当前训练接口

活动源码已经包含：

- `src/GOT-OCR-2.0/scripts/layout_page_dataset.py`：整页 dataset 与布局监督 collator；
- `src/GOT-OCR-2.0/GOT/model/layout_query.py`：VLQA、辅助头、零门控写回和布局损失；
- `src/GOT-OCR-2.0/scripts/train_GOT_layout.py`：`P1/P2` 开发态训练入口。

`P1` 只训练 layout queries、query cross-attention 和辅助头，OCR 标签全部 mask，残差 gate 固定为 0。`P2` 打开完整 VLQA 与 `mm_projector_vary`，联合优化页面 OCR 和布局损失。A100 上不需要安装 Playwright，也不重新渲染页面；应先在本机生成完整的 `manifest.jsonl`、`images/` 和 `html/`，再整体上传数据目录。

`tools/training/run_layout_a100.py` 已在 A100 打通 manifest 复审、CUDA 组件检查、P1/P2 训练衔接和 checkpoint 重载；`layout_overfit_20260812_002747` 已通过固定 1000 steps 的 P1 两样本实现诊断，`layout_validate_20260812_014816` 已完成同两页 P1 checkpoint 的 prompt-only 评测链路。正式入口 `--mode pretrain` 和 `--mode joint-train` 已在 `formal_pdf_short_seed20260812` 上完成首轮合成训练：P1 `1000` steps，P2 `2000` steps，均完成 300 页 validation。`tools/evaluation/compare_got2_vlqa.py` 和 `tools/evaluation/run_formal_layout_comparison.sh` 用于同一 test manifest 下比较原始 GOT2 与 P2 VLQA。现有结果只代表合成数据首轮训练链路已跑通，尚不能作为真实跨域泛化或 VLQA 结构收益结论。具体命令见 `../../docs/SYNC_AND_RUN.md`。
