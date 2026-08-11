# 环境与依赖

## 依赖分层

本目录使用三套彼此独立的依赖口径：

| 用途 | Python | 依赖文件 | 说明 |
|---|---:|---|---|
| Windows 本地编辑、语法检查、CPU 预处理 | 当前可用 Python 3 | `tools/environment/requirements-local-dev.txt` | 不安装 PyTorch、CUDA、DeepSpeed 或模型 |
| 当前 A100 GOT 训练与推理 profile | 3.10.20 | `tools/environment/requirements-got-server.lock.txt` | PyTorch 2.0.1 + CUDA 11.8；GOT 从仓库源码可编辑安装 |
| 当前 A100 AnandaSky 推理 profile | 3.11.15 | `tools/environment/requirements-anandasky-server.lock.txt` | PyTorch 2.5.1 + CUDA 12.1；FlashAttention 单独安装 |

服务器锁文件是已通过环境的复现快照，不应直接在 Windows 上执行 `pip install -r`。其中的 PyTorch CUDA wheel 需要官方 PyTorch wheel index；GOT 锁文件中的历史 `GOT==0.1.0` 由安装脚本过滤，实际执行：

```bash
python -m pip install --no-deps -e "$GOT_PROJECT_ROOT"
```

这样服务器运行的就是本地同步后的活动源码，而不是 PyPI 包或旧服务器仓库。

## 关键服务器版本

GOT 环境：

```text
Python 3.10.20
torch 2.0.1+cu118
torchvision 0.15.2+cu118
transformers 4.37.2
deepspeed 0.12.3
accelerate 0.28.0
numpy 1.26.4
```

AnandaSky 环境：

```text
Python 3.11.15
torch 2.5.1+cu121
transformers 4.55.4
accelerate 1.10.1
flash_attn 2.7.4.post1
numpy 1.26.4
```

## 使用原则

1. 先运行 `check_server_envs.py`。现有环境通过检查时，不重建环境。
2. 只有缺包、版本错误或 GOT editable source 指向旧目录时，才执行 `setup_server_envs.sh got`。
3. AnandaSky 环境和 FlashAttention 只有在需要 AnandaSky 推理时才处理，不作为 GOT 训练前置条件。
4. `setup_server_envs.sh` 不下载模型权重、不复制数据集、不修改系统 Python、系统 CUDA或 shell 启动文件。
5. 权重、数据和运行目录由机器本地的 `config/paths.env` 指定，不在代码中保存个人绝对路径。只读共享资产不得由安装或运行脚本修改。

## 整页合成与 VLQA 依赖状态

整页合成器已实现于 `tools/preprocessing/generate_synthetic_layout.py`，当前使用 Playwright/Chromium 读取 DOM bbox，并使用 NumPy/Pillow 完成保持几何不变的退化。Python 依赖已固定在 `tools/environment/requirements-layout-synthesis.lock.txt`：

```text
numpy==2.1.3
Pillow==11.1.0
playwright==1.62.0
```

Playwright 安装的 Chromium revision 与包版本配套。生成器固定 viewport、`deviceScaleFactor=1`，等待 `document.fonts.ready` 和全部图片解码完成，并把 Playwright、Chromium、字体栈和 seed 写入 provenance。本机已使用 Playwright 1.62.0＋系统 Edge 151.0.4129.59 完成 `S0` 与 `S2-hard` 真实截图 smoke；该系统 Edge 组合仅证明代码链路可运行，不作为正式数据的浏览器锁。

正式页面应优先在本地 CPU 侧用 Playwright 配套 Chromium 生成，再把截图和 manifest 作为外部数据同步到训练侧。A100 GOT 环境不应为了训练被动安装浏览器。正式字体包和缺字覆盖规则仍需随数据集固定；未完成字体覆盖审计前，不得扩大纯 HTML 字体数据规模。
