# 布局分支写回诊断：布局信息进得去，decoder 用不上

**日期**：2026-09-19　**相关 run**：`glmocr_dunhuang_layout_branch_scale_iou_20260919_v1`
**状态**：机制已定位；「decoder 为何不用」待下一步实验区分

## 一、要回答的问题

布局分支的 IoU 从 `0.0725` 训到 `0.6721`（近 10 倍），但对文字识别 CER 无可测贡献。
受控消融确认了三个表面事实，但未能解释**为什么**：

- `gate=0` 与 `gate=0.0129` 之间 CER 有显著差异（残差通道有作用）；
- 但随机查询机制的分支与训练分支的 CER 几乎相同（`0.11183` vs `0.11198`）；
- 布局 IoU `0.42 / 0.65 / 0.67` 的阶梯对最终 CER 无单调影响。

本文用一个前向插桩回答「布局信息到底有没有被写进解码器可见的特征」，并记录本诊断
过程中暴露的实验方法学问题。

## 二、结构前提

布局分支通往 OCR 的唯一路径（`layout_ocr/glm_bridge.py`）：

```
merged_tokens    = visual_tokens + gate * layout_context
layout_context[p] = content_norm( Σ_q token_weights[p,q] · sem[q] )
```

- `token_weights[p,:]` 是第 p 个 patch 对 32 个 query 的归一化注意力分布；
- `sem[q]` 是第 q 个 query 的语义特征（可经过 `sem_adapter`）；
- box / order / direction 头只经 `last_output` 侧信道进训练损失，从不进基座前向。

因此「布局信息能否到达 decoder」完全取决于 `layout_context[p]` 在不同 patch 之间
变化多大。近似常向量 → 布局结构无从表达；随 patch 显著变化 → 空间信息确实被写入，
剩下的问题是 decoder 是否利用它。

## 三、插桩

在 `adapter.py` 的 `merged_tokens` 写回前插入 env 门控探针
（`GLMOCR_ADAPTER_PROBE=<path>`，默认关闭，不影响正常路径）。每个前向记录四个量：

- `lc_flat`：`layout_context` 跨 patch 偏离度 / 自身范数。≈0 = 退化成全局偏置，
  1 = 每个 patch 的修正都不同。
- `vt_flat`：`visual_tokens` 的同一统计量，作参照系（视觉 token 已知 patch 特异）。
- `tw_flat`：`token_weights` 跨 patch 标准差均值。
- `lc_norm_over_vt`：`layout_context` 与 `visual_tokens` 的范数比。

对 Q-rand（随机查询机制）与 Q-065（giou10x@10000）两个 checkpoint 各跑 80 页
eval-only。

## 四、结果

| 臂 | 布局分支 | `lc_flat` | `vt_flat` | `tw_flat` | 残差/视觉范数 | validation CER |
|---|---|---|---|---|---|---|
| Q-rand | 随机查询机制 | **0.0080** | 0.7878 | 0.0197 | 1.595 | 0.111832 |
| Q-065 | giou10x@10000 | **0.3895** | 0.7889 | 0.0911 | 1.601 | 0.112168 |

（残差/视觉范数约 1.6——注入的残差幅度比视觉 token 本身还大一半，幅度不是瓶颈。）

## 五、结论

1. **随机分支的写回确实退化**：`lc_flat = 0.008`，`layout_context` 对每个 patch 近似
   同一向量——它本就没有布局信息，退化成全局偏置。
2. **训练分支的写回携带真实空间结构**：`lc_flat = 0.3895`，是随机分支的约 49 倍，
   跨 patch 有近 40% 的结构变异。**布局信息确实被写进了解码器可见的特征。**
3. **但两者对 CER 的影响几乎相同**（差异 `0.0003`，低于 80 页可检测阈值约 `0.0017`）。

因此真正的问题**不是「布局信息进不去」，而是「进去了但 decoder 不利用」**。此前基于
单条随机分支记录（`lc_flat ≈ 0.013`）断言「写回退化」，是过度下结论，本文予以更正。

## 六、候选机制（均需进一步实验区分）

- **A. 注入点错误**：写回发生在视觉编码器出口，但 decoder 主要依赖视觉 token 间全局
  注意力；逐 patch 局部修正可能被后续注意力/归一化洗掉。
- **B. 残差方向与语义通路正交**：`content_norm` 后的向量在语义空间，而 decoder 需要
  的是能区分字形/布局的表示；若子空间正交，范数再大也无用。
- **C. 注入时机**：`merged_tokens` 进入 language model 前还有视觉 projection/embedding
  层，可能已把 patch 级差异抹平。

## 七、下一步（判定优先级）

1. **验证「进去但不用」**：把训练分支的 `layout_context` 替换成保持逐 patch 结构
   （`lc_flat` 相同）的零均值高斯噪声，若 CER 仍不变，则证明 decoder 对「任何逐 patch
   修正」无感，问题在注入点而非内容。
2. **换注入点**：改写到 embedding 之后，或对 `visual_tokens` 逐 patch 重排（重排阅读
   顺序），看 CER 是否起反应。
3. **若 1 证实 decoder 无感**：布局分支的价值应转向生成后处理（用预测阅读顺序重排
   输出文本 / 框选裁剪逐区域识别），而非特征注入。

## 八、本诊断过程暴露的方法学问题

1. **不要把单条记录当统计结论**。最初只看 Q-rand 一条记录（`lc_flat=0.013`）就断言
   「写回退化」，而 Q-065 聚合值 `0.39` 完全相反。必须等两臂各 80 页完整记录再下结论。
2. **IoU 阶梯实验 Q-042 臂有缺陷**：首次 checkpoint 手术因瞬时 I/O 错误失败，但
   `cp -r` 已留下未替换查询机制的副本；随后加的「config 存在即跳过」幂等守卫正好跳过
   它，导致 Q-042 与 W 臂完全同臂（`eff_scale`/`res_norm`/`entropy`/`fusion_mass` 逐位
   相同）。IoU=0.4225 这一点实际未测得。
3. **可检测性 ≠ 不存在**：四臂 CER 差异 ≤0.0003，而 80 页可检测阈值约 0.0017；
   「测不出差异」不等于「无差异」。此前把 Q-rand 略优说成「反向相关」是过度解读，已撤回。

## 复现

```bash
# 远端探针（每页前向 append 一条 JSON 记录）：
#   GLMOCR_ADAPTER_PROBE=<out.jsonl> python -m layout_ocr.train_screen ... --eval-only
# 记录含 patches / lc_flat / vt_flat / tw_flat / lc_norm_over_vt
```

## 附：与「resolution 是主瓶颈」结论的关系

本诊断不改变 `STAGE2_IMPROVEMENT_PLAN.md` 的核心结论——输入分辨率（`max_pixels`）仍是
最大的、且无需训练的杠杆。本文针对的是更细的问题：**在分辨率已修正的前提下**，布局
分支为何仍无贡献。答案是它把空间结构写进了特征，但 decoder 不利用，这是注入点/耦合
方式的设计问题，而非布局分支的训练质量。
