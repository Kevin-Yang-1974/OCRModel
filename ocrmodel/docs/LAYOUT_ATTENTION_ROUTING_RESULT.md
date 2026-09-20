# 布局注意力路由：把偏置打到字框上

**日期**：2026-09-20
**前序**：`plans/LAYOUT_ATTENTION_ROUTING.md`（设计）、`docs/LAYOUT_WRITEBACK_INTERVENTION_RESULT.md`（残差缝的判定）
**状态**：4M 四臂扫描进行中。第一节到第五节是已确定的方法学结论，第六节待填。

---

## 一、与之前六次的区别

前面六次（含真值上界）全部是**往序列里加信息**：残差缝的逐 patch 上下文、K 个保留前缀槽。判定是不加最好，逐 patch 分量甚至是敌对的。

这一轮不动序列，只改**注意力落在哪里**：在每个解码步，给落在目标字框内的视觉 key 的注意力 logits 加一个常数。判据是逐页配对 bootstrap 的 CI，不是点估计。

## 二、方案 §7 的顾虑不成立

设计里写「给 4D 加性掩码会关掉 `is_causal`，因果性必须自己编进掩码」。查 `sdpa_attention_forward` 后不需要：

```python
is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
```

**解码步（q_len == 1）本来就是 `is_causal=False`**——单 query 在序列末尾，所有 key 都在过去，没有因果结构可编。所以偏置只作用在解码步：不必手写因果掩码，prefill（贵的那一趟，且会掉出 flash kernel）完全不碰。代价是一个字：字符 0 由 prefill 采样，拿不到偏置。

偏置是**加到** `create_causal_mask` 产出的掩码上，而不是替换它——解码步那个掩码没有因果项，但可能带 padding 结构，相加可以白拿。

## 三、指针不能按生成步数索引（实测）

「第 t 步 → 第 t 个字框」是检测器唯一能提供的形式（不需要文本），但它在本 checkpoint 上不可用。用真值 `page_text` 对齐回真实阅读位置后测偏移：

```
|drift| in chars   mean 121.3   median 6   p90 547   p99 992   max 1271
steps with |drift|<=5   49.7%
```

最差页生成 1568 字对 343 字真值，指针偏出 1271 字——好几列之外。原因是过生成（8% 的页写到 1536 上限，插入占编辑距离 60%）。

**所以真值上界臂用 `synced` 指针**：沿真值 `page_text` 走，指出模型真正读到的字。`step` 保留为「可部署策略值多少钱」的对照臂。

这同时是对可部署路径的坏消息：检测器只能给「阅读顺序第 n 个框」，模型生成一旦和阅读顺序脱同步，这个策略本身失效。原设计里没有这一条。

## 四、分辨率是这轮的隐藏变量

1M 像素下每个字只有约一个视觉 token：

```
转换后页面        1500 × 3000 = 4.5M px
max_pixels=1M     下采样 2.12×
一个字的原始尺寸   ≈ 68 × 76 px    （列宽 68px，行内 17 字铺满 1294px）
1M 下             ≈ 32 × 36 px
视觉 token        patch14 × merge2 = 28 px
→ 约 1.4 个视觉 token / 字
```

实测印证：1M 探针 `visual_tokens=630`、`mean_boxes_hit=1.0`；4M 探针 `visual_tokens=2496`、`mean_boxes_hit=3.24`。

4M 还顺带改善同步：1M 触顶页生成 1536 步，4M 同类页只有 414 步（多 20%）。分辨率既提升模型能力，又缓解第三节的偏移问题——两者是同一个病。

## 五、1M 那一轮测错了（接线 bug）

第一轮 1M 四臂的探针显示 `pointer_position: 0` 对 336 字的页——**偏置一直打在第一个字上**，不是模型当前读到的字。

根因：`prepare_inputs_for_generation` 用 `cache_position` 切片 `input_ids`，解码步只给模型 **1 个 token**，不是整条序列。按长度算「已生成多少」得到 `1 - prompt_length`（负数），每次提前返回，指针停在原地。

修复：按 `cache_position` 对齐——「推进位置 ≥ prompt_length 的那些 id」，一条规则同时覆盖 prefill 与解码步。产物保留在 `routing_eval/glmocr_layout_routing_eval_20260920_v1` 作为证据。

**同一处还有两个计数缺陷**：

- `steps` 在每个 decoder layer 的 hook 里自增，而模型有 16 层，于是报出解码步数的 16 倍。它被当作覆盖率用于接线判据（`biased_fraction`），会把**每一条臂**判成「接线未接通」（实测 6.2%）。改为按 forward 计数。
- 测试助手把整条序列喂给 hook，正是掩盖了这个 bug 的形状。测试改为按 `generate` 的方式喂。

## 六、结果

**实验设置**

| 项 | 值 |
|---|---|
| 数据 | MTHv2 sparse24 validation，149 页 |
| 字框 | `manifest.char.jsonl`（`tools/prepare_mthv2_char_manifest.py`） |
| 模型 | `glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000` |
| 分辨率 | `max_pixels=4000000` |
| 臂 | `bias0`(全零掩码，须复现记录基线) / `bias1` / `bias2` / `bias4` |
| 指针 | `synced` |
| 判据 | 每臂对 `bias0` 的逐页配对 bootstrap（10000 次），CI 不含零才算 |

**判定**

待 4M 扫描完成。

**已知的观察**

- `bias4` 慢 3 倍（30 分钟 29 页 vs 其它臂 103 页），提示偏置过强时生成退化。若成立，机制的有效区间很窄。
- 字框通道的顺序锚定在 `page_text` 上，属于**评测真值，不是推理输入**（AGENTS.md 第 3 条）。
