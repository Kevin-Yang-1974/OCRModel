# GLM-OCR decoder-mask hardsync window beta=1 快速验证日志

## 1. 运行边界

- run：`glmocr_decoder_mask_window_beta1_5gpu_quick_20260922_045950`
- 分支：`glm-ocr-layout-mask-routing`
- commit：`e1667b2`（叠加未提交改动）
- 远端代码：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/code/ocrmodel`
- 远端产物：`/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_decoder_mask_window_beta1_5gpu_quick_20260922_045950`
- 设备：A100 GPU `0,1,2,3,4`；五个单卡 shard worker，不是 DDP

本日志记录在 line beta4 被停止后直接启动的 window beta=1 快速验证。它使用同一 149 页 validation manifest；不读取 test、不参与 checkpoint selection，也不把五个 shard 当作五次重复实验。

## 2. 固定协议

| 项目 | 配置 |
| --- | --- |
| base / checkpoint | `ca5d8b3e287e52589e37c28385d9655ee4372f9d`；`checkpoint-3000` trained layout+decoder checkpoint |
| manifest | `/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1_q32_sparse24/validation/manifest.char.jsonl`；149 页；SHA256 `36ec845875e1ea18a48d4a523b6c5a3f007b46e2a07de0cafd2c26139929a348` |
| mask / pointer | window `3–5` 字；hard mask；synced pointer；`split_layer=0`，全部层注入 |
| inference | fast processor；math-SDPA 确定性执行；`max_pixels=4000000`；`max_new_tokens=1536`；seed `42`；`bias_max=1.0` |
| sharding | round-robin 五 shard：`30/30/30/30/29` 页；按 reference characters 与 I/D/S 聚合 |

## 3. 结果

| shard | pages | reference characters | CER | I/D/S | generation tokens | limit hits |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0 | 30 | 8686 | `0.1741883491` | `385/354/774` | 10514 | `0/30` |
| 1 | 30 | 8030 | `0.1437110834` | `103/132/919` | 9575 | `0/30` |
| 2 | 30 | 8863 | `0.1197111587` | `145/102/814` | 10853 | `0/30` |
| 3 | 30 | 8264 | `0.1101161665` | `127/100/683` | 9801 | `0/30` |
| 4 | 29 | 7811 | `0.1364742030` | `188/81/797` | 9578 | `0/29` |
| **aggregate** | **149** | **41654** | **`0.1369376290`** | **`948/769/3987`** | **50321** | **`0/149`** |

全部 shard 均 `status=complete`。五个 summary 均记录：

```text
reads_ground_truth=true
usable_for_selection=false
test_used_for_selection=false
```

未发现 CUDA、OOM、NaN、Inf 或 Traceback；完成后五张卡已释放。

## 4. 解释边界

同一 trained checkpoint、同一 149 页协议下，B0 的 CER 为 `0.1699236568`，本次 window beta=1 为 `0.1369376290`，绝对下降 `0.0329860278`，相对约 `19.4%`。这支持“训练模型＋3–5 字 window GT mask”存在明显 routing 潜力。

但本结果仍是 GT-mask oracle：它隔离了模型与 mask 形态的部分混淆，却不能证明 decoder 已能预测出同等有效的 mask。后续合并时应继续保持 attention-tracking 最佳版本的 checkpoint、processor、SDPA、hard mask、synced pointer 和全部层注入语义，只替换 mask provider。
