# AncientDoc 旧分析计划（已完成）

本文原计划中的 C4/C6 配对分析和书籍泄漏审计已经完成。旧 split 检出 131 个跨 split 书籍；后续首个 group allocator 又产生失衡的 `534/1021/1025` 划分。两者均不再作为正式实验入口。

当前流程已替换为：生成近 60/20/20 的书籍隔离 split，定期保存训练 checkpoint，只在 validation 选优，然后冻结并评估 test。请以 `docs/SYNC_AND_RUN.md` 和 `tools/training/README.md` 为准。
