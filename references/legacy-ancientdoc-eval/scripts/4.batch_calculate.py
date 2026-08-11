'''
=== OCR 批量评测脚本 (High Performance) ===

功能说明:
1. 自动扫描 results_root 下的所有子文件夹。
2. 寻找包含 results_final.json 的目录作为有效任务。
3. 并行计算 CER, Precision, Recall, F1 四大指标。
4. 支持断点续传：若目标目录下已有 metrics.json，则自动跳过计算，仅读取结果。
5. 输出: 在每个模型目录下生成 metrics.json (详细) 和 metrics.txt (简报)。

依赖库:
pip install tqdm python-Levenshtein
'''

import json
import os
import sys
import concurrent.futures
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import Levenshtein  # 核心加速库

# ==========================================
# 1. 全局配置区域 (Configuration)
# ==========================================
@dataclass
class Config:
    # 结果根目录 (脚本会自动扫描该目录下的子文件夹)
    results_root: str = "results/Ancient/dimension-gate_lambda-0.5"
    
    # 待处理的原始结果文件名
    input_filename: str = "results_final.json"
    
    # 输出文件名 (生成在各模型子目录下)
    output_json_name: str = "metrics.json"  # 详细每条数据的指标
    output_txt_name: str = "metrics.txt"    # 平均值简报
    
    # 并行进程数 (建议设置为 CPU 核心数 - 2)
    num_workers: int = 16
    
    # 是否强制重新计算 (False: 发现已有结果则跳过; True: 无论如何都重算)
    force_recompute: bool = False

config = Config()

# ==========================================
# 2. 核心算法 (Core Metric Algorithms)
# ==========================================
def compute_metrics_row(label: str, answer: str) -> Dict[str, float]:
    """
    计算单条数据的 CER, Precision, Recall, F1
    使用 Levenshtein C-Extension 极速计算
    """
    if label is None: label = ""
    if answer is None: answer = ""
    label = label.strip()
    answer = answer.strip()
    
    # 特殊情况：全空
    if not label and not answer:
        return {"cer": 0.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    len_ref = len(label)
    len_hyp = len(answer)
    
    # 获取编辑操作 (replace, delete, insert)
    ops = Levenshtein.editops(label, answer)
    subs = sum(1 for op in ops if op[0] == 'replace')
    dels = sum(1 for op in ops if op[0] == 'delete')
    ins  = sum(1 for op in ops if op[0] == 'insert')
    
    # 1. CER = (S + D + I) / Reference Length
    dist = subs + dels + ins
    cer = dist / len_ref if len_ref > 0 else (0.0 if len_hyp == 0 else 1.0)
    
    # 2. Matches = Ref - (S + D)  (或者 Hyp - (S + I))
    matches = len_ref - (subs + dels)
    
    # 3. Precision = Matches / Hypothesis Length
    precision = matches / len_hyp if len_hyp > 0 else 0.0
    
    # 4. Recall = Matches / Reference Length
    recall = matches / len_ref if len_ref > 0 else 0.0
    
    # 5. F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "cer": cer,
        "char_precision": precision,
        "char_recall": recall,
        "char_f1": f1
    }

def calculate_average(metrics_list: List[Dict]) -> Dict[str, float]:
    """计算指标列表的平均值"""
    count = len(metrics_list)
    if count == 0:
        return {"cer": 0, "precision": 0, "recall": 0, "f1": 0, "count": 0}
        
    return {
        "cer": sum(x['cer'] for x in metrics_list) / count,
        "precision": sum(x['char_precision'] for x in metrics_list) / count,
        "recall": sum(x['char_recall'] for x in metrics_list) / count,
        "f1": sum(x['char_f1'] for x in metrics_list) / count,
        "count": count
    }

# ==========================================
# 3. 任务处理逻辑 (Worker Process)
# ==========================================
def process_single_model(model_dir: Path) -> Dict[str, Any]:
    """
    处理单个模型目录：
    1. 检查是否已完成 (Cache Check)
    2. 加载 JSON -> 计算指标 -> 保存 metrics.json -> 保存 metrics.txt
    3. 返回统计摘要用于控制台打印
    """
    model_name = model_dir.name
    input_path = model_dir / config.input_filename
    output_json_path = model_dir / config.output_json_name
    output_txt_path = model_dir / config.output_txt_name
    
    # --- Check 1: 断点续传逻辑 ---
    if not config.force_recompute and output_json_path.exists():
        try:
            # 尝试快速读取已有结果
            with open(output_json_path, 'r', encoding='utf-8') as f:
                saved_results = json.load(f)
            
            # 简单的完整性校验
            if saved_results and isinstance(saved_results, list) and "metrics" in saved_results[0]:
                metrics_list = [r["metrics"] for r in saved_results]
                avgs = calculate_average(metrics_list)
                return {
                    "model": model_name,
                    "status": "Cached", # 标记为缓存命中
                    **avgs
                }
        except Exception:
            # 读取失败则继续执行计算逻辑
            pass

    # --- Step 2: 加载原始数据 ---
    if not input_path.exists():
        return {"error": f"Missing {config.input_filename}", "model": model_name}

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {"error": f"JSON Load Error: {e}", "model": model_name}

    if not data:
        return {"error": "Empty Data File", "model": model_name}

    # --- Step 3: 批量计算指标 ---
    results_detail = []
    metrics_only = []

    for idx, item in enumerate(data):
        label = item.get("label", "")
        answer = item.get("answer", "")
        
        m = compute_metrics_row(label, answer)
        
        # 构造详细结果
        results_detail.append({
            "index": idx,
            "image": item.get("image", ""),
            "label": label,
            "answer": answer,
            "metrics": m
        })
        metrics_only.append(m)

    # --- Step 4: 计算平均值并保存 ---
    avgs = calculate_average(metrics_only)
    
    # 4.1 保存详细 JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_detail, f, ensure_ascii=False, indent=2)

    # 4.2 保存简报 TXT
    report_text = (
        f"Evaluation Report for: {model_name}\n"
        f"Generated at: {os.path.basename(sys.argv[0])}\n"
        f"========================================\n"
        f"Total Samples : {avgs['count']}\n"
        f"Average CER   : {avgs['cer']:.4f} ({avgs['cer']:.2%})\n"
        f"Average Prec  : {avgs['precision']:.4f} ({avgs['precision']:.2%})\n"
        f"Average Recall: {avgs['recall']:.4f} ({avgs['recall']:.2%})\n"
        f"Average F1    : {avgs['f1']:.4f} ({avgs['f1']:.2%})\n"
        f"========================================\n"
    )
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # --- Step 5: 返回结果给主进程 ---
    return {
        "model": model_name,
        "status": "Computed",
        **avgs
    }

# ==========================================
# 4. 主控逻辑 (Main Controller)
# ==========================================
def main():
    root_dir = Path(config.results_root)
    
    # 1. 扫描任务
    if not root_dir.exists():
        print(f"❌ 根目录不存在: {root_dir}")
        return

    tasks = []
    print(f"🔍 正在扫描: {root_dir}")
    # 仅选择包含 results_final.json 的目录
    for sub_dir in sorted(root_dir.iterdir()):
        if sub_dir.is_dir() and (sub_dir / config.input_filename).exists():
            tasks.append(sub_dir)

    if not tasks:
        print("⚠️ 未发现任何有效任务目录。请检查 config.results_root 配置。")
        return

    print(f"📦 发现任务数: {len(tasks)}")
    print(f"🚀 并行进程池: {config.num_workers} workers")
    print("-" * 80)
    # 打印表头
    print(f"{'Model Name':<35} | {'State':<8} | {'CER':<7} | {'Prec':<7} | {'Rec':<7} | {'F1':<7}")
    print("-" * 80)

    # 2. 并行执行
    # 使用 ProcessPoolExecutor 进行多进程计算
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        # 提交所有任务
        future_to_model = {executor.submit(process_single_model, p): p.name for p in tasks}
        
        # 使用 tqdm 包装，position=0 保证在底部
        pbar = tqdm(concurrent.futures.as_completed(future_to_model), total=len(tasks), 
                    unit="model", desc="Processing", ncols=100)
        
        for future in pbar:
            model_name = future_to_model[future]
            try:
                res = future.result()
                
                if "error" in res:
                    pbar.write(f"❌ {model_name:<35} | Error: {res['error']}")
                else:
                    # 格式化输出
                    status_icon = "⚡ New" if res['status'] == 'Computed' else "💾 Cached"
                    msg = (f"{model_name:<35} | {status_icon:<8} | "
                           f"{res['cer']:.2%} | {res['precision']:.2%} | "
                           f"{res['recall']:.2%} | {res['f1']:.2%}")
                    pbar.write(msg)
                    
            except Exception as e:
                pbar.write(f"💥 {model_name:<35} | Exception: {e}")
                traceback.print_exc()

    print("-" * 80)
    print(f"✅ 所有任务处理完毕。详细结果已保存在各子目录下。")

if __name__ == "__main__":
    # Windows/macOS 下 multiprocessing 需要此保护，Linux 下无害
    import multiprocessing
    multiprocessing.freeze_support()
    main()
