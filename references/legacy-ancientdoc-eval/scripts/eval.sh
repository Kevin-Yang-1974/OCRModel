#!/bin/bash

# 单卡流式处理
export CUDA_VISIBLE_DEVICES=0
time \
python GOT/eval/myeval.py \
--model-name model_weights/Ancient/260204_element-gate_lambda-1_train1-2_test3 \
--gtfile_path data/train/AncientDoc/label_for_got_split5.json \
--image_path  data/train/AncientDoc \
--out_file results/test.json 

# 可视化
# export CUDA_VISIBLE_DEVICES=0
# time \
# python GOT/eval/myeval_visual.py \
# --model-name /work/project/GOT-OCR2.0/GOT-OCR-2.0-master/model_weights/Ancient/260102 \
# --gtfile_path /work/project/GOT-OCR2.0/GOT-OCR-2.0-master/data/train/AncientDoc/label_for_got.json \
# --image_path  /work/project/GOT-OCR2.0/GOT-OCR-2.0-master/data/train/AncientDoc \
# --out_file results/AncientDoc/results_260102.json \


# 官方脚本
# time \
# python GOT/eval/evaluate_GOT.py \
# --model-name model_weights/original \
# --gtfile_path /work/project/GOT-OCR2.0/GOT-OCR-2.0-master/data/test/chaos_images/label.json \
# --image_path  /work/project/GOT-OCR2.0/GOT-OCR-2.0-master/data/test/chaos_images/images \
# --out_path results/chaos-original \
# --num-chunks 8 \

# 官方脚本
# time \
# python GOT/eval/evaluate_GOT.py \
# --model-name model_weights/Ancient/260123_baseline_train1_test2 \
# --gtfile_path /work/project/GOT-OCR2.0/GOT-OCR-2.0-master/data/train/AncientDoc/label_for_got_split2.json \
# --image_path  /work/project/GOT-OCR2.0/GOT-OCR-2.0-master/data/train/AncientDoc \
# --out_path results/Ancient/260123_baseline_train1_test2 \
# --num-chunks 8 \
