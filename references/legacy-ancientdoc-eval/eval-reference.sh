#!/bin/bash

# 单卡流式处理
export CUDA_VISIBLE_DEVICES=0
time \
python GOT/eval/myeval.py \
--model-name model_weights/original \
--gtfile_path data/train/AncientDoc/label_for_got_split5.json \
--image_path  data/train/AncientDoc \
--out_file results/test.json 
