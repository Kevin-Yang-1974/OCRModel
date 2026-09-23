#!/usr/bin/env bash
set -Eeuo pipefail
code_root="$(cd "$(dirname "$0")/../.." && pwd)"
run_root="$1"
mode="${2:-smoke}"
export TMPDIR="${run_root}.tmp"
mkdir -p "$TMPDIR"
env_root=/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/envs/glmocr_a100_py311_cu128
nvidia_root=/data3/yky/yangky_ocr_models/envs/anandasky/lib/python3.11/site-packages/nvidia
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=2
export PYTHONPATH="${code_root}/src"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:${env_root}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    [[ ! -d "${nvidia_root}/${component}/lib" ]] || export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${nvidia_root}/${component}/lib"
done
"${env_root}/bin/python" - <<'PY'
import subprocess
rows = subprocess.check_output(['nvidia-smi','-i','0,1,2,3,4','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'], text=True).splitlines()
if len(rows) != 5 or any(int(row.strip()) >= 50 for row in rows):
    raise SystemExit('admission rejected: every allowed GPU must be strictly below 50%')
print('GPU admission passed for physical 0,1,2,3,4', flush=True)
PY
extra=()
case "$mode" in smoke) extra+=(--smoke);; train) ;; *) exit 64;; esac
mkdir -p "$(dirname "$run_root")"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then mkdir -p "$run_root"; printf "{\"status\":\"failed\",\"exit_code\":%s}\n" "$rc" > "$run_root/launcher_status.json"; fi' EXIT
"${env_root}/bin/python" -m torch.distributed.run --standalone --nproc_per_node=5 \
    "${code_root}/tools/training/train_line_mask_ddp.py" \
    --model-path /data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d \
    --backbone-checkpoint /data3/yky/yangky_ocr_models/glm_ocr_layout_ot/training_runs/glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000 \
    --train-manifest /data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1/train/manifest.char.jsonl \
    --validation-manifest /data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1_q32_sparse24/validation/manifest.char.jsonl \
    --output-dir "$run_root" "${extra[@]}"
