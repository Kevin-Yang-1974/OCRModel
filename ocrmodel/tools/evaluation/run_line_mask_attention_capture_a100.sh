#!/usr/bin/env bash
set -Eeuo pipefail

run_root="${1:?usage: run_line_mask_attention_capture_a100.sh RUN_ROOT CODE_ROOT}"
code_root="${2:?usage: run_line_mask_attention_capture_a100.sh RUN_ROOT CODE_ROOT}"
expected_prefix=/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/diagnostics/
[[ "$run_root" == "$expected_prefix"* ]] || {
    echo "run root must stay under the personal diagnostics directory: $expected_prefix" >&2
    exit 2
}

env_root=/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/envs/glmocr_a100_py311_cu128
nvidia_root=/data3/yky/yangky_ocr_models/envs/anandasky/lib/python3.11/site-packages/nvidia
model_path=/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d
backbone_checkpoint=/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/training_runs/glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000
mask_checkpoint=/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_line_mask_v2_full_20260922_1940/epoch-8.pt
validation_manifest=/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1_q32_sparse24/validation/manifest.char.jsonl
physical_gpu=2

admission="$(${env_root}/bin/python - <<'PY'
import subprocess
rows = subprocess.check_output(
    ['nvidia-smi', '-i', '2', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
    text=True,
).splitlines()
if len(rows) != 1:
    raise SystemExit('admission rejected: expected exactly physical GPU 2')
value = int(rows[0].strip())
if value >= 50:
    raise SystemExit(f'admission rejected: physical GPU 2 utilization is {value}%, must be <50%')
print(value)
PY
)"

mkdir -p "$run_root/logs" "$run_root/tmp"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export OMP_NUM_THREADS=2
export TMPDIR="$run_root/tmp"
export PYTHONPATH="$code_root/src:$code_root/tools/evaluation${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:${env_root}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    [[ ! -d "${nvidia_root}/${component}/lib" ]] || export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${nvidia_root}/${component}/lib"
done

write_status() {
    local status="$1"
    local temporary="$run_root/launcher_status.json.tmp"
    "$env_root/bin/python" - "$temporary" "$status" "$admission" "$physical_gpu" <<'PY'
import json, sys, time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    'status': sys.argv[2], 'admission_utilization_gpu': int(sys.argv[3]),
    'physical_gpu': sys.argv[4], 'time': time.time(),
}))
PY
    mv "$temporary" "$run_root/launcher_status.json"
}

trap 'rc=$?; if [[ $rc -ne 0 ]]; then write_status failed || true; fi' EXIT
write_status running

"$env_root/bin/python" "$run_root/code/capture_line_mask_attention_cases.py" \
    --code-root "$code_root" \
    --model-path "$model_path" \
    --backbone-checkpoint "$backbone_checkpoint" \
    --mask-checkpoint "$mask_checkpoint" \
    --validation-manifest "$validation_manifest" \
    --baseline-predictions "$run_root/input/baseline-selected.jsonl" \
    --routed-predictions "$run_root/input/line-mask-selected.jsonl" \
    --selected-cases "$run_root/input/selected_cases.json" \
    --output-dir "$run_root/attention" \
    --device cuda:0 \
    --hard-token-cap 512 2>&1 | tee "$run_root/logs/capture.log"

write_status complete
