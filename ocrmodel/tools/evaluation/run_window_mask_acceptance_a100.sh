#!/usr/bin/env bash
# One immutable line100-window GT acceptance run on physical GPUs 0,1,2,3,4.
set -Eeuo pipefail
run_root="$1"
code_root="${run_root}/code/ocrmodel"
env_dir=/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/envs/glmocr_a100_py311_cu128
nvidia_env=/data3/yky/yangky_ocr_models/envs/anandasky
python="${env_dir}/bin/python3"
model=/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d
checkpoint=/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/training_runs/glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000
manifest=/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1_q32_sparse24/validation/manifest.char.jsonl
mkdir -p "${run_root}/status"
trap 'rc=$?; printf "{\"status\":\"failed\",\"exit_code\":%s}\n" "$rc" > "${run_root}/status/run.json"; exit "$rc"' ERR
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTHONPATH="${code_root}/src:${code_root}"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cuda_libraries="${env_dir}/lib/python3.11/site-packages/torch/lib"
arch="$(uname -m)"
[[ "${arch}" != arm64 ]] || arch=aarch64
cuda_libraries="/usr/local/cuda/targets/${arch}-linux/lib:${cuda_libraries}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
    [[ ! -d "${lib}" ]] || cuda_libraries="${cuda_libraries}:${lib}"
done
export LD_LIBRARY_PATH="${cuda_libraries}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${python}" - "${manifest}" "${checkpoint}" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch, transformers
from layout_ocr.window_mask_routing import WindowRoutingProfile
manifest, checkpoint = map(Path, sys.argv[1:])
profile = WindowRoutingProfile()
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == profile.validation_sha256
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
assert len(rows) == profile.validation_pages
assert all(any('line_index' in c for c in r.get('characters', [])) for r in rows)
assert (checkpoint / 'decoder_lora.safetensors').is_file()
print(json.dumps({'preflight': 'passed', 'pages': len(rows), 'torch': torch.__version__,
                  'transformers': transformers.__version__, 'physical_gpus': [0,1,2,3,4],
                  'test_manifest_read': False}), flush=True)
PY
# Query only the explicit allow-list. All five must pass, with no waiting.
utils="$(nvidia-smi -i 0,1,2,3,4 --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
mapfile -t values <<< "${utils}"
[[ "${#values[@]}" -eq 5 ]]
for util in "${values[@]}"; do
    [[ "${util}" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]]
    (( util < 50 ))
done
printf '%s\n' "${utils}" > "${run_root}/status/admission_utilization.txt"
printf '{"status":"running","phase":"validation_gt_window_5shards","physical_gpus":[0,1,2,3,4],"test_manifest_read":false}\n' > "${run_root}/status/run.json"
cd "${code_root}"
mkdir -p "${run_root}/shards"
pids=()
for gpu in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES="${gpu}" "${python}" tools/evaluation/evaluate_window_mask_routing.py \
        --model-path "${model}" --backbone-checkpoint "${checkpoint}" \
        --validation-manifest "${manifest}" --output-dir "${run_root}/shards/${gpu}" \
        --mode gt --device cuda:0 --shard-count 5 --shard-index "${gpu}" \
        > "${run_root}/shards/${gpu}.log" 2>&1 &
    pids+=("$!")
    printf '%s %s\n' "${gpu}" "$!" >> "${run_root}/status/worker_pids.txt"
done
failed=0
for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
done
(( failed == 0 ))
"${python}" tools/evaluation/merge_window_mask_shards.py --run-root "${run_root}"
"${python}" - "${run_root}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
summary = json.loads((root/'results/summary.json').read_text())
(root/'status/run.json').write_text(json.dumps({'status': 'complete', 'acceptance': summary['acceptance'],
                                             'validation': summary['validation'], 'test_manifest_read': False}))
PY
