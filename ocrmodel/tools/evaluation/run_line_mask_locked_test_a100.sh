#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p "$(dirname "$1")"
run_root="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
code_root="${2:-/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/code/line_mask_20260922_v4/ocrmodel}"
script_root="$(cd "$(dirname "$0")" && pwd)"
env_root=/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/envs/glmocr_a100_py311_cu128
nvidia_root=/data3/yky/yangky_ocr_models/envs/anandasky/lib/python3.11/site-packages/nvidia
model_path=/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d
backbone_checkpoint=/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/training_runs/glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000
mask_run=/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_line_mask_v2_full_20260922_1940
mask_checkpoint="$mask_run/epoch-8.pt"
selection="$mask_run/selection.json"
test_manifest=/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1/test/manifest.jsonl

mkdir -p "$run_root" "$run_root/tmp" "$run_root/logs" "$run_root/shards" "$run_root/results" "$run_root/code"
export TMPDIR="$run_root/tmp"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2
export PYTHONPATH="$code_root/src:$code_root/tools/evaluation${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:${env_root}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    [[ ! -d "${nvidia_root}/${component}/lib" ]] || export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${nvidia_root}/${component}/lib"
done

write_status() {
    local status="$1"
    local code="${2:-0}"
    "$env_root/bin/python" - "$run_root/launcher_status.json" "$status" "$code" <<'PY'
import json, sys, time
from pathlib import Path
path = Path(sys.argv[1])
temporary = path.with_suffix('.json.tmp')
temporary.write_text(json.dumps({"status": sys.argv[2], "exit_code": int(sys.argv[3]), "time": time.time()}))
temporary.replace(path)
PY
}

trap 'rc=$?; if [[ $rc -ne 0 ]]; then write_status failed "$rc"; fi' EXIT

# Admission is restricted to the five explicitly authorized physical GPUs.
util="$(${env_root}/bin/python - <<'PY'
import subprocess
rows = subprocess.check_output(
    ['nvidia-smi', '-i', '0,1,2,3,4', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
    text=True,
).splitlines()
if len(rows) != 5:
    raise SystemExit('admission rejected: expected exactly five allowed GPUs')
values = [int(value.strip()) for value in rows]
if any(value >= 50 for value in values):
    raise SystemExit('admission rejected: every allowed GPU must be strictly below 50%')
print(','.join(map(str, values)))
PY
)"
printf '{"status":"running","phase":"preflight","physical_gpus":"0,1,2,3,4","admission_utilization":"%s"}\n' "$util" > "$run_root/launcher_status.json"

# Reconfirm the locked selection and full official test coverage before launch.
"$env_root/bin/python" - "$run_root/protocol.json" "$selection" "$mask_checkpoint" "$backbone_checkpoint" "$test_manifest" "$model_path" "$code_root" "$script_root/evaluate_line_mask_locked_test.py" "$script_root/merge_line_mask_locked_test.py" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, selection_path, mask_path, backbone, manifest, model, code, evaluator, merger = map(Path, sys.argv[1:])
def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
selection = json.loads(selection_path.read_text(encoding='utf-8'))
run_status = json.loads((mask_path.parent / 'status.json').read_text(encoding='utf-8'))
if run_status.get('status') != 'complete':
    raise SystemExit('source training run is not complete')
if selection.get('epoch') != 8 or selection.get('step') != 3456:
    raise SystemExit('locked selection is not epoch 8 / step 3456')
if selection.get('test_manifest_read') is not False or selection.get('test_used_for_selection') is not False:
    raise SystemExit('selection metadata does not preserve the test boundary')
if Path(selection.get('checkpoint', '')).resolve() != mask_path.resolve():
    raise SystemExit('selection checkpoint path mismatch')
if not model.is_dir() or not (backbone / 'decoder_lora.safetensors').is_file():
    raise SystemExit('base model or backbone checkpoint is missing')
rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
ids = [str(row.get('page_id', '')) for row in rows]
if len(rows) != 800 or len(set(ids)) != 800:
    raise SystemExit('official test manifest must have 800 unique pages')
if any(row.get('split', row.get('official_split')) != 'test' for row in rows):
    raise SystemExit('test manifest contains a page outside the official test split')
for row in rows:
    image = Path(row.get('image_path') or row.get('image') or '')
    if not image.is_absolute():
        image = manifest.parent / image
    if not image.is_file():
        raise SystemExit(f'missing test image for {row.get("page_id")}')
if not (code / 'src/layout_ocr/line_mask_head.py').is_file() or not (code / 'src/layout_ocr/line_mask_runtime.py').is_file():
    raise SystemExit('training code snapshot does not contain the selected head/runtime')
protocol = {
    'status': 'locked_before_inference',
    'dataset': 'MTHv2 full official test',
    'pages': 800,
    'test_manifest': str(manifest),
    'test_manifest_sha256': sha(manifest),
    'test_manifest_read': True,
    'test_used_for_selection': False,
    'selection': str(selection_path),
    'selection_epoch': selection['epoch'],
    'selection_step': selection['step'],
    'selected_validation_cer': selection['validation']['cer'],
    'mask_checkpoint': str(mask_path),
    'mask_checkpoint_sha256': sha(mask_path),
    'backbone_checkpoint': str(backbone),
    'backbone_lora_sha256': sha(backbone / 'decoder_lora.safetensors'),
    'model_path': str(model),
    'code_root': str(code),
    'training_source_tar_sha256': '838c06d66617a001210731043ac735dc5d240cb6cc46148168c3fb8841ff4c26',
    'evaluator_sha256': sha(evaluator),
    'merger_sha256': sha(merger),
    'generation': {'input': 'full-page image plus fixed Text Recognition prompt only', 'max_pixels': 4000000,
                   'max_new_tokens': 1536, 'do_sample': False, 'precision': 'bfloat16',
                   'attention_backend': 'sdpa', 'processor': 'fast', 'seed': 42},
    'arms': {'baseline': 'same backbone LoRA; recurrent line-mask routing disabled',
             'line_mask_epoch8_step3456': 'same backbone LoRA plus selected learned line-mask head'},
    'execution': {'physical_gpus': [0, 1, 2, 3, 4], 'workers': 5,
                  'pages_per_shard': [160, 160, 160, 160, 160],
                  'both_arms_run_in_each_worker': True},
}
temporary = out.with_suffix('.json.tmp')
temporary.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
temporary.replace(out)
print(json.dumps({'pages': len(rows), 'test_manifest_sha256': protocol['test_manifest_sha256'],
                  'checkpoint_sha256': protocol['mask_checkpoint_sha256']}, ensure_ascii=False))
PY

worker_pids=()
for shard in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES="$shard" "$env_root/bin/python" -u "$script_root/evaluate_line_mask_locked_test.py" \
        --code-root "$code_root" \
        --model-path "$model_path" \
        --backbone-checkpoint "$backbone_checkpoint" \
        --mask-checkpoint "$mask_checkpoint" \
        --selection "$selection" \
        --test-manifest "$test_manifest" \
        --output-root "$run_root/shards" \
        --shard-index "$shard" --shard-count 5 --device cuda:0 \
        > "$run_root/logs/worker-${shard}.log" 2>&1 &
    worker_pids+=("$!")
done

worker_failure=0
for pid in "${worker_pids[@]}"; do
    wait "$pid" || worker_failure=1
done
if [[ "$worker_failure" -ne 0 ]]; then
    printf '{"status":"failed","phase":"workers"}\n' > "$run_root/launcher_status.json"
    exit 1
fi

"$env_root/bin/python" -u "$script_root/merge_line_mask_locked_test.py" \
    --code-root "$code_root" \
    --run-root "$run_root" \
    --test-manifest "$test_manifest" \
    --selection "$selection" \
    --mask-checkpoint "$mask_checkpoint" \
    --backbone-checkpoint "$backbone_checkpoint" \
    --train-manifest /data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1/train/manifest.char.jsonl \
    > "$run_root/logs/merge.log" 2>&1
write_status complete 0
printf 'locked test complete: %s\n' "$run_root"
