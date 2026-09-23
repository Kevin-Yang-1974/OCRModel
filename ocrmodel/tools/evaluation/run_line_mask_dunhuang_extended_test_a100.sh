#!/usr/bin/env bash
set -Eeuo pipefail
root=/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing
run_id=glmocr_dunhuang_local_plus_new77_line_mask_20260923
run_root="${root}/locked_tests/${run_id}"
session=glmocr_dunhuang_new77_mask_20260923
code_root=/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/code/line_mask_20260922_v4/ocrmodel
dataset_root=/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1_portable
manifest_root="${dataset_root}/manifests"
sample_root=/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/additional_77_20260923/source/20260820
backbone_checkpoint=/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/training_runs/glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000
mask_run=/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_line_mask_v2_full_20260922_1940
mask_checkpoint="${mask_run}/epoch-8.pt"
selection="${mask_run}/selection.json"
model_path=/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d
env_root=/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/envs/glmocr_a100_py311_cu128
driver_root="${run_root}/code"
foreground=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --foreground) foreground=1; shift ;;
        *) exit 64 ;;
    esac
done

write_status() {
    "${env_root}/bin/python" - "${run_root}/launcher_status.json" "$1" "$2" "${3:-0}" <<'PY'
import json, sys, time
from pathlib import Path
p = Path(sys.argv[1]); t = p.with_suffix('.json.tmp')
t.write_text(json.dumps({"status": sys.argv[2], "phase": sys.argv[3], "exit_code": int(sys.argv[4]), "time": time.time()}))
t.replace(p)
PY
}
current_phase=preflight
on_error() { rc=$?; write_status failed "${current_phase}_failed" "$rc" || true; exit "$rc"; }
trap on_error ERR

if (( foreground == 0 )); then
    command -v tmux >/dev/null 2>&1 || exit 69
    tmux has-session -t "${session}" 2>/dev/null && exit 73
    [[ -d "${driver_root}" ]] || exit 66
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    tmux new-session -d -s "${session}" "cd $(printf '%q' "${driver_root}") && exec bash $(printf '%q' "${script_path}") --foreground >$(printf '%q' "${run_root}/launcher.log") 2>&1"
    printf '{"event":"line_mask_dunhuang_extended_test_armed","session":"%s","run_id":"%s","physical_gpus":"0,1,2,3,4"}\n' "${session}" "${run_id}"
    exit 0
fi

mkdir -p "${run_root}/tmp" "${run_root}/logs" "${run_root}/shards" "${run_root}/inputs"
export TMPDIR="${run_root}/tmp" CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2
export PYTHONPATH="${code_root}/src:${code_root}/tools/evaluation:${driver_root}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:${env_root}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
nvidia_root=/data3/yky/yangky_ocr_models/envs/anandasky/lib/python3.11/site-packages/nvidia
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    [[ ! -d "${nvidia_root}/${component}/lib" ]] || export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${nvidia_root}/${component}/lib"
done

[[ -x "${env_root}/bin/python" && -d "${code_root}/src/layout_ocr" ]] || exit 66
[[ -f "${manifest_root}/test.jsonl" && -f "${manifest_root}/train.jsonl" && -f "${manifest_root}/validation.jsonl" ]] || exit 66
[[ -d "${sample_root}/img" && -d "${sample_root}/work" ]] || exit 66
[[ -f "${selection}" && -f "${mask_checkpoint}" && -d "${backbone_checkpoint}" && -d "${model_path}" ]] || exit 66
[[ ! -e "${run_root}/launcher_status.json" && ! -e "${run_root}/summary.json" ]] || exit 73
for f in prepare_line_mask_dunhuang_extended_test.py evaluate_line_mask_dunhuang_extended_test.py merge_line_mask_dunhuang_extended_test.py; do
    [[ -f "${driver_root}/${f}" ]] || exit 66
done

# Read utilization.gpu only on physical GPUs 0-4 before admission.
util="$(${env_root}/bin/python - <<'PY'
import subprocess
rows = subprocess.check_output(['nvidia-smi', '-i', '0,1,2,3,4', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()
if len(rows) != 5: raise SystemExit('expected exactly GPUs 0-4')
values = [int(value.strip()) for value in rows]
if any(value >= 50 for value in values): raise SystemExit('admission rejected: allowed GPU utilization must be below 50%')
print(','.join(map(str, values)))
PY
)"
write_status running preflight 0

current_phase=prepare_test_extension
"${env_root}/bin/python" -u "${driver_root}/prepare_line_mask_dunhuang_extended_test.py" \
    --dataset-root "${dataset_root}" --train-manifest "${manifest_root}/train.jsonl" \
    --validation-manifest "${manifest_root}/validation.jsonl" --test-manifest "${manifest_root}/test.jsonl" \
    --new-sample-root "${sample_root}" --output-root "${run_root}/inputs" --expected-new-pages 77 \
    > "${run_root}/logs/prepare-test-extension.log" 2>&1
cp "${run_root}/inputs/protocol.json" "${run_root}/protocol.json"
cp "${run_root}/inputs/expanded-test-manifest.jsonl" "${run_root}/expanded-test-manifest.jsonl"

current_phase=lock_protocol
"${env_root}/bin/python" - "${run_root}/protocol.json" "${selection}" "${mask_checkpoint}" "${backbone_checkpoint}" "${driver_root}" "${model_path}" "${util}" <<'PY'
import hashlib, json, sys
from pathlib import Path
protocol_path, selection_path, mask_path, backbone, driver_root, model_path = map(Path, sys.argv[1:7]); util = sys.argv[7]
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()
selection = json.loads(selection_path.read_text(encoding='utf-8'))
if selection.get('epoch') != 8 or selection.get('step') != 3456 or selection.get('test_manifest_read') is not False or selection.get('test_used_for_selection') is not False:
    raise SystemExit('selection is not validation-locked epoch8/step3456')
p = json.loads(protocol_path.read_text(encoding='utf-8'))
p.update({'status':'locked_before_inference','selection':str(selection_path),'selection_epoch':8,'selection_step':3456,
           'selected_validation_cer':selection.get('validation',{}).get('cer'),'mask_checkpoint':str(mask_path),
           'mask_checkpoint_sha256':sha(mask_path),'backbone_checkpoint':str(backbone),
           'backbone_lora_sha256':sha(backbone/'decoder_lora.safetensors'),'model_revision':'ca5d8b3e287e52589e37c28385d9655ee4372f9d',
           'model_path':str(model_path),'driver_root':str(driver_root),
           'driver_sha256':{n:sha(driver_root/n) for n in ('prepare_line_mask_dunhuang_extended_test.py','evaluate_line_mask_dunhuang_extended_test.py','merge_line_mask_dunhuang_extended_test.py','run_line_mask_dunhuang_extended_test_a100.sh')},
           'execution':{'physical_gpus':[0,1,2,3,4],'workers':5,'pages_per_shard':'round-robin balanced across five GPUs',
                        'admission_utilization_gpu0_to_4':[int(v) for v in util.split(',')],
                        'arms':{'baseline':'shared checkpoint-3000 decoder LoRA; routing disabled',
                                'line_mask_epoch8_step3456':'same backbone LoRA plus validation-selected line-mask head'}}})
t = protocol_path.with_suffix('.json.tmp'); t.write_text(json.dumps(p,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8'); t.replace(protocol_path)
PY

current_phase=workers
worker_pids=()
for shard in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES="${shard}" "${env_root}/bin/python" -u "${driver_root}/evaluate_line_mask_dunhuang_extended_test.py" \
        --code-root "${code_root}" --model-path "${model_path}" --backbone-checkpoint "${backbone_checkpoint}" \
        --mask-checkpoint "${mask_checkpoint}" --selection "${selection}" --protocol "${run_root}/protocol.json" \
        --test-manifest "${run_root}/expanded-test-manifest.jsonl" --output-root "${run_root}/shards" \
        --shard-index "${shard}" --shard-count 5 --device cuda:0 > "${run_root}/logs/worker-${shard}.log" 2>&1 &
    worker_pids+=("$!")
done
worker_failure=0
for pid in "${worker_pids[@]}"; do
    wait "${pid}" || worker_failure=1
done
if (( worker_failure != 0 )); then write_status failed workers 1; exit 1; fi

current_phase=merge
"${env_root}/bin/python" -u "${driver_root}/merge_line_mask_dunhuang_extended_test.py" \
    --code-root "${code_root}" --run-root "${run_root}" --test-manifest "${run_root}/expanded-test-manifest.jsonl" \
    --protocol "${run_root}/protocol.json" --shard-count 5 > "${run_root}/logs/merge.log" 2>&1
current_phase=verify_summary
"${env_root}/bin/python" - "${run_root}/summary.json" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
if s.get('status') != 'complete' or s.get('expanded_test_pages') != 136 or s.get('labeled_test_pages') != 59 or s.get('unscored_new_pages') != 77 or s.get('test_used_for_selection') is not False:
    raise SystemExit('expanded-test coverage or protocol verification failed')
for arm in ('baseline','line_mask_epoch8_step3456'):
    if s['arms'][arm]['all_prediction_pages'] != 136 or s['arms'][arm]['labeled_test_metrics']['pages'] != 59:
        raise SystemExit(f'{arm} coverage mismatch')
PY
write_status complete complete 0
printf 'extended Dunhuang locked test complete: %s\n' "${run_root}"
