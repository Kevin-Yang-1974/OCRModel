#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo "usage: run_line_mask_v3_diagnostic_a100.sh RUN_ROOT CODE_ROOT DUNHUANG_TRAIN_MANIFEST DUNHUANG_VAL_TUNE_MANIFEST [SOURCE_GROUP_FIELD]" >&2
    exit 64
}

[[ $# -ge 4 && $# -le 5 ]] || usage
requested_root="$1"
code_root="$(realpath -e "$2")"
dunhuang_train_manifest="$(realpath -e "$3")"
dunhuang_val_tune_manifest="$(realpath -e "$4")"
source_group_field="${5:-}"
expected_root=/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/line_mask_v3
run_root="$(realpath -m "$requested_root")"
case "$run_root/" in
    "$expected_root/"*) ;;
    *) echo "run root must remain under $expected_root" >&2; exit 2 ;;
esac
[[ ! -e "$run_root" ]] || { echo "run root already exists: $run_root" >&2; exit 2; }
[[ -f "$code_root/tools/evaluation/diagnose_line_mask_v3.py" ]] || {
    echo "code root does not contain the v3 diagnostic evaluator" >&2; exit 2;
}

env_root=/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/envs/glmocr_a100_py311_cu128
nvidia_root=/data3/yky/yangky_ocr_models/envs/anandasky/lib/python3.11/site-packages/nvidia
model_path=/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d
mask_checkpoint=/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing/training_runs/glmocr_line_mask_v2_full_20260922_1940/epoch-8.pt
decoder_lora_checkpoint=/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/training_runs/glmocr_mthv2_sparse24_q32_layout_boxeq820_3000_from_boxeq58_a100_260916_v1/seed42/checkpoint-3000
mthv2_train_manifest=/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1/train/manifest.char.jsonl
mthv2_val_tune_manifest=/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1/validation/manifest.char.jsonl

mkdir -p "$run_root/logs" "$run_root/tmp" "$run_root/manifests" "$run_root/results"
export TMPDIR="$run_root/tmp"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2
export LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib:${env_root}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    [[ ! -d "${nvidia_root}/${component}/lib" ]] || export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${nvidia_root}/${component}/lib"
done

phase=source_snapshot
write_status() {
    local status="$1"
    local temporary="$run_root/launcher_status.json.tmp"
    "${env_root}/bin/python" - "$temporary" "$status" "$phase" "${2:-0}" <<'PY'
import json, sys, time
from pathlib import Path
path = Path(sys.argv[1])
path.write_text(json.dumps({"status": sys.argv[2], "phase": sys.argv[3],
                            "exit_code": int(sys.argv[4]), "time": time.time()}))
path.replace(path.with_suffix(""))
PY
}
trap 'rc=$?; if [[ $rc -ne 0 ]]; then write_status failed "$rc" || true; fi' EXIT
write_status running

source_root="$run_root/source/ocrmodel"
mkdir -p "$source_root"
snapshot_tar="$run_root/source/line-mask-v3-code.tar"
tar --exclude='__pycache__' --exclude='*.pyc' \
    -C "$code_root" -cf "$snapshot_tar" src tools/evaluation configs/line_mask_v3 pyproject.toml
tar -C "$source_root" -xf "$snapshot_tar"
sha256sum "$snapshot_tar" > "$run_root/source/source-tar.sha256"
"${env_root}/bin/python" - "$source_root" "$run_root/source/source-fingerprint.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, output = Path(sys.argv[1]), Path(sys.argv[2])
files = {}
for path in sorted(item for item in root.rglob('*') if item.is_file()):
    relative = path.relative_to(root).as_posix()
    files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
digest = hashlib.sha256()
for relative, file_hash in files.items():
    digest.update(relative.encode('utf-8'))
    digest.update(b'\0')
    digest.update(file_hash.encode('ascii'))
    digest.update(b'\n')
output.write_text(json.dumps({"source_tree_sha256": digest.hexdigest(),
                              "files": files}, indent=2), encoding='utf-8')
PY
code_root="$source_root"
export PYTHONPATH="$code_root/src:$code_root/tools/evaluation${PYTHONPATH:+:$PYTHONPATH}"

phase=paired_model_weight_fingerprints
"${env_root}/bin/python" - "$model_path" "$code_root/configs/line_mask_v3/diagnostic.json" \
    "$decoder_lora_checkpoint" "$run_root/base-model-fingerprint.json" \
    "$run_root/decoder-lora-fingerprint.json" > "$run_root/logs/model-fingerprints.log" 2>&1 <<'PY'
import json, sys
from pathlib import Path
model_path, config_path, lora_checkpoint, base_output, lora_output = map(Path, sys.argv[1:])
from layout_ocr.line_mask_v3_diagnostics import fingerprint_decoder_lora, fingerprint_safetensors
config = json.loads(config_path.read_text(encoding='utf-8'))
revision = config["shared_start"]["model_revision"]
base = fingerprint_safetensors(model_path, revision)
lora = fingerprint_decoder_lora(lora_checkpoint, revision)
expected = "ada4cdd3bb1f2f417d1c0a68054dc61e70ab59f215b3bc2018074b4f2fdcd5d5"
if lora["decoder_lora_sha256"] != expected:
    raise SystemExit("decoder LoRA fingerprint does not match the registered line-mask v2 start")
base_output.write_text(json.dumps(base, indent=2), encoding='utf-8')
lora_output.write_text(json.dumps(lora, indent=2), encoding='utf-8')
print(json.dumps({"model_revision": revision,
                  "base_model_weights_sha256": base["model_weights_sha256"],
                  "decoder_lora_sha256": lora["decoder_lora_sha256"],
                  "weight_files": len(base["safetensors"])}))
PY

phase=manifest_lock_and_overlap_audit
source_group_args=()
if [[ -n "$source_group_field" ]]; then source_group_args+=(--source-group-field "$source_group_field"); fi
"${env_root}/bin/python" "$code_root/tools/evaluation/prepare_line_mask_v3_diagnostic_manifests.py" \
    --domain mthv2 --train-manifest "$mthv2_train_manifest" \
    --val-tune-manifest "$mthv2_val_tune_manifest" \
    --base-model-weights-fingerprint "$run_root/base-model-fingerprint.json" \
    --decoder-lora-fingerprint "$run_root/decoder-lora-fingerprint.json" \
    --output-dir "$run_root/manifests/mthv2" "${source_group_args[@]}" \
    > "$run_root/logs/prepare-mthv2.log" 2>&1
"${env_root}/bin/python" "$code_root/tools/evaluation/prepare_line_mask_v3_diagnostic_manifests.py" \
    --domain dunhuang_local_gazetteer --train-manifest "$dunhuang_train_manifest" \
    --val-tune-manifest "$dunhuang_val_tune_manifest" \
    --base-model-weights-fingerprint "$run_root/base-model-fingerprint.json" \
    --decoder-lora-fingerprint "$run_root/decoder-lora-fingerprint.json" \
    --output-dir "$run_root/manifests/dunhuang_local_gazetteer" "${source_group_args[@]}" \
    > "$run_root/logs/prepare-dunhuang.log" 2>&1

phase=gpu_admission
admission="$("${env_root}/bin/python" - <<'PY'
import subprocess
rows = subprocess.check_output(
    ['nvidia-smi', '-i', '0,1,2,3,4', '--query-gpu=index,utilization.gpu',
     '--format=csv,noheader,nounits'], text=True
).splitlines()
values = []
for row in rows:
    parts = [part.strip() for part in row.split(',')]
    if len(parts) != 2:
        raise SystemExit('admission rejected: could not parse the allowed GPU set')
    values.append((int(parts[0]), int(parts[1])))
if [index for index, _ in values] != [0, 1, 2, 3, 4]:
    raise SystemExit('admission rejected: expected exactly physical GPUs 0 through 4')
if any(utilization >= 50 for _, utilization in values):
    raise SystemExit('admission rejected: all allowed GPUs must be strictly below 50%')
print(','.join(f'{index}:{utilization}' for index, utilization in values))
PY
)"
printf '{"phase":"gpu_admission","physical_gpu_utilization":"%s"}\n' "$admission" > "$run_root/admission.json"
df -Pk /data3 > "$run_root/data3-disk-space.txt"

run_worker() {
    local shard="$1" stage="$2" domain manifest protocol evidence output log gpu
    gpu="$shard"
    for domain in mthv2 dunhuang_local_gazetteer; do
        if [[ "$stage" == diagnostic32 ]]; then
            manifest="$run_root/manifests/$domain/selected-val-tune-32.jsonl"
            protocol="$run_root/manifests/$domain/protocol.json"
            evidence="$run_root/manifests/$domain/line-evidence.json"
            output="$run_root/results/$domain/shard-$shard"
            log="$run_root/logs/${domain}-worker-${shard}.log"
        else
            manifest="$run_root/manifests/$domain/full-val-tune/val-tune-full.jsonl"
            protocol="$run_root/manifests/$domain/full-val-tune/protocol.json"
            evidence="$run_root/manifests/$domain/full-val-tune/line-evidence.json"
            output="$run_root/results/full-val-tune/$domain/shard-$shard"
            log="$run_root/logs/full-val-tune-${domain}-worker-${shard}.log"
        fi
        CUDA_VISIBLE_DEVICES="$gpu" "${env_root}/bin/python" -u \
            "$code_root/tools/evaluation/diagnose_line_mask_v3.py" \
            --code-root "$code_root" --model-path "$model_path" \
            --base-model-weights-fingerprint "$run_root/base-model-fingerprint.json" \
            --decoder-lora-checkpoint "$decoder_lora_checkpoint" \
            --decoder-lora-fingerprint "$run_root/decoder-lora-fingerprint.json" \
            --mask-checkpoint "$mask_checkpoint" \
            --val-tune-manifest "$manifest" --diagnostic-protocol "$protocol" \
            --line-evidence "$evidence" \
            --output-root "$output" --shard-index "$shard" --shard-count 5 --device cuda:0 \
            > "$log" 2>&1
    done
}
run_five_shard_stage() {
    local stage="$1" pid worker_failure=0
    local worker_pids=()
    phase="five_gpu_${stage}"
    for shard in 0 1 2 3 4; do
        run_worker "$shard" "$stage" &
        worker_pids+=("$!")
    done
    for pid in "${worker_pids[@]}"; do
        wait "$pid" || worker_failure=1
    done
    if [[ "$worker_failure" -ne 0 ]]; then
        echo "one or more ${stage} workers failed; outputs are retained" >&2
        exit 1
    fi
}
run_five_shard_stage diagnostic32

phase=merge_and_preregistered_screen
for domain in mthv2 dunhuang_local_gazetteer; do
    "${env_root}/bin/python" -u "$code_root/tools/evaluation/merge_line_mask_v3_diagnostic.py" \
        --code-root "$code_root" --run-root "$run_root/results/$domain" \
        --val-tune-manifest "$run_root/manifests/$domain/selected-val-tune-32.jsonl" \
        --diagnostic-protocol "$run_root/manifests/$domain/protocol.json" \
        --line-evidence "$run_root/manifests/$domain/line-evidence.json" \
        > "$run_root/logs/merge-${domain}.log" 2>&1
done
"${env_root}/bin/python" -u "$code_root/tools/evaluation/select_line_mask_v3_diagnostic_candidate.py" \
    --mthv2-summary "$run_root/results/mthv2/summary.json" \
    --dunhuang-summary "$run_root/results/dunhuang_local_gazetteer/summary.json" \
    --output "$run_root/results/candidate-selection.json" \
    > "$run_root/logs/candidate-selection.log" 2>&1
phase=lock_full_val_tune
for domain in mthv2 dunhuang_local_gazetteer; do
    if [[ "$domain" == mthv2 ]]; then
        train_manifest="$mthv2_train_manifest"
        val_tune_manifest="$mthv2_val_tune_manifest"
    else
        train_manifest="$dunhuang_train_manifest"
        val_tune_manifest="$dunhuang_val_tune_manifest"
    fi
    "${env_root}/bin/python" -u \
        "$code_root/tools/evaluation/prepare_line_mask_v3_full_val_tune.py" \
        --domain "$domain" --train-manifest "$train_manifest" \
        --val-tune-manifest "$val_tune_manifest" \
        --screen-protocol "$run_root/manifests/$domain/protocol.json" \
        --candidate-selection "$run_root/results/candidate-selection.json" \
        --base-model-weights-fingerprint "$run_root/base-model-fingerprint.json" \
        --decoder-lora-fingerprint "$run_root/decoder-lora-fingerprint.json" \
        --output-dir "$run_root/manifests/$domain/full-val-tune" \
        > "$run_root/logs/lock-full-val-tune-${domain}.log" 2>&1
done
phase=full_val_tune_gpu_admission
admission="$(${env_root}/bin/python - <<'PY'
import subprocess
rows = subprocess.check_output(
    ['nvidia-smi', '-i', '0,1,2,3,4', '--query-gpu=index,utilization.gpu',
     '--format=csv,noheader,nounits'], text=True
).splitlines()
values = []
for row in rows:
    index, utilization = (int(part.strip()) for part in row.split(','))
    values.append((index, utilization))
if [index for index, _ in values] != [0, 1, 2, 3, 4] or any(value >= 50 for _, value in values):
    raise SystemExit('full val_tune admission rejected: all allowed GPUs must be strictly below 50%')
print(','.join(f'{index}:{value}' for index, value in values))
PY
)"
printf '{"phase":"full_val_tune_gpu_admission","physical_gpu_utilization":"%s"}\n' "$admission" > "$run_root/full-val-tune-admission.json"
run_five_shard_stage full_val_tune

phase=merge_full_val_tune
for domain in mthv2 dunhuang_local_gazetteer; do
    "${env_root}/bin/python" -u "$code_root/tools/evaluation/merge_line_mask_v3_diagnostic.py" \
        --code-root "$code_root" --run-root "$run_root/results/full-val-tune/$domain" \
        --val-tune-manifest "$run_root/manifests/$domain/full-val-tune/val-tune-full.jsonl" \
        --diagnostic-protocol "$run_root/manifests/$domain/full-val-tune/protocol.json" \
        --line-evidence "$run_root/manifests/$domain/full-val-tune/line-evidence.json" \
        > "$run_root/logs/merge-full-val-tune-${domain}.log" 2>&1
done
phase=complete
write_status complete 0
printf 'line-mask v3 stage-D diagnostic32 screen and full val_tune complete: %s\n' "$run_root"
