#!/usr/bin/env bash
# Run PaddleOCR-VL and MinerU serially on all five A100 GPUs, while OpenDoc
# runs independently on CPU, for the AncientDoc split5 test set.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
code_root="$(cd -- "${script_dir}/../.." && pwd -P)"

run_id="ancientdoc_sota_$(date +%Y%m%d_%H%M%S)"
run_root_base="/data3/yky/yangky_ocr_models/evaluation_runs/SOTA"
models_root="/data3/yky/yangky_ocr_models/models/sota"
label_json="/data4/hyf/backup/GOT-OCR2.0/reference-260707/AncientDoc/label_for_got_split5.json"
image_root="/data4/hyf/project/古籍/AncientDoc"
gpu_ids="0,1,2,3,4"
max_output_tokens="1536"
foreground=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) run_id="$2"; shift 2 ;;
        --run-root-base) run_root_base="$2"; shift 2 ;;
        --models-root) models_root="$2"; shift 2 ;;
        --label-json) label_json="$2"; shift 2 ;;
        --image-root) image_root="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --max-output-tokens) max_output_tokens="$2"; shift 2 ;;
        --foreground) foreground=1; shift ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done

[[ "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid run id" >&2; exit 64; }
[[ "$gpu_ids" =~ ^[0-9]+(,[0-9]+){4}$ ]] || { echo "exactly five GPU ids are required" >&2; exit 64; }
[[ "$max_output_tokens" =~ ^[1-9][0-9]*$ ]] || { echo "invalid max output tokens" >&2; exit 64; }
IFS=',' read -r -a gpu_array <<< "$gpu_ids"
declare -A seen_gpu=()
for gpu in "${gpu_array[@]}"; do
    [[ -z "${seen_gpu[$gpu]+present}" ]] || { echo "duplicate GPU id" >&2; exit 64; }
    seen_gpu["$gpu"]=1
done
for expected_gpu in 0 1 2 3 4; do
    [[ -n "${seen_gpu[$expected_gpu]+present}" ]] || { echo "GPU 0-4 must all be included" >&2; exit 64; }
done

run_root="${run_root_base}/${run_id}"
manifest="${run_root}/dataset/manifest.jsonl"
paddle_root="${models_root}/paddleocr_vl_1_6/c5630abae1d940eafe0697512a0325494b02ab42"
mineru_root="${models_root}/mineru2_5_pro/bff20d4ae2bf202df9f45284b4d43681555a97ed"
opendoc_root="${models_root}/opendoc_0_1b/a377e00d62c01b6544603e2a90f2cffe2a0388e1"

python_bin="${A100_SOTA_PYTHON:-/data3/yky/yangky_ocr_models/envs/anandasky/bin/python}"
paddle_site="${A100_PADDLE_SITE_PACKAGES:-/data3/yky/yangky_ocr_models/envs/sota/glm_overlay_20260823:/data3/yky/yangky_ocr_models/envs/anandasky/lib/python3.11/site-packages}"
mineru_site="${A100_MINERU_SITE_PACKAGES:-/data3/yky/yangky_ocr_models/envs/sota/mineru_overlay_20260913:/data3/yky/yangky_ocr_models/envs/anandasky/lib/python3.11/site-packages:/data3/yky/yangky_ocr_models/envs/sota/paddleocr_vl_1_6/site-packages}"
opendoc_site="${A100_OPENDOC_SITE_PACKAGES:-/data3/yky/yangky_ocr_models/envs/sota/opendoc_onnx_overlay_20260824:/data3/yky/yangky_ocr_models/envs/sota/opendoc_overlay_20260823:/data3/yky/yangky_ocr_models/envs/sota/opendoc_0_1b/site-packages:/data3/yky/yangky_ocr_models/envs/anandasky/lib/python3.11/site-packages}"
opendoc_pythonpath="${A100_OPENDOC_PYTHONPATH:-/data3/yky/yangky_ocr_models/envs/sota/opendoc_0_1b/site-packages}"
opendoc_cache="${A100_OPENDOC_ONNX_CACHE:-/data3/yky/yangky_ocr_models/models/sota/opendoc_onnx_cache_20260824}"

if (( foreground == 0 )); then
    tmux_session="${A100_SOTA_TMUX_SESSION:-${run_id}}"
    [[ "$tmux_session" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid tmux session" >&2; exit 64; }
    tmux has-session -t "$tmux_session" 2>/dev/null && {
        printf '{"event":"ancientdoc_sota_refused","error":"tmux_session_exists","session":"%s"}\n' "$tmux_session" >&2
        exit 73
    }
    startup_log="${run_root_base}/${run_id}.launcher.log"
    script_path="$(realpath "${BASH_SOURCE[0]}")"
    tmux new-session -d -s "$tmux_session" \
        "cd $(printf '%q' "$code_root") && exec bash $(printf '%q' "$script_path") --foreground --run-id $(printf '%q' "$run_id") --run-root-base $(printf '%q' "$run_root_base") --models-root $(printf '%q' "$models_root") --label-json $(printf '%q' "$label_json") --image-root $(printf '%q' "$image_root") --gpu-ids $(printf '%q' "$gpu_ids") --max-output-tokens $(printf '%q' "$max_output_tokens") >$(printf '%q' "$startup_log") 2>&1"
    sleep 5
    if tmux has-session -t "$tmux_session" 2>/dev/null; then
        printf '{"event":"ancientdoc_sota_started","session":"%s","run_id":"%s","run_root":"%s","gpu_ids":"%s","opendoc_device":"cpu"}\n' \
            "$tmux_session" "$run_id" "$run_root" "$gpu_ids"
        exit 0
    fi
    printf '{"event":"ancientdoc_sota_failed_to_start","session":"%s","run_id":"%s","startup_log":"%s"}\n' \
        "$tmux_session" "$run_id" "$startup_log" >&2
    exit 1
fi

[[ ! -e "$run_root" ]] || { printf '{"event":"ancientdoc_sota_refused","error":"run_root_exists","run_root":"%s"}\n' "$run_root" >&2; exit 73; }
[[ -x "$python_bin" ]] || { echo "missing Python: $python_bin" >&2; exit 66; }
[[ -f "$label_json" && -d "$image_root" ]] || { echo "missing AncientDoc source" >&2; exit 66; }
for model_root in "$paddle_root" "$mineru_root" "$opendoc_root"; do
    [[ -d "$model_root" ]] || { echo "missing model root: $model_root" >&2; exit 66; }
done

# Admission checks query only the five explicitly requested physical GPUs.
gpu_rows="$(nvidia-smi -i "$gpu_ids" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)"
declare -A observed=()
while IFS=',' read -r observed_id observed_util; do
    observed_id="${observed_id//[[:space:]]/}"
    observed_util="${observed_util//[[:space:]]/}"
    [[ "$observed_id" =~ ^[0-9]+$ && "$observed_util" =~ ^[0-9]+$ ]] || { echo "cannot parse GPU utilization" >&2; exit 79; }
    observed["$observed_id"]="$observed_util"
done <<< "$gpu_rows"
for gpu in "${gpu_array[@]}"; do
    [[ -n "${observed[$gpu]+present}" ]] || { echo "GPU was not reported: $gpu" >&2; exit 79; }
    (( observed[$gpu] < 50 )) || {
        printf '{"event":"ancientdoc_sota_refused","error":"gpu_admission_failed","gpu":"%s","utilization":%s,"limit":50}\n' "$gpu" "${observed[$gpu]}" >&2
        exit 79
    }
done

mkdir -p "${run_root}/dataset" "${run_root}/paddleocr_vl_1_6" "${run_root}/mineru2_5_pro" "${run_root}/opendoc_0_1b"
env PYTHONPATH="$code_root" PYTHONUNBUFFERED=1 "$python_bin" -m tools.sota.build_ancientdoc_manifest \
    --label-json "$label_json" --image-root "$image_root" --output "$manifest" \
    >"${run_root}/dataset/build_manifest.json" 2>&1
printf '{"event":"ancientdoc_sota_armed","run_id":"%s","dataset":"AncientDoc","source_split":"split5","split":"test","manifest":"%s","gpu_ids":"%s","opendoc_device":"cpu","max_output_tokens":%s,"test_used_for_selection":false}\n' \
    "$run_id" "$manifest" "$gpu_ids" "$max_output_tokens" >"${run_root}/launch.json"

run_model() {
    local model="$1" model_root="$2" device="$3" site="$4" output="$5" log="$6" visible_gpu="$7" dtype="$8" shard_index="$9" shard_count="${10}"
    local env_args=(
        "PYTHONPATH=${code_root}:${site}"
        "TOKENIZERS_PARALLELISM=false"
        "HF_HUB_OFFLINE=1"
        "TRANSFORMERS_OFFLINE=1"
        "PYTHONUNBUFFERED=1"
    )
    if [[ "$model" == "opendoc_0_1b" ]]; then
        env_args+=(
            "OPENOCR_ONNX=1"
            "OPENOCR_ONNX_CACHE=${opendoc_cache}"
            "OPENOCR_PYTHONPATH=${opendoc_pythonpath}"
        )
    fi
    env "${env_args[@]}" CUDA_VISIBLE_DEVICES="$visible_gpu" \
        "$python_bin" -m tools.sota.run_ancientdoc_zero_shot \
        --model "$model" --model-root "$model_root" --manifest "$manifest" \
        --image-root "$image_root" --source-label "$label_json" --output-dir "$output" \
        --device "$device" --dtype "$dtype" --max-output-tokens "$max_output_tokens" \
        --progress-every 5 --shard-index "$shard_index" --shard-count "$shard_count" \
        --allow-test >"$log" 2>&1
}

run_sharded_model() {
    local model="$1" model_root="$2" site="$3"
    local model_root_dir="${run_root}/${model}/test"
    local pids=() index gpu
    for index in 0 1 2 3 4; do
        gpu="${gpu_array[$index]}"
        run_model "$model" "$model_root" cuda:0 "$site" \
            "${model_root_dir}/shard-$(printf '%03d' "$index")" \
            "${run_root}/${model}/shard-$(printf '%03d' "$index").log" \
            "$gpu" bf16 "$index" 5 &
        pids+=("$!")
    done
    local rc=0 child_rc pid
    for pid in "${pids[@]}"; do
        wait "$pid"
        child_rc=$?
        (( child_rc == 0 )) || rc=$child_rc
    done
    return "$rc"
}

merge_model() {
    local model="$1"
    local model_root_dir="${run_root}/${model}/test"
    env PYTHONPATH="$code_root" PYTHONUNBUFFERED=1 "$python_bin" \
        "${code_root}/tools/sota/merge_predictions.py" \
        --manifest "$manifest" --shard-root "$model_root_dir" \
        --output "${model_root_dir}/predictions.jsonl" --expected-pages 516 \
        >"${run_root}/${model}/merge.log" 2>&1 || return $?
    env PYTHONPATH="$code_root" PYTHONUNBUFFERED=1 "$python_bin" \
        "${code_root}/tools/sota/summarize_metrics.py" \
        --manifest "$manifest" --predictions "${model_root_dir}/predictions.jsonl" \
        --output "${model_root_dir}/unified_metrics.json" --expected-pages 516 \
        >"${run_root}/${model}/metrics.log" 2>&1
}

set +e
# OpenDoc is CPU-only and may progress while the two GPU suites run.
run_model opendoc_0_1b "$opendoc_root" cpu "$opendoc_site" \
    "${run_root}/opendoc_0_1b/test" "${run_root}/opendoc_0_1b/run.log" \
    "" fp32 0 1 &
opendoc_pid=$!

# PaddleOCR and MinerU are intentionally serial; each one uses all five GPUs.
run_sharded_model paddleocr_vl_1_6 "$paddle_root" "$paddle_site"
paddle_rc=$?
if (( paddle_rc == 0 )); then
    merge_model paddleocr_vl_1_6
    paddle_merge_rc=$?
else
    paddle_merge_rc=3
fi

run_sharded_model mineru2_5_pro "$mineru_root" "$mineru_site"
mineru_rc=$?
if (( mineru_rc == 0 )); then
    merge_model mineru2_5_pro
    mineru_merge_rc=$?
else
    mineru_merge_rc=3
fi

wait "$opendoc_pid"
opendoc_rc=$?
if (( opendoc_rc == 0 )); then
    env PYTHONPATH="$code_root" PYTHONUNBUFFERED=1 "$python_bin" \
        "${code_root}/tools/sota/summarize_metrics.py" \
        --manifest "$manifest" --predictions "${run_root}/opendoc_0_1b/test/predictions.jsonl" \
        --output "${run_root}/opendoc_0_1b/test/unified_metrics.json" --expected-pages 516 \
        >"${run_root}/opendoc_0_1b/metrics.log" 2>&1
    opendoc_metrics_rc=$?
else
    opendoc_metrics_rc=3
fi
set -e

printf '{"event":"ancientdoc_sota_inference_finished","run_id":"%s","paddle_rc":%s,"paddle_merge_rc":%s,"mineru_rc":%s,"mineru_merge_rc":%s,"opendoc_rc":%s,"opendoc_metrics_rc":%s,"test_used_for_selection":false}\n' \
    "$run_id" "$paddle_rc" "$paddle_merge_rc" "$mineru_rc" "$mineru_merge_rc" "$opendoc_rc" "$opendoc_metrics_rc" >"${run_root}/inference_finished.json"

if (( paddle_rc != 0 || paddle_merge_rc != 0 || mineru_rc != 0 || mineru_merge_rc != 0 || opendoc_rc != 0 || opendoc_metrics_rc != 0 )); then
    printf '%s\n' '{"event":"ancientdoc_sota_failed","reason":"one_or_more_model_runs_or_merges_failed"}' >"${run_root}/failed.json"
    exit 3
fi

env PYTHONPATH="$code_root" PYTHONUNBUFFERED=1 "$python_bin" \
    "${code_root}/tools/sota/format_ancientdoc_results.py" \
    --run-root "$run_root" --run-id "$run_id" --manifest "$manifest" \
    --source-label "$label_json" --image-root "$image_root" \
    --output "${run_root}/RESULTS.md" --paddle-gpu "$gpu_ids" \
    --mineru-gpu "$gpu_ids" --max-output-tokens "$max_output_tokens" \
    >"${run_root}/format.log" 2>&1

printf '{"event":"ancientdoc_sota_complete","run_id":"%s","run_root":"%s","results":"%s/RESULTS.md","test_used_for_selection":false}\n' \
    "$run_id" "$run_root" "$run_root" | tee "${run_root}/completed.json"
