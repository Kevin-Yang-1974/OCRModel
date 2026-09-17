#!/usr/bin/env bash
# Re-run the MinerU AncientDoc split5 test after a provider/runtime fix.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
code_root="$(cd -- "${script_dir}/../.." && pwd -P)"
run_id="ancientdoc_mineru_retest_$(date +%Y%m%d_%H%M%S)"
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
mineru_root="${models_root}/mineru2_5_pro/bff20d4ae2bf202df9f45284b4d43681555a97ed"
python_bin="${A100_SOTA_PYTHON:-/data3/yky/yangky_ocr_models/envs/anandasky/bin/python}"
mineru_site="${A100_MINERU_SITE_PACKAGES:-/data3/yky/yangky_ocr_models/envs/sota/mineru_overlay_20260913:/data3/yky/yangky_ocr_models/envs/anandasky/lib/python3.11/site-packages:/data3/yky/yangky_ocr_models/envs/sota/paddleocr_vl_1_6/site-packages}"

if (( foreground == 0 )); then
    tmux_session="${A100_SOTA_TMUX_SESSION:-${run_id}}"
    [[ "$tmux_session" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid tmux session" >&2; exit 64; }
    tmux has-session -t "$tmux_session" 2>/dev/null && {
        printf '{"event":"ancientdoc_mineru_retest_refused","error":"tmux_session_exists","session":"%s"}\n' "$tmux_session" >&2
        exit 73
    }
    startup_log="${run_root_base}/${run_id}.launcher.log"
    script_path="$(realpath "${BASH_SOURCE[0]}")"
    tmux new-session -d -s "$tmux_session" \
        "cd $(printf '%q' "$code_root") && exec bash $(printf '%q' "$script_path") --foreground --run-id $(printf '%q' "$run_id") --run-root-base $(printf '%q' "$run_root_base") --models-root $(printf '%q' "$models_root") --label-json $(printf '%q' "$label_json") --image-root $(printf '%q' "$image_root") --gpu-ids $(printf '%q' "$gpu_ids") --max-output-tokens $(printf '%q' "$max_output_tokens") >$(printf '%q' "$startup_log") 2>&1"
    sleep 5
    if tmux has-session -t "$tmux_session" 2>/dev/null; then
        printf '{"event":"ancientdoc_mineru_retest_started","session":"%s","run_id":"%s","run_root":"%s","gpu_ids":"%s"}\n' "$tmux_session" "$run_id" "$run_root" "$gpu_ids"
        exit 0
    fi
    printf '{"event":"ancientdoc_mineru_retest_failed_to_start","session":"%s","run_id":"%s","startup_log":"%s"}\n' "$tmux_session" "$run_id" "$startup_log" >&2
    exit 1
fi

[[ ! -e "$run_root" ]] || { printf '{"event":"ancientdoc_mineru_retest_refused","error":"run_root_exists","run_root":"%s"}\n' "$run_root" >&2; exit 73; }
[[ -x "$python_bin" ]] || { echo "missing Python: $python_bin" >&2; exit 66; }
[[ -f "$label_json" && -d "$image_root" ]] || { echo "missing AncientDoc source" >&2; exit 66; }
[[ -d "$mineru_root" ]] || { echo "missing MinerU model root: $mineru_root" >&2; exit 66; }

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
        printf '{"event":"ancientdoc_mineru_retest_refused","error":"gpu_admission_failed","gpu":"%s","utilization":%s,"limit":50}\n' "$gpu" "${observed[$gpu]}" >&2
        exit 79
    }
done

mkdir -p "${run_root}/dataset" "${run_root}/mineru2_5_pro"
env PYTHONPATH="$code_root" PYTHONUNBUFFERED=1 "$python_bin" -m tools.sota.build_ancientdoc_manifest \
    --label-json "$label_json" --image-root "$image_root" --output "$manifest" \
    >"${run_root}/dataset/build_manifest.json" 2>&1
printf '{"event":"ancientdoc_mineru_retest_armed","run_id":"%s","dataset":"AncientDoc","source_split":"split5","split":"test","manifest":"%s","gpu_ids":"%s","max_output_tokens":%s,"test_used_for_selection":false}\n' \
    "$run_id" "$manifest" "$gpu_ids" "$max_output_tokens" >"${run_root}/launch.json"

run_shard() {
    local index="$1" gpu="$2"
    local output="${run_root}/mineru2_5_pro/test/shard-$(printf '%03d' "$index")"
    local log="${run_root}/mineru2_5_pro/shard-$(printf '%03d' "$index").log"
    env PYTHONPATH="$code_root:${mineru_site}" \
        TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
        CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m tools.sota.run_ancientdoc_zero_shot \
        --model mineru2_5_pro --model-root "$mineru_root" --manifest "$manifest" \
        --image-root "$image_root" --source-label "$label_json" --output-dir "$output" \
        --device cuda:0 --dtype bf16 --max-output-tokens "$max_output_tokens" \
        --progress-every 5 --shard-index "$index" --shard-count 5 --allow-test \
        >"$log" 2>&1
}

set +e
pids=()
for index in 0 1 2 3 4; do
    run_shard "$index" "${gpu_array[$index]}" &
    pids+=("$!")
done
shard_rc=0
for pid in "${pids[@]}"; do
    if wait "$pid"; then
        :
    else
        child_rc=$?
        (( shard_rc == 0 )) && shard_rc=$child_rc
    fi
done
set -e

if (( shard_rc != 0 )); then
    printf '{"event":"ancientdoc_mineru_retest_failed","run_id":"%s","stage":"shards","return_code":%s}\n' "$run_id" "$shard_rc" >"${run_root}/failed.json"
    exit 3
fi

model_test_root="${run_root}/mineru2_5_pro/test"
if ! env PYTHONPATH="$code_root" PYTHONUNBUFFERED=1 "$python_bin" \
    "${code_root}/tools/sota/merge_predictions.py" \
    --manifest "$manifest" --shard-root "$model_test_root" \
    --output "${model_test_root}/predictions.jsonl" --expected-pages 516 \
    >"${run_root}/mineru2_5_pro/merge.log" 2>&1; then
    printf '{"event":"ancientdoc_mineru_retest_failed","run_id":"%s","stage":"merge"}\n' "$run_id" >"${run_root}/failed.json"
    exit 3
fi

validate_file="${run_root}/mineru2_5_pro/validate_output.json"
if "$python_bin" - "${model_test_root}/predictions.jsonl" >"$validate_file" <<'PY'
import json
import sys

pages = nonempty = failed = 0
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        record = json.loads(line)
        pages += 1
        nonempty += bool((record.get("normalized_text") or "").strip())
        failed += record.get("status") != "ok"
print(json.dumps({"pages": pages, "nonempty_pages": nonempty, "failed_pages": failed}, separators=(",", ":")))
if pages != 516 or nonempty != 516 or failed != 0:
    raise SystemExit(74)
PY
then
    :
else
    printf '{"event":"ancientdoc_mineru_retest_failed","run_id":"%s","stage":"nonempty_output"}\n' "$run_id" >"${run_root}/failed.json"
    exit 3
fi

if ! env PYTHONPATH="$code_root" PYTHONUNBUFFERED=1 "$python_bin" \
    "${code_root}/tools/sota/summarize_metrics.py" \
    --manifest "$manifest" --predictions "${model_test_root}/predictions.jsonl" \
    --output "${model_test_root}/unified_metrics.json" --expected-pages 516 \
    >"${run_root}/mineru2_5_pro/metrics.log" 2>&1; then
    printf '{"event":"ancientdoc_mineru_retest_failed","run_id":"%s","stage":"metrics"}\n' "$run_id" >"${run_root}/failed.json"
    exit 3
fi

cp "${model_test_root}/shard-000/protocol.json" "${model_test_root}/protocol.json"
cp "${model_test_root}/shard-000/load_status.json" "${model_test_root}/load_status.json"
cp "${model_test_root}/shard-000/run_summary.json" "${model_test_root}/run_summary_shard_000.json"
printf '{"event":"ancientdoc_mineru_retest_inference_finished","run_id":"%s","pages":516,"nonempty_pages":516,"failed_pages":0,"test_used_for_selection":false}\n' \
    "$run_id" >"${run_root}/inference_finished.json"
printf '{"event":"ancientdoc_mineru_retest_completed","run_id":"%s","pages":516,"nonempty_pages":516,"failed_pages":0,"test_used_for_selection":false}\n' \
    "$run_id" >"${run_root}/completed.json"
