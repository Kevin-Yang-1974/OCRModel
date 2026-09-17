#!/usr/bin/env bash
# Run the two supplied synthetic-data GLM-OCR adapter variants on AncientDoc.
# Each formal arm uses all five requested A100 GPUs; the arms run serially.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
code_root="$(cd -- "${script_dir}/../.." && pwd -P)"

run_id="ancientdoc_glmocr_synthetic_split5_test_20260913_q32_v1"
run_root_base="/data3/yky/yangky_ocr_models/evaluation_runs/GLMOCR_ANCIENTDOC"
checkpoint_root="/data3/yky/yangky_ocr_models/evaluation_runs/GLMOCR_ANCIENTDOC/weights/synthetic_q32"
model_dir="/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d"
label_json="/data4/hyf/backup/GOT-OCR2.0/reference-260707/AncientDoc/label_for_got_split5.json"
image_root="/data4/hyf/project/古籍/AncientDoc"
gpu_ids="0,1,2,3,4"
num_queries="32"
decoder_lora_rank="16"
decoder_lora_alpha="16"
decoder_lora_dropout="0"
max_pixels="1003520"
max_output_tokens="1536"
python_bin="${A100_GLMOCR_PYTHON:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/envs/glmocr_a100_py311_cu128/bin/python}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
processor_mode="${A100_GLMOCR_PROCESSOR_MODE:-fast}"
foreground=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) run_id="$2"; shift 2 ;;
        --run-root-base) run_root_base="$2"; shift 2 ;;
        --checkpoint-root) checkpoint_root="$2"; shift 2 ;;
        --model-dir) model_dir="$2"; shift 2 ;;
        --label-json) label_json="$2"; shift 2 ;;
        --image-root) image_root="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --max-output-tokens) max_output_tokens="$2"; shift 2 ;;
        --foreground) foreground=1; shift ;;
        *) printf '{"event":"ancientdoc_glmocr_synthetic_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

[[ "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
[[ "$gpu_ids" == "0,1,2,3,4" ]] || { echo "this run requires GPU 0,1,2,3,4" >&2; exit 64; }
[[ "$num_queries" == "32" && "$decoder_lora_rank" == "16" ]] || exit 64
[[ "$max_output_tokens" =~ ^[1-9][0-9]*$ ]] || exit 64

# The standalone torch wheel omits the CUDA component wheels.  Reuse the
# maintained A100 runtime's component libraries, matching the project
# training launchers, before any Python process imports torch.
cuda_library_path="$(dirname "$(dirname "$python_bin")")/lib/python3.11/site-packages/torch/lib"
system_cuda="/usr/local/cuda/targets/$(uname -m)-linux/lib"
[[ -d "$system_cuda" ]] && cuda_library_path="${system_cuda}:${cuda_library_path}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    component_lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
    [[ -d "$component_lib" ]] && cuda_library_path="${cuda_library_path}:${component_lib}"
done
export LD_LIBRARY_PATH="${cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

run_root="${run_root_base}/${run_id}"
manifest="${run_root}/dataset/manifest.jsonl"
session="${A100_GLMOCR_TMUX_SESSION:-${run_id}}"

if (( foreground == 0 )); then
    command -v tmux >/dev/null 2>&1 || exit 69
    tmux has-session -t "$session" 2>/dev/null && {
        printf '{"event":"ancientdoc_glmocr_synthetic_refused","error":"tmux_session_exists","session":"%s"}\n' "$session" >&2
        exit 73
    }
    startup_log="${run_root_base}/${run_id}.launcher.log"
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    tmux new-session -d -s "$session" \
        "cd $(printf '%q' "$code_root") && exec bash $(printf '%q' "$script_path") --foreground --run-id $(printf '%q' "$run_id") --run-root-base $(printf '%q' "$run_root_base") --checkpoint-root $(printf '%q' "$checkpoint_root") --model-dir $(printf '%q' "$model_dir") --label-json $(printf '%q' "$label_json") --image-root $(printf '%q' "$image_root") --gpu-ids $(printf '%q' "$gpu_ids") --max-output-tokens $(printf '%q' "$max_output_tokens") >$(printf '%q' "$startup_log") 2>&1"
    sleep 5
    if tmux has-session -t "$session" 2>/dev/null; then
        printf '{"event":"ancientdoc_glmocr_synthetic_started","session":"%s","run_id":"%s","run_root":"%s","gpu_ids":"%s","num_queries":32}\n' "$session" "$run_id" "$run_root" "$gpu_ids"
        exit 0
    fi
    printf '{"event":"ancientdoc_glmocr_synthetic_failed_to_start","session":"%s","run_id":"%s","startup_log":"%s"}\n' "$session" "$run_id" "$startup_log" >&2
    exit 1
fi

[[ ! -e "$run_root" ]] || { printf '{"event":"ancientdoc_glmocr_synthetic_refused","error":"run_root_exists","run_root":"%s"}\n' "$run_root" >&2; exit 73; }
[[ -x "$python_bin" && -f "${code_root}/tools/sota/run_ancientdoc_glmocr_adapter_eval.py" ]] || exit 66
[[ -f "$label_json" && -d "$image_root" && -d "$model_dir" ]] || exit 66
for mode in content_only geometry; do
    [[ -d "${checkpoint_root}/${mode}" ]] || { echo "missing checkpoint: ${checkpoint_root}/${mode}" >&2; exit 66; }
    for name in adapter.safetensors decoder_lora.safetensors adapter_config.json; do
        [[ -f "${checkpoint_root}/${mode}/${name}" ]] || exit 66
    done
done

# Admission checks query only the five explicitly requested physical GPUs.
gpu_rows="$(nvidia-smi -i "$gpu_ids" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)"
declare -A observed=()
while IFS=',' read -r observed_id observed_util; do
    observed_id="${observed_id//[[:space:]]/}"
    observed_util="${observed_util//[[:space:]]/}"
    [[ "$observed_id" =~ ^[0-9]+$ && "$observed_util" =~ ^[0-9]+$ ]] || exit 79
    observed["$observed_id"]="$observed_util"
done <<< "$gpu_rows"
for gpu in 0 1 2 3 4; do
    [[ -n "${observed[$gpu]+present}" ]] || exit 79
    (( observed[$gpu] < 50 )) || {
        printf '{"event":"ancientdoc_glmocr_synthetic_refused","error":"gpu_admission_failed","gpu":"%s","utilization":%s,"limit":50}\n' "$gpu" "${observed[$gpu]}" >&2
        exit 79
    }
done

mkdir -p "${run_root}/dataset" "${run_root}/content_only" "${run_root}/geometry"
env PYTHONPATH="${code_root}" PYTHONUNBUFFERED=1 "$python_bin" -m tools.sota.build_ancientdoc_manifest \
    --label-json "$label_json" --image-root "$image_root" --output "$manifest" \
    >"${run_root}/dataset/build_manifest.json" 2>&1

printf '{"event":"ancientdoc_glmocr_synthetic_armed","run_id":"%s","dataset":"AncientDoc","source_split":"split5","split":"test","manifest":"%s","gpu_ids":"%s","num_queries":32,"decoder_lora_rank":16,"decoder_lora_alpha":16,"max_output_tokens":%s,"test_used_for_selection":false}\n' \
    "$run_id" "$manifest" "$gpu_ids" "$max_output_tokens" >"${run_root}/launch.json"

run_one_shard() {
    local mode="$1" phase="$2" index="$3" limit="${4:-}"
    local gpu="${index}"
    local output_dir="${run_root}/${mode}/${phase}/shard-$(printf '%03d' "$index")"
    local log_path="${run_root}/${mode}/${phase}/shard-$(printf '%03d' "$index").log"
    mkdir -p "${run_root}/${mode}/${phase}"
    local command=(
        env "PYTHONPATH=${code_root}/src:${code_root}" PYTHONUNBUFFERED=1
        CUDA_VISIBLE_DEVICES="$gpu" "$python_bin"
        "${code_root}/tools/sota/run_ancientdoc_glmocr_adapter_eval.py"
        --model-path "$model_dir" --checkpoint-dir "${checkpoint_root}/${mode}"
        --mode "$mode" --num-queries "$num_queries"
        --decoder-lora-rank "$decoder_lora_rank" --decoder-lora-alpha "$decoder_lora_alpha"
        --decoder-lora-dropout "$decoder_lora_dropout" --manifest "$manifest"
        --image-root "$image_root" --source-label "$label_json" --output-dir "$output_dir"
        --device cuda:0 --max-pixels "$max_pixels" --processor-mode "$processor_mode"
        --max-eval-new-tokens "$max_output_tokens" --shard-index "$index" --shard-count 5
        --progress-every 5 --allow-test
    )
    if [[ -n "$limit" ]]; then
        command+=(--limit "$limit")
    fi
    "${command[@]}" >"$log_path" 2>&1
}

run_sharded() {
    local mode="$1" phase="$2" limit="$3"
    local pids=() index rc=0 child_rc
    for index in 0 1 2 3 4; do
        run_one_shard "$mode" "$phase" "$index" "$([[ "$limit" == "none" ]] && echo || echo "$limit")" &
        pids+=("$!")
    done
    set +e
    for pid in "${pids[@]}"; do
        wait "$pid"
        child_rc=$?
        (( child_rc == 0 )) || rc=$child_rc
    done
    set -e
    return "$rc"
}

validate_smoke() {
    local mode="$1"
    "$python_bin" - "${run_root}/${mode}/smoke" "$mode" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
mode = sys.argv[2]
for index in range(5):
    shard = root / f"shard-{index:03d}"
    summary = json.loads((shard / "run_summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (shard / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if summary.get("status") != "loaded" or summary.get("pages") != 1 or summary.get("ok") != 1 or summary.get("failed") != 0:
        raise SystemExit(f"smoke_summary_failed {mode} shard={index}")
    if len(rows) != 1 or rows[0].get("status") != "ok" or not rows[0].get("normalized_text"):
        raise SystemExit(f"smoke_nonempty_failed {mode} shard={index}")
print(json.dumps({"event": "ancientdoc_glmocr_synthetic_smoke_ok", "mode": mode, "shards": 5, "nonempty_pages": 5}, ensure_ascii=False))
PY
}

aggregate_run_summary() {
    local mode="$1"
    "$python_bin" - "${run_root}/${mode}/test" "$mode" <<'PY'
import json
import sys
from pathlib import Path
src = Path(sys.argv[1])
mode = sys.argv[2]
summaries = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(src.glob("shard-*/run_summary.json"))]
out = {
    "event": "ancientdoc_glmocr_synthetic_test_complete",
    "status": "complete",
    "model": f"GLM-OCR synthetic {mode}",
    "mode": mode,
    "pages": sum(int(item.get("pages", 0)) for item in summaries),
    "ok": sum(int(item.get("ok", 0)) for item in summaries),
    "failed": sum(int(item.get("failed", 0)) for item in summaries),
    "shard_count": len(summaries),
    "num_queries": 32,
    "decoder_lora_rank": 16,
    "decoder_lora_alpha": 16,
    "test_used_for_selection": False,
}
(src / "run_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
PY
}

run_formal() {
    local mode="$1"
    run_sharded "$mode" test none
    env PYTHONPATH="${code_root}" PYTHONUNBUFFERED=1 "$python_bin" \
        "${code_root}/tools/sota/merge_predictions.py" \
        --manifest "$manifest" --shard-root "${run_root}/${mode}/test" \
        --output "${run_root}/${mode}/test/predictions.jsonl" --expected-pages 516 \
        >"${run_root}/${mode}/merge.log" 2>&1
    env PYTHONPATH="${code_root}" PYTHONUNBUFFERED=1 "$python_bin" \
        "${code_root}/tools/sota/summarize_metrics.py" \
        --manifest "$manifest" --predictions "${run_root}/${mode}/test/predictions.jsonl" \
        --output "${run_root}/${mode}/test/unified_metrics.json" --expected-pages 516 \
        >"${run_root}/${mode}/metrics.log" 2>&1
    cp "${run_root}/${mode}/test/shard-000/protocol.json" "${run_root}/${mode}/test/protocol.json"
    cp "${run_root}/${mode}/test/shard-000/load_status.json" "${run_root}/${mode}/test/load_status.json"
    aggregate_run_summary "$mode" >"${run_root}/${mode}/run_summary.json"
    env PYTHONPATH="${code_root}" PYTHONUNBUFFERED=1 "$python_bin" - "${run_root}/${mode}/test" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
metrics = json.loads((root / "unified_metrics.json").read_text(encoding="utf-8"))
load = json.loads((root / "load_status.json").read_text(encoding="utf-8"))
if metrics.get("pages") != 516 or metrics.get("failed_pages") != 0 or load.get("status") != "loaded":
    raise SystemExit("formal_result_contract_failed")
print(json.dumps({"event": "ancientdoc_glmocr_synthetic_formal_ok", "pages": metrics["pages"], "failed_pages": metrics["failed_pages"]}, ensure_ascii=False))
PY
}

set +e
run_sharded content_only smoke 5
content_smoke_rc=$?
set -e
if (( content_smoke_rc != 0 )); then
    printf '{"event":"ancientdoc_glmocr_synthetic_failed","phase":"content_only_smoke","return_code":%s}\n' "$content_smoke_rc" >"${run_root}/failed.json"
    exit "$content_smoke_rc"
fi
validate_smoke content_only >"${run_root}/content_only/smoke_summary.json"

set +e
run_sharded geometry smoke 5
geometry_smoke_rc=$?
set -e
if (( geometry_smoke_rc != 0 )); then
    printf '{"event":"ancientdoc_glmocr_synthetic_failed","phase":"geometry_smoke","return_code":%s}\n' "$geometry_smoke_rc" >"${run_root}/failed.json"
    exit "$geometry_smoke_rc"
fi
validate_smoke geometry >"${run_root}/geometry/smoke_summary.json"
printf '{"event":"ancientdoc_glmocr_synthetic_smoke_gate_complete","content_only":true,"geometry":true,"num_queries":32,"test_used_for_selection":false}\n' >"${run_root}/smoke_completed.json"

run_formal content_only
run_formal geometry

env PYTHONPATH="${code_root}" PYTHONUNBUFFERED=1 "$python_bin" \
    "${code_root}/tools/sota/format_ancientdoc_glmocr_results.py" \
    --run-root "$run_root" --run-id "$run_id" --manifest "$manifest" \
    --source-label "$label_json" --image-root "$image_root" --output "${run_root}/RESULTS.md" \
    --gpu-ids "$gpu_ids" --num-queries "$num_queries" \
    --decoder-lora-rank "$decoder_lora_rank" --decoder-lora-alpha "$decoder_lora_alpha" \
    --max-output-tokens "$max_output_tokens" >"${run_root}/format.log" 2>&1

printf '{"event":"ancientdoc_glmocr_synthetic_complete","run_id":"%s","run_root":"%s","results":"%s/RESULTS.md","num_queries":32,"decoder_lora_rank":16,"decoder_lora_alpha":16,"test_used_for_selection":false}\n' \
    "$run_id" "$run_root" "$run_root" | tee "${run_root}/completed.json"
