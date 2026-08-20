#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_layout_ablation_smoke.sh [DATASET_ID]
  run_layout_ablation_smoke.sh [--dataset-id ID] [--ablations ID[,ID...]] \
    [--gpu-id ID | --gpu-ids ID[,ID...] | --parallel-gpu-ids ID[,ID...]]

--gpu-ids runs every selected ablation on the same DeepSpeed GPU set.
--parallel-gpu-ids maps one physical GPU to each ablation by list order.
USAGE
}

dataset_id="formal_pdf_short_seed20260812"
ablations="got2_zero_shot,projector_only,generic_adapter_projector,vlqa_ocr_only,vlqa_layout_direct,vlqa_layout_p1_p2"
gpu_id=""
gpu_ids=""
parallel_gpu_ids=""
if [[ $# -gt 0 && "$1" != -* ]]; then
    dataset_id="$1"
    shift
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-id) dataset_id="${2:-}"; shift 2 ;;
        --ablations) ablations="${2:-}"; shift 2 ;;
        --gpu-id) gpu_id="${2:-}"; shift 2 ;;
        --gpu-ids) gpu_ids="${2:-}"; shift 2 ;;
        --parallel-gpu-ids) parallel_gpu_ids="${2:-}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'ERROR: unknown option: %s\n' "$1" >&2; usage; exit 64 ;;
    esac
done

gpu_mode_count=0
[[ -n "$gpu_id" ]] && ((gpu_mode_count+=1))
[[ -n "$gpu_ids" ]] && ((gpu_mode_count+=1))
[[ -n "$parallel_gpu_ids" ]] && ((gpu_mode_count+=1))
if (( gpu_mode_count > 1 )); then
    printf 'ERROR: --gpu-id, --gpu-ids, and --parallel-gpu-ids are mutually exclusive.\n' >&2
    exit 64
fi
IFS=',' read -r -a groups <<< "$ablations"
declare -A seen_groups=()
for group in "${groups[@]}"; do
    case "$group" in
        got2_zero_shot|projector_only|generic_adapter_projector|vlqa_ocr_only|vlqa_layout_direct|vlqa_layout_p1_p2) ;;
        *) printf 'ERROR: unsupported ablation: %s\n' "$group" >&2; exit 64 ;;
    esac
    if [[ -n "${seen_groups[$group]:-}" ]]; then
        printf 'ERROR: duplicate ablation: %s\n' "$group" >&2; exit 64
    fi
    seen_groups[$group]=1
done

selected_gpu_ids="${gpu_ids:-${gpu_id:-${GOT_PHYSICAL_GPUS:-${GOT_PHYSICAL_GPU:-0}}}}"
[[ -n "$parallel_gpu_ids" ]] && selected_gpu_ids="$parallel_gpu_ids"
if [[ ! "$selected_gpu_ids" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    printf 'ERROR: GPU ids must be unique comma-separated numeric ids.\n' >&2; exit 64
fi
IFS=',' read -r -a selected_gpu_array <<< "$selected_gpu_ids"
declare -A seen_gpus=()
for selected_gpu in "${selected_gpu_array[@]}"; do
    if [[ -n "${seen_gpus[$selected_gpu]:-}" ]]; then
        printf 'ERROR: duplicate GPU id: %s\n' "$selected_gpu" >&2; exit 64
    fi
    seen_gpus[$selected_gpu]=1
done
if [[ -n "$parallel_gpu_ids" && ${#selected_gpu_array[@]} -ne ${#groups[@]} ]]; then
    printf 'ERROR: --parallel-gpu-ids count must equal --ablations count.\n' >&2; exit 64
fi

require_gpus_below_limit() {
    local target_gpu output utilization
    for target_gpu in "$@"; do
        if ! output="$(nvidia-smi -i "$target_gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>&1)"; then
            printf 'ERROR: cannot query physical GPU %s: %s\n' "$target_gpu" "$output" >&2
            exit 66
        fi
        utilization="${output//[[:space:]]/}"
        if [[ ! "$utilization" =~ ^[0-9]+$ ]]; then
            printf 'ERROR: GPU%s utilization is not numeric: %s\n' "$target_gpu" "$output" >&2
            exit 66
        fi
        if (( utilization >= 50 )); then
            printf 'ERROR: GPU%s_BUSY utilization=%s limit=50\n' "$target_gpu" "$utilization" >&2
            exit 75
        fi
    done
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"
dataset_root="${GOT_LAYOUT_DATA}/${dataset_id}"
bash "${script_dir}/check_layout_dataset_mount.sh" --dataset-root "$dataset_root" >/dev/null
require_gpus_below_limit "${selected_gpu_array[@]}"

stamp="$(date +%Y%m%d_%H%M%S)"
prefix="layout_ablation_smoke_${stamp}"
log_root="${GOT_TRAINING_RUNS}/${prefix}_launcher"
mkdir -p "$log_root"

run_group() {
    local ablation="$1"
    local assigned_gpu_ids="$2"
    local group_log="$3"
    local run_id="${prefix}_${ablation}"
    local args=(
        --mode ablation --ablation "$ablation"
        --dataset-root "$dataset_root/train"
        --manifest "$dataset_root/train/manifest.jsonl"
        --validation-manifest "$dataset_root/validation/manifest.jsonl"
        --validation-image-root "$dataset_root/validation"
        --test-manifest "$dataset_root/test/manifest.jsonl"
        --test-image-root "$dataset_root/test"
        --source-model "$GOT_SOURCE_MODEL"
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-$GOT_SOURCE_MODEL}"
        --run-id "$run_id" --gpu-ids "$assigned_gpu_ids" --skip-source-hash
    )
    if [[ "$ablation" == "got2_zero_shot" ]]; then
        args+=(--validation-max-records 1)
    else
        args+=(--p2-max-steps 1 --p2-max-records 1 --skip-post-training-validation)
        if [[ "$ablation" == "vlqa_layout_p1_p2" ]]; then
            args+=(--p1-max-steps 1 --p1-max-records 1)
        fi
    fi
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$assigned_gpu_ids" \
        bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/training/run_layout_a100.py" "${args[@]}" \
        >"$group_log" 2>&1
}

pids=()
logs=()
if [[ -n "$parallel_gpu_ids" ]]; then
    for index in "${!groups[@]}"; do
        group="${groups[$index]}"
        group_log="${log_root}/${group}.log"
        run_group "$group" "${selected_gpu_array[$index]}" "$group_log" &
        pids+=("$!")
        logs+=("$group_log")
    done
    failed=0
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            failed=1
            printf 'ERROR: smoke ablation %s failed; last 20 log lines follow.\n' "${groups[$index]}" >&2
            tail -n 20 "${logs[$index]}" >&2
        fi
    done
    if (( failed != 0 )); then
        printf '{"event":"layout_ablation_parallel_smoke_failed","run_prefix":"%s","log_root":"%s"}\n' \
            "$prefix" "$log_root" >&2
        exit 1
    fi
else
    for group in "${groups[@]}"; do
        group_log="${log_root}/${group}.log"
        if ! run_group "$group" "$selected_gpu_ids" "$group_log"; then
            tail -n 20 "$group_log" >&2
            printf '{"event":"layout_ablation_smoke_failed","ablation":"%s","log":"%s"}\n' \
                "$group" "$group_log" >&2
            exit 1
        fi
        logs+=("$group_log")
    done
fi

python3 - "$GOT_TRAINING_RUNS" "$prefix" "$log_root" "$dataset_id" "$selected_gpu_ids" "${groups[@]}" <<'PY'
from __future__ import annotations
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
prefix = sys.argv[2]
log_root = sys.argv[3]
dataset_id = sys.argv[4]
physical_gpu_ids = sys.argv[5]
groups = sys.argv[6:]
results = []
for group in groups:
    summary_path = runs_root / f"{prefix}_{group}" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    item = {
        "ablation": group,
        "status": summary.get("status"),
        "physical_gpus": (summary.get("settings") or {}).get("physical_gpu_ids"),
        "summary": str(summary_path),
    }
    if group == "got2_zero_shot":
        validation = summary.get("validation") or {}
        item["validation_pages"] = validation.get("pages")
        item["model_kind"] = validation.get("model_kind")
    else:
        for stage in ("p1", "p2"):
            payload = summary.get(stage)
            if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
                metrics = payload["metrics"]
                item[stage] = {
                    "step": metrics.get("global_step"),
                    "world_size": (metrics.get("training_budget") or {}).get("world_size"),
                    "trainable_parameters": metrics.get("trainable_parameters"),
                    "module_parameters": metrics.get("module_parameters"),
                    "loss_weights": metrics.get("loss_weights"),
                    "checkpoint_verified": isinstance(payload.get("verification"), dict),
                }
    results.append(item)
print(json.dumps({
    "event": "layout_ablation_smoke_completed",
    "dataset_id": dataset_id,
    "run_prefix": prefix,
    "physical_gpu_ids": physical_gpu_ids,
    "log_root": log_root,
    "groups": results,
}, ensure_ascii=False, separators=(",", ":")))
PY
