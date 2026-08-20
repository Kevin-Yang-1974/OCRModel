#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_layout_ablation_suite.sh --dataset-id ID --ablations ID[,ID...] \
    --p2-steps N --checkpoint-steps N [--p1-steps N] [--seed N] \
    [--gpu-id ID | --gpu-ids ID[,ID...] | --parallel-gpu-ids ID[,ID...]] \
    [--layout-loss-preset PRESET] [--ocr-loss-weight FLOAT] \
    [--run-prefix NAME] [--test-set Category:dataset-id] [--selection-only] [--resume]

--gpu-ids gives every ablation one distributed DeepSpeed run on all listed GPUs.
--parallel-gpu-ids maps one physical GPU to each ablation by list order.
Categories: Synthetic-ID, Synthetic-OOD, Real-OOD. If --test-set is omitted,
the training dataset test split is evaluated as Synthetic-ID.
USAGE
}

dataset_id=""
ablations=""
p1_steps=""
p2_steps=""
checkpoint_steps=""
gpu_utilization_limit="50"
seed="42"
run_prefix="layout_ablation"
resume="0"
selection_only="0"
layout_loss_preset=""
ocr_loss_weight="1"
gpu_id=""
gpu_ids=""
parallel_gpu_ids=""
test_sets=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-id) dataset_id="${2:-}"; shift 2 ;;
        --ablations) ablations="${2:-}"; shift 2 ;;
        --p1-steps) p1_steps="${2:-}"; shift 2 ;;
        --p2-steps) p2_steps="${2:-}"; shift 2 ;;
        --checkpoint-steps) checkpoint_steps="${2:-}"; shift 2 ;;
        --gpu-utilization-limit) gpu_utilization_limit="${2:-}"; shift 2 ;;
        --seed) seed="${2:-}"; shift 2 ;;
        --run-prefix) run_prefix="${2:-}"; shift 2 ;;
        --layout-loss-preset) layout_loss_preset="${2:-}"; shift 2 ;;
        --ocr-loss-weight) ocr_loss_weight="${2:-}"; shift 2 ;;
        --gpu-id) gpu_id="${2:-}"; shift 2 ;;
        --gpu-ids) gpu_ids="${2:-}"; shift 2 ;;
        --parallel-gpu-ids) parallel_gpu_ids="${2:-}"; shift 2 ;;
        --test-set) test_sets+=("${2:-}"); shift 2 ;;
        --selection-only) selection_only="1"; shift ;;
        --resume) resume="1"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'ERROR: unknown option: %s\n' "$1" >&2; usage; exit 64 ;;
    esac
done
if [[ -z "$dataset_id" || -z "$ablations" || -z "$p2_steps" || -z "$checkpoint_steps" ]]; then
    usage; exit 64
fi
if [[ ! "$gpu_utilization_limit" =~ ^[1-9][0-9]*$ ]] || (( gpu_utilization_limit > 100 )); then
    printf 'ERROR: --gpu-utilization-limit must be an integer in 1..100.\n' >&2
    exit 64
fi
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
for ablation in "${groups[@]}"; do
    case "$ablation" in
        got2_zero_shot|projector_only|generic_adapter_projector|vlqa_ocr_only|vlqa_layout_direct|vlqa_layout_p1_p2) ;;
        *) printf 'ERROR: unsupported ablation: %s\n' "$ablation" >&2; exit 64 ;;
    esac
    if [[ -n "${seen_groups[$ablation]:-}" ]]; then
        printf 'ERROR: duplicate ablation: %s\n' "$ablation" >&2; exit 64
    fi
    seen_groups[$ablation]=1
done

selected_gpu_ids="${gpu_ids:-${gpu_id:-${GOT_PHYSICAL_GPUS:-${GOT_PHYSICAL_GPU:-0}}}}"
if [[ -n "$parallel_gpu_ids" ]]; then
    selected_gpu_ids="$parallel_gpu_ids"
fi
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
        if (( utilization >= gpu_utilization_limit )); then
            printf 'ERROR: GPU%s_BUSY utilization=%s limit=%s\n' "$target_gpu" "$utilization" "$gpu_utilization_limit" >&2
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
audit_root="${GOT_EVALUATION_RUNS}/${run_prefix}_dataset_audits"
mkdir -p "$audit_root"
bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
    "${ocrmodel_root}/tools/preprocessing/audit_synthetic_layout.py" \
    --manifest "$dataset_root/train/manifest.jsonl" \
    --manifest "$dataset_root/validation/manifest.jsonl" \
    --manifest "$dataset_root/test/manifest.jsonl" \
    --summary-json "$audit_root/${dataset_id}.json" >/dev/null
if [[ ${#test_sets[@]} -eq 0 ]]; then
    test_sets+=("Synthetic-ID:${dataset_id}")
fi

run_ablation() (
    set -euo pipefail
    local ablation="$1"
    local assigned_gpu_ids="$2"
    local model_kind run_id run_root model_root selection_root
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$assigned_gpu_ids"
    case "$ablation" in
        got2_zero_shot|projector_only) model_kind="baseline" ;;
        generic_adapter_projector) model_kind="generic" ;;
        vlqa_ocr_only|vlqa_layout_direct|vlqa_layout_p1_p2) model_kind="vlqa" ;;
    esac
    run_id="${run_prefix}_${ablation}_seed${seed}"
    run_root="${GOT_TRAINING_RUNS}/${run_id}"
    if [[ "$ablation" == "got2_zero_shot" ]]; then
        model_root="${GOT_SOURCE_MODEL}"
    else
        if [[ -e "$run_root" ]]; then
            if [[ "$resume" != "1" || ! -f "$run_root/LAYOUT_A100_FINISHED" ]]; then
                printf 'ERROR: run exists; use --resume only for a completed run: %s\n' "$run_root" >&2
                exit 74
            fi
        else
            train_args=(
                --mode ablation --ablation "$ablation"
                --dataset-root "$dataset_root/train"
                --manifest "$dataset_root/train/manifest.jsonl"
                --validation-manifest "$dataset_root/validation/manifest.jsonl"
                --validation-image-root "$dataset_root/validation"
                --test-manifest "$dataset_root/test/manifest.jsonl"
                --test-image-root "$dataset_root/test"
                --source-model "$GOT_SOURCE_MODEL"
                --tokenizer-model "${GOT_TOKENIZER_MODEL:-$GOT_SOURCE_MODEL}"
                --p2-max-steps "$p2_steps" --checkpoint-steps "$checkpoint_steps"
                --gpu-utilization-limit "$gpu_utilization_limit"
                --seed "$seed" --run-id "$run_id" --gpu-ids "$assigned_gpu_ids"
                --p2-ocr-loss-weight "$ocr_loss_weight"
                --skip-post-training-validation
            )
            [[ -n "$layout_loss_preset" ]] && train_args+=(--layout-loss-preset "$layout_loss_preset")
            if [[ "$ablation" == "vlqa_layout_p1_p2" ]]; then
                if [[ -z "$p1_steps" ]]; then
                    printf 'ERROR: A5 requires --p1-steps.\n' >&2; exit 64
                fi
                train_args+=(--p1-max-steps "$p1_steps")
            fi
            bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
                "${ocrmodel_root}/tools/training/run_layout_a100.py" "${train_args[@]}"
        fi
        model_root="$run_root/p2/model"
    fi

    selection_root="${GOT_EVALUATION_RUNS}/${run_id}_selection"
    inference_gpu_id="${assigned_gpu_ids%%,*}"
    select_args=(
        --ablation "$ablation" --model-root "$model_root" --model-kind "$model_kind"
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-$GOT_SOURCE_MODEL}"
        --validation-manifest "$dataset_root/validation/manifest.jsonl"
        --validation-image-root "$dataset_root/validation"
        --project-root "$ocrmodel_root/src/GOT-OCR-2.0"
        --output-dir "$selection_root"
        --gpu-id "$inference_gpu_id"
        --gpu-utilization-limit "$gpu_utilization_limit"
    )
    [[ "$resume" == "1" ]] && select_args+=(--resume)
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/evaluation/select_layout_ablation_checkpoint.py" "${select_args[@]}"

    if [[ "$selection_only" == "1" ]]; then
        printf '{"event":"layout_ablation_group_selected","ablation":"%s","selection":"%s","physical_gpu_ids":"%s"}\n' \
            "$ablation" "$selection_root/selection.json" "$assigned_gpu_ids"
        exit 0
    fi

    local test_set category test_dataset_id test_root test_dataset_root output_root
    for test_set in "${test_sets[@]}"; do
        category="${test_set%%:*}"
        test_dataset_id="${test_set#*:}"
        if [[ "$category" == "$test_dataset_id" ]]; then
            printf 'ERROR: --test-set must be Category:dataset-id: %s\n' "$test_set" >&2; exit 64
        fi
        test_root="${GOT_LAYOUT_DATA}/${test_dataset_id}/test"
        test_dataset_root="${GOT_LAYOUT_DATA}/${test_dataset_id}"
        bash "${script_dir}/check_layout_dataset_mount.sh" --dataset-root "$test_dataset_root" >/dev/null
        bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
            "${ocrmodel_root}/tools/preprocessing/audit_synthetic_layout.py" \
            --manifest "$dataset_root/train/manifest.jsonl" \
            --manifest "$dataset_root/validation/manifest.jsonl" \
            --manifest "$test_root/manifest.jsonl" \
            --summary-json "$audit_root/${dataset_id}__${test_dataset_id}__${ablation}.json" >/dev/null
        output_root="${GOT_EVALUATION_RUNS}/${run_id}_test_${category//-/_}_${test_dataset_id}"
        test_args=(
            --selection "$selection_root/selection.json" --test-category "$category"
            --test-manifest "$test_root/manifest.jsonl" --test-image-root "$test_root"
            --model-kind "$model_kind" --tokenizer-model "${GOT_TOKENIZER_MODEL:-$GOT_SOURCE_MODEL}"
            --project-root "$ocrmodel_root/src/GOT-OCR-2.0" --output-dir "$output_root"
            --gpu-id "$inference_gpu_id"
        )
        [[ "$resume" == "1" ]] && test_args+=(--resume)
        bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
            "${ocrmodel_root}/tools/evaluation/evaluate_layout_ablation_test.py" "${test_args[@]}"
    done
    printf '{"event":"layout_ablation_group_completed","ablation":"%s","physical_gpu_ids":"%s"}\n' \
        "$ablation" "$assigned_gpu_ids"
)

if [[ -n "$parallel_gpu_ids" ]]; then
    parallel_log_root="${GOT_TRAINING_RUNS}/${run_prefix}_parallel_launcher"
    if [[ -e "$parallel_log_root" && "$resume" != "1" ]]; then
        printf 'ERROR: parallel launcher output exists: %s\n' "$parallel_log_root" >&2; exit 74
    fi
    mkdir -p "$parallel_log_root"
    pids=()
    logs=()
    for index in "${!groups[@]}"; do
        group="${groups[$index]}"
        group_gpu="${selected_gpu_array[$index]}"
        group_log="${parallel_log_root}/${group}.log"
        run_ablation "$group" "$group_gpu" >"$group_log" 2>&1 &
        pids+=("$!")
        logs+=("$group_log")
    done
    failed=0
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            failed=1
            printf 'ERROR: ablation %s failed; last 20 log lines follow.\n' "${groups[$index]}" >&2
            tail -n 20 "${logs[$index]}" >&2
        fi
    done
    if (( failed != 0 )); then
        printf '{"event":"layout_ablation_parallel_failed","run_prefix":"%s","log_root":"%s"}\n' \
            "$run_prefix" "$parallel_log_root" >&2
        exit 1
    fi
    printf '{"event":"layout_ablation_parallel_completed","run_prefix":"%s","ablations":"%s","parallel_gpu_ids":"%s","log_root":"%s"}\n' \
        "$run_prefix" "$ablations" "$parallel_gpu_ids" "$parallel_log_root"
else
    for group in "${groups[@]}"; do
        run_ablation "$group" "$selected_gpu_ids"
    done
    printf '{"event":"layout_ablation_suite_completed","run_prefix":"%s","ablations":"%s","physical_gpu_ids":"%s"}\n' \
        "$run_prefix" "$ablations" "$selected_gpu_ids"
fi
