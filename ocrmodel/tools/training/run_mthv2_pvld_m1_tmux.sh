#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

session="mthv2_pvld_m1_20260825"
run_id="mthv2_pvld_m1_boundary_count_20260825_v1"
gpu_ids="1,2,3,4"
dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1"
utilization_limit=50
distributed_strategy="ddp"
nccl_p2p_disable=1
session_inner=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="$2"; shift 2 ;;
        --run-id) run_id="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --gpu-utilization-limit) utilization_limit="$2"; shift 2 ;;
        --distributed-strategy) distributed_strategy="$2"; shift 2 ;;
        --nccl-p2p-disable) nccl_p2p_disable=1; shift ;;
        --session-inner) session_inner=1; shift ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done

IFS=',' read -r -a gpus <<< "${gpu_ids}"
[[ ${#gpus[@]} -ge 1 ]] || { printf 'ERROR: at least one target GPU is required.\n' >&2; exit 64; }
for gpu in "${gpus[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { printf 'ERROR: invalid GPU ID: %s.\n' "${gpu}" >&2; exit 64; }
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
    [[ "${utilization}" =~ ^[0-9]+$ ]] || { printf 'ERROR: GPU%s utilization query failed.\n' "${gpu}" >&2; exit 75; }
    (( utilization < utilization_limit )) || { printf 'ERROR: GPU%s utilization=%s.\n' "${gpu}" "${utilization}" >&2; exit 75; }
done

run_root="${GOT_TRAINING_RUNS}/${run_id}"
selection_root="${GOT_EVALUATION_RUNS}/${run_id}_p2_validation_selection"
launcher_root="${GOT_TRAINING_RUNS}/${run_id}_tmux"
launcher_log="${launcher_root}/launcher.log"
mkdir -p "${launcher_root}"

run_pipeline() {
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/training/run_variable_layout_a100.py" \
        --dataset-root "${dataset_root}/train" \
        --manifest "${dataset_root}/train/manifest.jsonl" \
        --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
        --test-manifest "${dataset_root}/test/manifest.jsonl" \
        --source-model "${GOT_SOURCE_MODEL}" \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --stages p1,p2 --ablation vlqa_layout_p1_p2 \
        --layout-loss-preset layout_full \
        --num-layout-prompt-queries 32 --max-layout-records 512 \
        --max-layout-tokens 2048 --layout-decoder-layers 2 \
        --layout-decoder-hidden-size 256 --layout-decoder-num-heads 8 \
        --p1-max-steps 3000 --p2-max-steps 7500 \
        --p1-checkpoint-steps 1000 --p2-checkpoint-steps 2500 \
        --checkpoint-retention 3 \
        --layout-boundary-loss-weight 1.0 \
        --layout-count-condition-strength 1.0 \
        --per-device-batch-size 1 --gradient-accumulation-steps 1 \
        --p1-learning-rate 1e-4 --p2-learning-rate 5e-5 \
        --gpu-ids "${gpu_ids}" --gpu-utilization-limit "${utilization_limit}" \
        --distributed-strategy "${distributed_strategy}" \
        --nccl-p2p-disable \
        --seed 42 --run-id "${run_id}"

    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/evaluation/select_layout_ablation_checkpoint.py" \
        --ablation vlqa_layout_p1_p2 \
        --model-root "${run_root}/p2/model" --model-kind pvld \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
        --validation-image-root "${dataset_root}/validation" \
        --output-dir "${selection_root}" \
        --project-root "${ocrmodel_root}/src/GOT-OCR-2.0" \
        --max-regions 512 --max-records 0 --max-new-tokens 2048 \
        --no-repeat-ngram-size 20 \
        --parallel-gpu-ids "${gpu_ids}" \
        --gpu-utilization-limit "${utilization_limit}"

    python - "${run_root}" "${selection_root}" <<'PY'
import json
import sys
from pathlib import Path

run_root, selection_root = map(Path, sys.argv[1:])
checkpoint_count = sum(1 for path in run_root.glob("p*/model/checkpoint-*") if path.is_dir())
selection = json.loads((selection_root / "selection.json").read_text(encoding="utf-8"))
if checkpoint_count > 6:
    raise RuntimeError(f"checkpoint cap exceeded: {checkpoint_count}")
print(json.dumps({
    "event": "mthv2_pvld_m1_validation_selected",
    "run_root": str(run_root),
    "selection": str(selection_root / "selection.json"),
    "selected_step": selection["selected"]["optimizer_step"],
    "checkpoint_directories": checkpoint_count,
    "test_run": False,
}, ensure_ascii=False, separators=(",", ":")))
PY
}

if (( session_inner == 1 )); then
    run_pipeline
    exit
fi

[[ ! -e "${run_root}" ]] || { printf 'ERROR: run already exists: %s\n' "${run_root}" >&2; exit 73; }
[[ ! -e "${selection_root}" ]] || { printf 'ERROR: selection already exists: %s\n' "${selection_root}" >&2; exit 73; }
tmux has-session -t "${session}" 2>/dev/null && { printf 'ERROR: tmux session exists.\n' >&2; exit 73; }
script_path="$(realpath "${BASH_SOURCE[0]}")"
tmux new-session -d -s "${session}" \
    "cd '${ocrmodel_root}' && exec bash '${script_path}' --session-inner --session '${session}' --run-id '${run_id}' --gpu-ids '${gpu_ids}' --gpu-utilization-limit '${utilization_limit}' --distributed-strategy '${distributed_strategy}' --nccl-p2p-disable >'${launcher_log}' 2>&1"
sleep 5
tmux has-session -t "${session}" 2>/dev/null || { tail -n 20 "${launcher_log}" >&2 || true; exit 1; }
printf '{"event":"mthv2_pvld_m1_started","session":"%s","run_id":"%s","gpu_ids":"%s","distributed_strategy":"%s","p1_steps":3000,"p2_steps":7500,"effective_batch":%d,"page_exposure":{"p1":12000,"p2":30000},"checkpoint_cap":6,"test_run":false,"log":"%s"}\n' \
    "${session}" "${run_id}" "${gpu_ids}" "${distributed_strategy}" "${#gpus[@]}" "${launcher_log}"
