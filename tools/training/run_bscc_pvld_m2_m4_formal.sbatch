#!/usr/bin/env bash
#SBATCH --job-name=pvld_formal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -euo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
env_dir="${BSCC_GOT_ENV:-${workspace}/envs/got2-py310-cu118}"
model_dir="${BSCC_GOT_MODEL:-${workspace}/models/GOT-OCR2-original}"
dataset_root="${BSCC_MTHV2_PAGE_ROOT:-${workspace}/datasets/MTHv2/converted/mthv2_layout_page_v1}"
control="${PVLD_CONTROL:?set PVLD_CONTROL=M2|M3|M4}"
run_id="${PVLD_RUN_ID:?set a new PVLD_RUN_ID}"
run_root="${workspace}/training_runs/GOT/${run_id}"
selection_root="${workspace}/evaluation_runs/GOT/${run_id}_p2_validation_selection"
test_root="${workspace}/evaluation_runs/GOT/${run_id}_selection_locked_test"
control_log="${workspace}/runs/${run_id}.pipeline.log"
snapshot_root="${workspace}/run_code/${run_id}/ocrmodel"

case "${control}" in
    M2) shared_scale=1.0; predicted_routing=0 ;;
    M3) shared_scale=0.25; predicted_routing=0 ;;
    M4) shared_scale=0.25; predicted_routing=1 ;;
    *) printf 'ERROR: unsupported PVLD_CONTROL=%s.\n' "${control}" >&2; exit 64 ;;
esac
control_args=(
    --pvld-use-spatial-memory
    --pvld-shared-gradient-scale "${shared_scale}"
    --pvld-record-gradient-scale "${shared_scale}"
)
if [[ "${predicted_routing}" == 1 ]]; then
    control_args+=(--pvld-predicted-layout-routing)
fi

[[ -f "${dataset_root}/train/manifest.jsonl" \
   && -f "${dataset_root}/validation/manifest.jsonl" \
   && -f "${dataset_root}/test/manifest.jsonl" ]] || {
    printf 'ERROR: prepared MTHv2 manifests are missing.\n' >&2
    exit 66
}
[[ -f "${workspace}/datasets/MTHv2/metadata/MTHV2_PREPARED" ]] || {
    printf 'ERROR: MTHv2 preparation marker is missing.\n' >&2
    exit 66
}
[[ ! -e "${run_root}" && ! -e "${selection_root}" && ! -e "${test_root}" \
   && ! -e "${snapshot_root}" ]] || {
    printf 'ERROR: formal run, selection, test, or code snapshot already exists.\n' >&2
    exit 73
}

gpu_ids="${CUDA_VISIBLE_DEVICES:?Slurm did not assign CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a gpus <<< "${gpu_ids}"
[[ ${#gpus[@]} -eq 4 ]] || { printf 'ERROR: formal run requires four assigned GPUs.\n' >&2; exit 75; }
for gpu in "${gpus[@]}"; do
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
    [[ "${utilization}" =~ ^[0-9]+$ ]] || { printf 'ERROR: invalid GPU utilization.\n' >&2; exit 75; }
    (( utilization < 50 )) || { printf 'ERROR: GPU%s utilization=%s.\n' "${gpu}" "${utilization}" >&2; exit 75; }
done

mkdir -p "${workspace}/training_runs/GOT" "${workspace}/evaluation_runs/GOT" \
    "${workspace}/runs" "${workspace}/tmp" "${workspace}/cache/huggingface" \
    "${workspace}/cache/torch_extensions" "$(dirname "${snapshot_root}")"
mkdir "${snapshot_root}"
cp -a "${workspace}/ocrmodel/src" "${workspace}/ocrmodel/tools" \
    "${workspace}/ocrmodel/config" "${snapshot_root}/"
export TMPDIR="${workspace}/tmp"
export HF_HOME="${workspace}/cache/huggingface"
export TORCH_EXTENSIONS_DIR="${workspace}/cache/torch_extensions"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export GOT_TRAINING_RUNS="${workspace}/training_runs/GOT"
export GOT_EVALUATION_RUNS="${workspace}/evaluation_runs/GOT"
source /home/bingxing2/apps/package/pytorch/2.0.1+cu118_cp310/env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u

project_root="${snapshot_root}/src/GOT-OCR-2.0"
runner="${snapshot_root}/tools/training/run_variable_layout_a100.py"
selector="${snapshot_root}/tools/evaluation/select_layout_ablation_checkpoint.py"
tester="${snapshot_root}/tools/evaluation/evaluate_layout_ablation_test.py"

run_pipeline() {
    python "${runner}" \
        --dataset-root "${dataset_root}/train" \
        --manifest "${dataset_root}/train/manifest.jsonl" \
        --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
        --test-manifest "${dataset_root}/test/manifest.jsonl" \
        --source-model "${model_dir}" --tokenizer-model "${model_dir}" \
        --stages p1,p2 --ablation vlqa_layout_p1_p2 --layout-loss-preset layout_full \
        --num-layout-prompt-queries 32 --max-layout-records 512 --max-layout-tokens 2048 \
        --layout-decoder-layers 2 --layout-decoder-hidden-size 256 --layout-decoder-num-heads 8 \
        --p1-max-steps 3000 --p2-max-steps 7500 \
        --p1-checkpoint-steps 1000 --p2-checkpoint-steps 2500 --checkpoint-retention 3 \
        --layout-boundary-loss-weight 1.0 --layout-count-condition-strength 1.0 \
        "${control_args[@]}" \
        --per-device-batch-size 1 --gradient-accumulation-steps 1 \
        --p1-learning-rate 1e-4 --p2-learning-rate 5e-5 \
        --gpu-ids "${gpu_ids}" --gpu-utilization-limit 50 \
        --distributed-strategy deepspeed_zero2 --seed 42 --run-id "${run_id}"

    python "${selector}" \
        --ablation vlqa_layout_p1_p2 --model-root "${run_root}/p2/model" --model-kind pvld \
        --tokenizer-model "${model_dir}" \
        --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
        --validation-image-root "${dataset_root}/validation" --output-dir "${selection_root}" \
        --project-root "${project_root}" --max-regions 512 --max-records 0 \
        --max-new-tokens 2048 --no-repeat-ngram-size 20 \
        --parallel-gpu-ids "${gpu_ids}" --gpu-utilization-limit 50

    python "${tester}" \
        --selection "${selection_root}/selection.json" --test-category Real-OOD \
        --test-manifest "${dataset_root}/test/manifest.jsonl" \
        --test-image-root "${dataset_root}/test" --model-kind pvld \
        --tokenizer-model "${model_dir}" --project-root "${project_root}" \
        --output-dir "${test_root}" --max-regions 512 --max-records 0 \
        --max-new-tokens 2048 --no-repeat-ngram-size 20 \
        --parallel-gpu-ids "${gpu_ids}" --gpu-utilization-limit 50
}

run_pipeline >"${control_log}" 2>&1
python - "${control}" "${run_id}" "${selection_root}/selection.json" "${test_root}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

control, run_id, selection_path, test_path = sys.argv[1:]
selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
test = json.loads(Path(test_path).read_text(encoding="utf-8"))
if selection.get("selection_split") != "validation" or selection.get("test_used_for_selection") is not False:
    raise RuntimeError("Validation-only selection contract was not preserved")
payload = {
    "event": "bscc_pvld_formal_completed",
    "control": control,
    "run_id": run_id,
    "selection": selection_path,
    "selected_step": selection["selected"]["optimizer_step"],
    "test": test_path,
    "test_used_for_selection": False,
    "input_granularity": "whole_page_image",
}
print(json.dumps(payload, separators=(",", ":")))
PY
