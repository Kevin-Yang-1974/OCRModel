#!/usr/bin/env bash
# P2-only retry for the failed BSCC M4 run. It preserves the failed run and
# initializes exclusively from that run's validation-selected P1 checkpoint.
# Validation controls and test are deliberately not launched by this entry.
#SBATCH --job-name=pvld_m4_p2_retry
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
failed_run_id="${PVLD_M4_FAILED_RUN_ID:-mthv2_pvld_m4_predrouting_20260826_v1}"
run_id="${PVLD_M4_RETRY_RUN_ID:?set a new PVLD_M4_RETRY_RUN_ID}"
selection="${workspace}/training_runs/GOT/${failed_run_id}/p1/validation_selection/selection.json"
run_root="${workspace}/training_runs/GOT/${run_id}"
snapshot_root="${workspace}/run_code/${run_id}/ocrmodel"
control_log="${workspace}/runs/${run_id}.pipeline.log"

[[ -f "${selection}" ]] || { printf 'ERROR: failed M4 P1 validation selection is missing.\n' >&2; exit 66; }
[[ -f "${dataset_root}/train/manifest.jsonl" && -f "${dataset_root}/validation/manifest.jsonl" ]] || {
    printf 'ERROR: required MTHv2 train/validation manifests are missing.\n' >&2; exit 66;
}
[[ ! -e "${run_root}" && ! -e "${snapshot_root}" ]] || {
    printf 'ERROR: retry run or code snapshot already exists.\n' >&2; exit 73;
}

gpu_ids="${CUDA_VISIBLE_DEVICES:?Slurm did not assign CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a gpus <<< "${gpu_ids}"
[[ ${#gpus[@]} -eq 4 ]] || { printf 'ERROR: M4 P2 retry requires four assigned GPUs.\n' >&2; exit 75; }
for gpu in "${gpus[@]}"; do
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
    [[ "${utilization}" =~ ^[0-9]+$ ]] || { printf 'ERROR: invalid GPU utilization.\n' >&2; exit 75; }
    (( utilization < 50 )) || { printf 'ERROR: GPU%s utilization=%s.\n' "${gpu}" "${utilization}" >&2; exit 75; }
done

mkdir -p "${workspace}/training_runs/GOT" "${workspace}/runs" "${workspace}/tmp" \
    "${workspace}/cache/huggingface" "${workspace}/cache/torch_extensions" "$(dirname "${snapshot_root}")"
mkdir "${snapshot_root}"
cp -a "${workspace}/ocrmodel/src" "${workspace}/ocrmodel/tools" \
    "${workspace}/ocrmodel/config" "${snapshot_root}/"

export TMPDIR="${workspace}/tmp"
export HF_HOME="${workspace}/cache/huggingface"
export TORCH_EXTENSIONS_DIR="${workspace}/cache/torch_extensions"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export GOT_TRAINING_RUNS="${workspace}/training_runs/GOT"
source /home/bingxing2/apps/package/pytorch/2.0.1+cu118_cp310/env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u

selected_model="$(python - "${selection}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("selection_split") != "validation" or payload.get("test_used_for_selection") is not False:
    raise RuntimeError("M4 retry requires validation-only P1 selection")
print(payload["selected"]["model_path"])
PY
)"

project_root="${snapshot_root}/src/GOT-OCR-2.0"
runner="${snapshot_root}/tools/training/run_variable_layout_a100.py"
python "${runner}" \
    --dataset-root "${dataset_root}/train" --manifest "${dataset_root}/train/manifest.jsonl" \
    --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
    --test-manifest "${dataset_root}/test/manifest.jsonl" \
    --source-model "${selected_model}" --source-validation-selection "${selection}" \
    --tokenizer-model "${model_dir}" --stages p2 --ablation vlqa_layout_p1_p2 \
    --layout-loss-preset layout_full --num-layout-prompt-queries 32 \
    --max-layout-records 512 --max-layout-tokens 2048 --layout-decoder-layers 2 \
    --layout-decoder-hidden-size 256 --layout-decoder-num-heads 8 --p2-max-steps 7500 \
    --p2-checkpoint-steps 2500 --checkpoint-retention 3 --layout-boundary-loss-weight 1.0 \
    --layout-count-condition-strength 1.0 --pvld-use-spatial-memory \
    --pvld-shared-gradient-scale 0.25 --pvld-record-gradient-scale 0.25 \
    --pvld-predicted-layout-routing --per-device-batch-size 1 --gradient-accumulation-steps 1 \
    --p1-learning-rate 1e-4 --p2-learning-rate 5e-5 --gpu-ids "${gpu_ids}" \
    --gpu-utilization-limit 50 --distributed-strategy deepspeed_zero2 --seed 42 \
    --run-id "${run_id}" >"${control_log}" 2>&1

python - "${run_id}" "${failed_run_id}" "${selection}" "${run_root}" <<'PY'
import json
import sys
from pathlib import Path

run_id, failed_run_id, selection, run_root = sys.argv[1:]
payload = {
    "event": "bscc_m4_p2_retry_completed",
    "run_id": run_id,
    "retry_of": failed_run_id,
    "source_p1_selection": selection,
    "scope": "p2_only; validation controls and test not launched",
}
(Path(run_root) / "m4_retry_provenance.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, separators=(",", ":")))
PY
