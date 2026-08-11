#!/usr/bin/env bash
set -euo pipefail

# Compatibility evaluation only. It reuses the preserved legacy evaluator's
# decoding parameters and keeps temporary links and outputs outside the repository.
umask 027

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
workspace="${OCR_WORKSPACE:-$(cd -- "${ocrmodel_root}/.." && pwd -P)}"
project_root="${GOT_PROJECT_ROOT:-${ocrmodel_root}/src/GOT-OCR-2.0}"
runner="${GOT_RUNNER:-${ocrmodel_root}/tools/environment/run_got2.sh}"
reference_eval="${ocrmodel_root}/references/legacy-ancientdoc-eval/GOT/eval/myeval.py"
metrics_script="${ocrmodel_root}/tools/evaluation/calculate_legacy_split5_metrics.py"
source_model="${GOT_EVAL_MODEL:-${GOT_SOURCE_MODEL:-${workspace}/models/GOT-OCR2_0}}"
tokenizer_model="${GOT_EVAL_TOKENIZER:-${GOT_SOURCE_MODEL:-${source_model}}}"
data_root="${ANCIENTDOC_ROOT:-${workspace}/datasets/AncientDoc}"
labels="${GOT_EVAL_LABELS:-${data_root}/label_for_got_split5.json}"
runs_root="${GOT_EVALUATION_RUNS:-${workspace}/runs/evaluation/GOT}"
eval_id="${GOT_EVAL_ID:-legacy_ancientdoc_split5_$(date +%Y%m%d_%H%M%S)}"
run_root="${runs_root}/${eval_id}"
facade_model="${run_root}/worktree/model_weights/model"
predictions="${run_root}/predictions.json"
metrics_dir="${run_root}/metrics"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

for required in \
    "${project_root}/pyproject.toml" \
    "${runner}" \
    "${reference_eval}" \
    "${metrics_script}" \
    "${source_model}/model.safetensors" \
    "${tokenizer_model}/tokenizer_config.json" \
    "${labels}" \
    "${data_root}/imgs"; do
    [[ -e "${required}" ]] || die "Required evaluation asset is missing: ${required}"
done
[[ -x "${workspace}/envs/got2/bin/python" ]] || die "Missing GOT environment: ${workspace}/envs/got2"

mkdir -p "${runs_root}"
exec 9>"${runs_root}/.legacy_ancientdoc_eval.lock"
flock -n 9 || die "GOT_ANCIENTDOC_EVAL_ALREADY_RUNNING"
[[ ! -e "${run_root}" ]] || die "Evaluation output already exists: ${run_root}"

gpu_apps="$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
if [[ -n "${gpu_apps//[[:space:]]/}" ]]; then
    printf 'GPU0_BUSY\n%s\n' "${gpu_apps}" >&2
    exit 75
fi

mkdir -p "${facade_model}" "${metrics_dir}" "${run_root}/metadata"
printf 'GOT_LEGACY_ANCIENTDOC_EVAL_STARTED run_root=%s log=%s\n' "${run_root}" "${run_root}/eval.log"
status_file="${run_root}/metadata/status.txt"
finished=0
finish_run() {
    local exit_code="$?"
    {
        echo "finished_at=$(date --iso-8601=seconds)"
        echo "exit_code=${exit_code}"
        echo "completed=${finished}"
    } >> "${status_file}"
}
trap finish_run EXIT

{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "eval_id=${eval_id}"
    echo "physical_gpu=0"
    echo "protocol=ancientdoc_page_reference_compatibility"
    echo "model=${source_model}"
    echo "labels=${labels}"
    echo "reference_evaluator=${reference_eval}"
} > "${status_file}"
sha256sum "${source_model}/model.safetensors" > "${run_root}/metadata/model.sha256"
sha256sum "${labels}" > "${run_root}/metadata/labels.sha256"

# A real directory containing symlinks avoids writing myeval.py's temporary
# model-name workaround beside a read-only shared model.
shopt -s dotglob nullglob
for asset in "${source_model}"/*; do
    ln -s "${asset}" "${facade_model}/$(basename -- "${asset}")"
done
shopt -u dotglob nullglob

export OCR_WORKSPACE="${workspace}"
export OCRMODEL_ROOT="${ocrmodel_root}"
export GOT_PROJECT_ROOT="${project_root}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1

cd "${run_root}/worktree"
"${runner}" "${reference_eval}" \
    --model-name "model_weights/model" \
    --gtfile_path "${labels}" \
    --image_path "${data_root}" \
    --out_file "${predictions}" \
    > "${run_root}/eval.log" 2>&1 || {
        exit_code="$?"
        echo "GOT_LEGACY_ANCIENTDOC_EVAL_FAILED exit_code=${exit_code}" >&2
        tail -n 20 "${run_root}/eval.log" >&2
        exit "${exit_code}"
    }

"${runner}" "${metrics_script}" \
    --predictions "${predictions}" \
    --labels "${labels}" \
    --tokenizer "${tokenizer_model}" \
    --output-dir "${metrics_dir}" \
    >> "${run_root}/eval.log" 2>&1 || {
        exit_code="$?"
        echo "GOT_LEGACY_METRICS_FAILED exit_code=${exit_code}" >&2
        tail -n 20 "${run_root}/eval.log" >&2
        exit "${exit_code}"
    }

touch "${run_root}/GOT_LEGACY_ANCIENTDOC_EVAL_FINISHED"
finished=1
echo "GOT_LEGACY_ANCIENTDOC_EVAL_OK"
echo "RUN_ROOT=${run_root}"
