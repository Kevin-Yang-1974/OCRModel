#!/usr/bin/env bash
#SBATCH --job-name=mthv2_col_compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%A_%a.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%A_%a.log

set -euo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
code_root="${workspace}/glm_ocr_layout_ot/code/ocrmodel"
data_root="${workspace}/datasets/MTHv2/converted/mthv2_anandasky_gt_lines_v1"
run_id="${COL_COMPARE_RUN_ID:-mthv2_column_compare_1000_260920_v1}"
stage="${COL_COMPARE_STAGE:-smoke}"
limit="${COL_COMPARE_LIMIT:-2}"
output_root="${workspace}/runs/${run_id}/${stage}"
manifest="${data_root}/test/manifest_subset_1000.jsonl"
train_manifest="${data_root}/train/manifest.jsonl"
pytorch_package="/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311"
got_pytorch_package="/home/bingxing2/apps/package/pytorch/2.0.1+cu118_cp310"

[[ -f "${manifest}" && -f "${train_manifest}" ]] || {
    printf 'ERROR: MTHv2 column manifests are missing.\n' >&2
    exit 66
}
[[ -f "${code_root}/tools/evaluation/evaluate_mthv2_columns.py" ]] || {
    printf 'ERROR: column evaluator is missing.\n' >&2
    exit 66
}

mkdir -p "${output_root}"
case "${SLURM_ARRAY_TASK_ID:?}" in
    0)
        model="got"
        python="${workspace}/envs/got2-py310-cu118/bin/python"
        model_dir="${workspace}/models/GOT-OCR2-original"
        got_source_root="${workspace}/ocrmodel/src/GOT-OCR-2.0"
        output_dir="${output_root}/got"
        extra=(--got-source-root "${got_source_root}")
        ;;
    1)
        model="glm"
        python="${workspace}/glm_ocr_layout_ot/envs/glm-ocr-py311-cu128/bin/python"
        model_dir="${workspace}/models/GLM-OCR-ca5d8b3"
        output_dir="${output_root}/glm"
        extra=()
        ;;
    *)
        printf 'ERROR: unsupported array task %s.\n' "${SLURM_ARRAY_TASK_ID}" >&2
        exit 64
        ;;
esac

[[ -x "${python}" && -f "${model_dir}/model.safetensors" ]] || {
    printf 'ERROR: model environment or weights are missing for %s.\n' "${model}" >&2
    exit 66
}
[[ ! -e "${output_dir}" ]] || {
    printf 'ERROR: output already exists: %s\n' "${output_dir}" >&2
    exit 74
}

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID

if [[ "${model}" == "got" ]]; then
    [[ -f "${got_pytorch_package}/env.sh" ]] || {
        printf 'ERROR: GOT PyTorch runtime is missing.\n' >&2
        exit 66
    }
    source "${got_pytorch_package}/env.sh"
    got_python_lib="${workspace}/envs/got2-py310-cu118/lib/python3.10/site-packages"
    export LD_LIBRARY_PATH="${got_python_lib}/torch/lib:${got_python_lib}/nvidia/cuda_runtime/lib:${got_python_lib}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
else
    [[ -f "${pytorch_package}/env.sh" ]] || {
        printf 'ERROR: shared PyTorch runtime is missing.\n' >&2
        exit 66
    }
    source "${pytorch_package}/env.sh"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    set +u
    conda activate "${workspace}/glm_ocr_layout_ot/envs/glm-ocr-py311-cu128"
    set -u
fi

printf '{"event":"column_eval_start","model":"%s","limit":%s,"output_dir":"%s","test_used_for_selection":false}\n' \
    "${model}" "${limit}" "${output_dir}"
"${python}" "${code_root}/tools/evaluation/evaluate_mthv2_columns.py" \
    --model "${model}" \
    --model-dir "${model_dir}" \
    --manifest "${manifest}" \
    --train-manifest "${train_manifest}" \
    --output-dir "${output_dir}" \
    --max-new-tokens 256 \
    --limit "${limit}" \
    "${extra[@]}"
