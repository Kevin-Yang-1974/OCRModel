#!/usr/bin/env bash
set -euo pipefail

models_root="${SOTA_MODELS_ROOT:-/data3/yky/yangky_ocr_models/models/sota}"
env_root="${SOTA_ENVS_ROOT:-/data3/yky/yangky_ocr_models/envs/sota}"
python_bin="${SOTA_PYTHON:-python3}"
hf_python="${SOTA_HF_PYTHON:-}"
# overwrite=false: existing model directories are never replaced by this script.
mkdir -p "${models_root}" "${env_root}"

declare -A ids=(
  [paddleocr_vl_1_6]="PaddlePaddle/PaddleOCR-VL-1.6"
  [mineru2_5_pro]="opendatalab/MinerU2.5-Pro-2605-1.2B"
  [glm_ocr]="zai-org/GLM-OCR"
  [opendoc_0_1b]="topdu/unirec-0.1b"
)
declare -A revisions=(
  [paddleocr_vl_1_6]="c5630abae1d940eafe0697512a0325494b02ab42"
  [mineru2_5_pro]="bff20d4ae2bf202df9f45284b4d43681555a97ed"
  [glm_ocr]="ca5d8b3e287e52589e37c28385d9655ee4372f9d"
  [opendoc_0_1b]="a377e00d62c01b6544603e2a90f2cffe2a0388e1"
)

for name in paddleocr_vl_1_6 mineru2_5_pro glm_ocr opendoc_0_1b; do
  model_dir="${models_root}/${name}/${revisions[$name]}"
  env_dir="${env_root}/${name}"
  status_file="${model_dir}/deployment_status.json"
  mkdir -p "${models_root}/${name}"
  if [[ -e "${model_dir}" && ! -f "${status_file}" ]]; then
    printf '{"model":"%s","status":"blocked_existing_unregistered_directory","path":"%s"}\n' "$name" "$model_dir"
    continue
  fi
  if [[ ! -d "${env_dir}" ]]; then
    if ! "${python_bin}" -m venv "${env_dir}"; then
      printf '{"model":"%s","status":"blocked_missing_python_venv","env_dir":"%s"}\n' "$name" "$env_dir" | tee "${env_dir}.status.json"
      # Keep deploying independent model weights; dependencies can be installed
      # later by the provider-specific environment setup.
    fi
  fi
  if [[ ! -f "${model_dir}/config.json" && ! -f "${model_dir}/.download_complete" ]]; then
    mkdir -p "${model_dir}"
    if command -v hf >/dev/null 2>&1; then
      hf download "${ids[$name]}" --revision "${revisions[$name]}" --local-dir "${model_dir}" --local-dir-use-symlinks False
    elif command -v huggingface-cli >/dev/null 2>&1; then
      huggingface-cli download "${ids[$name]}" --revision "${revisions[$name]}" --local-dir "${model_dir}" --local-dir-use-symlinks False
    elif [[ -n "${hf_python}" && -x "${hf_python}" ]]; then
      "${hf_python}" -c 'from huggingface_hub import snapshot_download; import sys; snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])' \
        "${ids[$name]}" "${revisions[$name]}" "${model_dir}"
    else
      printf '{"model":"%s","status":"blocked_missing_huggingface_downloader","checkpoint_id":"%s"}\n' "$name" "${ids[$name]}" | tee "${status_file}"
      continue
    fi
    touch "${model_dir}/.download_complete"
  fi
  printf '{"model":"%s","checkpoint_id":"%s","revision":"%s","model_dir":"%s","env_dir":"%s","status":"deployed","overwrite":false}\n' \
    "$name" "${ids[$name]}" "${revisions[$name]}" "$model_dir" "$env_dir" | tee "$status_file"
done
