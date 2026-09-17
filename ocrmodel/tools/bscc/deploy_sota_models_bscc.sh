#!/usr/bin/env bash
# Download official external-SOTA checkpoints for the MTHv2 zero-shot protocol.
#
# Runs on the BSCC login node.  Uses ModelScope (fast from CN) instead of the
# Hugging Face mirror; the downloader is a plain-HTTP helper that records file
# sha256 digests.  Compute nodes are offline, so weights are staged here before
# any Slurm inference job starts.  Existing complete files are never replaced.
set -euo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
models_root="${SOTA_MODELS_ROOT:-${workspace}/models/sota}"
code_root="${workspace}/glm_ocr_layout_ot/code/ocrmodel"
downloader="${code_root}/tools/bscc/download_modelscope.py"
python_bin="${SOTA_DOWNLOAD_PYTHON:-python3}"
mkdir -p "${models_root}"

download() {
    local repo="$1" target="$2" marker="$3"
    if [[ -f "${target}/${marker}" ]]; then
        printf '{"event":"sota_model_present","repo":"%s","target":"%s"}\n' "${repo}" "${target}"
        return 0
    fi
    rm -rf "${target}/.cache" 2>/dev/null || true
    "${python_bin}" "${downloader}" --repo "${repo}" --target "${target}" --exclude .gitattributes
}

download "PaddlePaddle/PaddleOCR-VL-1.6" \
    "${models_root}/paddleocr_vl_1_6/c5630abae1d940eafe0697512a0325494b02ab42" \
    "model.safetensors"

download "opendatalab/MinerU2.5-Pro-2605-1.2B" \
    "${models_root}/mineru2_5_pro/bff20d4ae2bf202df9f45284b4d43681555a97ed" \
    "model.safetensors"

download "topdktu/unirec_0_1b_onnx" \
    "${models_root}/opendoc_onnx_20260912" \
    "unirec_decoder.onnx"

printf '{"event":"sota_deploy_complete","models_root":"%s"}\n' "${models_root}"
