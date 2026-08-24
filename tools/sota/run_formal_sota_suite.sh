#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_FORMAL_SOTA:-0}" != "1" ]]; then
  printf '%s\n' '{"event":"formal_sota_locked","reason":"Set ALLOW_FORMAL_SOTA=1 only after explicit user authorization."}' >&2
  exit 77
fi

run_root="${SOTA_RUN_ROOT:?Set SOTA_RUN_ROOT to a new evaluation_runs/SOTA directory}"
manifest_root="${MTHV2_PAGE_ROOT:?Set MTHV2_PAGE_ROOT to mthv2_layout_page_v1}"
models_root="${SOTA_MODELS_ROOT:?Set SOTA_MODELS_ROOT to the deployed model root}"
export PYTHONPATH="${SOTA_SOURCE_ROOT:-/data3/yky/yangky_ocr_models/ocrmodel}:${PYTHONPATH:-}"
cd /data3/yky/yangky_ocr_models/ocrmodel
mkdir -p "$run_root"
test_used_for_selection=false

gpu_ids="${SOTA_GPU_IDS:-0,1,3,4}"
if [[ ",$gpu_ids," == *,2,* ]]; then
  printf '%s\n' '{"event":"formal_sota_refused","reason":"GPU 2 is reserved and must not be included."}' >&2
  exit 78
fi
IFS=',' read -r -a gpu_array <<< "$gpu_ids"
for gpu in "${gpu_array[@]}"; do
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d '[:space:]')"
  if [[ "$util" -ge 50 ]]; then
    printf '{"event":"formal_sota_refused","gpu":"%s","utilization_gpu":%s}\n' "$gpu" "$util" >&2
    exit 79
  fi
done

declare -A revisions=(
  [paddleocr_vl_1_6]="c5630abae1d940eafe0697512a0325494b02ab42"
  [mineru2_5_pro]="bff20d4ae2bf202df9f45284b4d43681555a97ed"
  [glm_ocr]="ca5d8b3e287e52589e37c28385d9655ee4372f9d"
  [opendoc_0_1b]="a377e00d62c01b6544603e2a90f2cffe2a0388e1"
)
declare -A devices=(
  [paddleocr_vl_1_6]="cuda:0"
  [mineru2_5_pro]="cuda:1"
  [glm_ocr]="cuda:3"
  [opendoc_0_1b]="cuda:4"
)
declare -A python_bins=(
  [paddleocr_vl_1_6]="/data3/yky/yangky_ocr_models/envs/anandasky/bin/python"
  [mineru2_5_pro]="/data3/yky/yangky_ocr_models/envs/sota/mineru2_5_pro/bin/python"
  [glm_ocr]="/data3/yky/yangky_ocr_models/envs/anandasky/bin/python"
  # OpenDoc's installed OpenCV/NumPy stack is ABI-compatible with the
  # existing Python 3.11 AnandaSky environment; its original 3.12 venv
  # mixed a cp311 NumPy wheel and failed before model loading.
  [opendoc_0_1b]="/data3/yky/yangky_ocr_models/envs/anandasky/bin/python"
)
declare -A site_packages=(
  [paddleocr_vl_1_6]="/data3/yky/yangky_ocr_models/envs/sota/glm_overlay_20260823:/data3/yky/yangky_ocr_models/envs/anandasky/lib/python3.11/site-packages"
  [mineru2_5_pro]="/data3/yky/yangky_ocr_models/envs/sota/mineru2_5_pro/site-packages"
  [glm_ocr]="/data3/yky/yangky_ocr_models/envs/sota/glm_overlay_20260823:/data3/yky/yangky_ocr_models/envs/anandasky/lib/python3.11/site-packages"
  [opendoc_0_1b]="/data3/yky/yangky_ocr_models/envs/sota/opendoc_onnx_overlay_20260824:/data3/yky/yangky_ocr_models/envs/sota/opendoc_overlay_20260823:/data3/yky/yangky_ocr_models/envs/sota/opendoc_0_1b/site-packages:/data3/yky/yangky_ocr_models/envs/anandasky/lib/python3.11/site-packages"
)

models_to_run="${SOTA_MODELS:-paddleocr_vl_1_6 mineru2_5_pro glm_ocr opendoc_0_1b}"
limit_args=()
if [[ -n "${SOTA_LIMIT:-}" ]]; then
  limit_args=(--limit "$SOTA_LIMIT")
fi
for model in $models_to_run; do
  revision="${revisions[$model]}"
  model_root="$models_root/$model/$revision"
  base="$run_root/$model"
  python_bin="${python_bins[$model]}"
  model_pythonpath="${site_packages[$model]}:/data3/yky/yangky_ocr_models/ocrmodel:${PYTHONPATH}"
  model_path="${site_packages[$model]}/bin:${PATH}"
  mkdir -p "$base"
  printf '{"event":"sota_model_started","model":"%s","device":"%s","test_used_for_selection":false}\n' "$model" "${devices[$model]}" | tee "$base/status.json"

  # The repository has no verified long-run trainer for MinerU/OpenDoc and
  # the provider-specific Paddle/GLM trainers are external contracts. Record
  # that fact explicitly, then run the common official checkpoint evaluator.
  if [[ "$model" == "glm_ocr" || "$model" == "paddleocr_vl_1_6" ]]; then
    PATH="$model_path" PYTHONPATH="$model_pythonpath" "$python_bin" -m tools.sota.run_finetune_smoke \
      --model "$model" --model-root "$model_root" \
      --manifest "$manifest_root/train/manifest.jsonl" \
      --image-root "$manifest_root/train" --output-dir "$base/training_smoke" \
      --device "${devices[$model]}" --dtype bf16 > "$base/training.log" 2>&1 || true
  else
    printf '{"event":"formal_training_blocked","model":"%s","reason":"official_finetuning_unavailable","test_used":false}\n' "$model" | tee "$base/training_blocked.json"
  fi

  if [[ "$model" == "opendoc_0_1b" ]]; then
    PATH="$model_path" PYTHONPATH="/data3/yky/yangky_ocr_models/ocrmodel:$model_pythonpath" OPENOCR_PYTHONPATH="${site_packages[$model]}" OPENOCR_ONNX=1 OPENOCR_ONNX_CACHE=/data3/yky/yangky_ocr_models/models/sota/opendoc_onnx_cache_20260824 "$python_bin" -m tools.sota.run_zero_shot \
      --model "$model" --model-root "$model_root" \
      --manifest "$manifest_root/validation/manifest.jsonl" \
      --image-root "$manifest_root/validation" --output-dir "$base/validation" \
      --split validation "${limit_args[@]}" --device "${devices[$model]}" --dtype bf16 > "$base/validation.log" 2>&1 || true
  else
    PATH="$model_path" PYTHONPATH="$model_pythonpath" OPENOCR_PYTHONPATH="${site_packages[$model]}" "$python_bin" -m tools.sota.run_zero_shot \
      --model "$model" --model-root "$model_root" \
      --manifest "$manifest_root/validation/manifest.jsonl" \
      --image-root "$manifest_root/validation" --output-dir "$base/validation" \
      --split validation "${limit_args[@]}" --device "${devices[$model]}" --dtype bf16 > "$base/validation.log" 2>&1 || true
  fi
  PATH="$model_path" PYTHONPATH="/data3/yky/yangky_ocr_models/ocrmodel:$model_pythonpath" OPENOCR_PYTHONPATH="${site_packages[$model]}" OPENOCR_ONNX=1 OPENOCR_ONNX_CACHE=/data3/yky/yangky_ocr_models/models/sota/opendoc_onnx_cache_20260824 "$python_bin" -m tools.sota.select_validation \
    --predictions "$base/validation/predictions.jsonl" --output "$base/selection.json" \
    --model "$model" --checkpoint "$model_root" > "$base/selection.log" 2>&1 || true
  if [[ -f "$base/selection.json" && "${SOTA_SKIP_TEST:-0}" != "1" ]]; then
    PATH="$model_path" PYTHONPATH="/data3/yky/yangky_ocr_models/ocrmodel:$model_pythonpath" OPENOCR_PYTHONPATH="${site_packages[$model]}" OPENOCR_ONNX=1 OPENOCR_ONNX_CACHE=/data3/yky/yangky_ocr_models/models/sota/opendoc_onnx_cache_20260824 "$python_bin" -m tools.sota.run_selection_locked_test \
      --selection "$base/selection.json" --model "$model" --model-root "$model_root" \
      --test-manifest "$manifest_root/test/manifest.jsonl" --image-root "$manifest_root/test" \
      --output-dir "$base/test" --device "${devices[$model]}" --dtype bf16 \
      --allow-formal-test > "$base/test.log" 2>&1 || true
  fi
  printf '{"event":"sota_model_finished","model":"%s","test_used_for_selection":false}\n' "$model" | tee "$base/finished.json"
done

printf '{"event":"formal_sota_suite_complete","run_root":"%s","gpu_ids":"%s","test_used_for_selection":false}\n' "$run_root" "$gpu_ids"
