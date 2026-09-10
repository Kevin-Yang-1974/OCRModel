#!/usr/bin/env bash
# Run the bounded seed42 mechanism check for the validity/no-object fix.
# Training remains on the full MTHv2 train split; validation is a deterministic
# small subset so the mechanism check does not spend most of its budget in
# generation.  The subset tool reads validation only and never opens test.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
dataset_root="${GLMOCR_A100_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1}"
run_id="${GLMOCR_VALIDITY_RUN_ID:-glmocr_mthv2_validity_assignment_256_v1}"
validation_pages="${GLMOCR_VALIDATION_SUBSET_PAGES:-32}"
validation_seed="${GLMOCR_VALIDATION_SUBSET_SEED:-42}"
validation_manifest="${remote_root}/protocols/${run_id}.validation${validation_pages}.seed${validation_seed}.jsonl"
protocol_file="${remote_root}/protocols/${run_id}.train_validation${validation_pages}.without_test.json"
python="${env_dir}/bin/python"

[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ && "${validation_pages}" =~ ^[1-9][0-9]*$ && "${validation_seed}" =~ ^[0-9]+$ ]] || {
    printf '{"event":"glmocr_validity_assignment_failed","error":"invalid_validation_subset"}\n' >&2
    exit 64
}
[[ -x "${python}" ]] || {
    printf '{"event":"glmocr_validity_assignment_failed","error":"missing_python","python":"%s"}\n' "${python}" >&2
    exit 66
}
[[ ! -e "${validation_manifest}" && ! -e "${protocol_file}" ]] || {
    printf '{"event":"glmocr_validity_assignment_failed","error":"subset_protocol_already_exists","validation_manifest":"%s","protocol_file":"%s"}\n' \
        "${validation_manifest}" "${protocol_file}" >&2
    exit 74
}
"${python}" "${code_root}/tools/subset_mthv2_manifest.py" \
    --input "${dataset_root}/validation/manifest.jsonl" \
    --output "${validation_manifest}" \
    --count "${validation_pages}" \
    --seed "${validation_seed}"

exec bash "${script_dir}/run_glmocr_mthv2_ddp.sh" \
    --run-id "${run_id}" \
    --seed 42 \
    --max-steps 256 \
    --lr-schedule-steps 256 \
    --learning-rate 2.5e-5 \
    --warmup-steps 32 \
    --min-lr-ratio 0.5 \
    --gradient-accumulation-steps 4 \
    --initial-residual-scale 0 \
    --gate-freeze-steps 64 \
    --auxiliary-weight-start 0.05 \
    --auxiliary-weight 0.2 \
    --auxiliary-ramp-steps 256 \
    --layout-loss-profile validity_assignment \
    --initial-valid-probability 0.066 \
    --validity-gating-mode raw_mass \
    --validity-use-transport-evidence \
    --validation-manifest "${validation_manifest}" \
    --protocol-file "${protocol_file}" \
    --remote-root "${remote_root}" \
    --code-root "${code_root}" \
    --env-dir "${env_dir}" \
    --dataset-root "${dataset_root}" \
    --diagnostic-steps 0,64,128,256 \
    --validation-interval 256 \
    --log-steps 16 \
    --max-eval-new-tokens 1536 \
    --skip-selection \
    --without-test \
    "$@"
