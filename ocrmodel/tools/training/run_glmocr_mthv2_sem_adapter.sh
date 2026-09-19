#!/usr/bin/env bash
# Stage-two OCR training for layout/OCR decoupling.
# The generic MTHv2 launcher owns GPU admission, protocol handling, checkpoint
# selection, and the optional tmux foreground/background behavior.  This
# wrapper fixes the stage-two architecture and objective while leaving runtime
# paths and schedule overrides to the caller.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner="${GLMOCR_SEM_ADAPTER_RUNNER:-${script_dir}/run_glmocr_mthv2_ddp.sh}"
run_id="${GLMOCR_SEM_ADAPTER_RUN_ID:-glmocr_mthv2_sem_adapter_stage2_v1}"
seed="${GLMOCR_SEM_ADAPTER_SEED:-42}"
init_checkpoint_dir="${GLMOCR_SEM_ADAPTER_INIT_CHECKPOINT_DIR:-}"
stage2_learning_rate="${GLMOCR_SEM_ADAPTER_LEARNING_RATE:-1e-5}"
# The active GLMOCR runner caps the effective residual scale at 0.03.  Keep
# the default warm-start below that cap so the gate remains differentiable.
stage2_warm_start_gate="${GLMOCR_SEM_ADAPTER_WARM_START_CONTENT_GATE:-0.01}"
learning_rate_override_seen=0
gate_override_seen=0
forward_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --init-checkpoint-dir)
            [[ $# -ge 2 ]] || {
                printf '{"event":"glmocr_sem_adapter_failed","error":"missing_phase1_checkpoint_value"}\n' >&2
                exit 64
            }
            init_checkpoint_dir="$2"
            shift 2
            ;;
        --learning-rate)
            [[ $# -ge 2 ]] || {
                printf '{"event":"glmocr_sem_adapter_failed","error":"missing_learning_rate_value"}\n' >&2
                exit 64
            }
            learning_rate_override_seen=1
            forward_args+=("$1" "$2")
            shift 2
            ;;
        --warm-start-content-gate)
            [[ $# -ge 2 ]] || {
                printf '{"event":"glmocr_sem_adapter_failed","error":"missing_warm_start_gate_value"}\n' >&2
                exit 64
            }
            gate_override_seen=1
            forward_args+=(--init-checkpoint-override-residual-scale "$2")
            shift 2
            ;;
        --init-checkpoint-override-residual-scale)
            [[ $# -ge 2 ]] || {
                printf '{"event":"glmocr_sem_adapter_failed","error":"missing_checkpoint_gate_value"}\n' >&2
                exit 64
            }
            gate_override_seen=1
            forward_args+=("$1" "$2")
            shift 2
            ;;
        *)
            forward_args+=("$1")
            shift
            ;;
    esac
done
set -- "${forward_args[@]}"

stage2_defaults=()
(( learning_rate_override_seen == 1 )) || stage2_defaults+=(--learning-rate "${stage2_learning_rate}")
(( gate_override_seen == 1 )) || stage2_defaults+=(--init-checkpoint-override-residual-scale "${stage2_warm_start_gate}")

[[ -f "${runner}" ]] || {
    printf '{"event":"glmocr_sem_adapter_failed","error":"missing_stage2_runner","runner":"%s"}\n' "${runner}" >&2
    exit 66
}
[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ && "${seed}" =~ ^[0-9]+$ ]] || {
    printf '{"event":"glmocr_sem_adapter_failed","error":"invalid_run_or_seed"}\n' >&2
    exit 64
}
[[ -n "${init_checkpoint_dir}" ]] || {
    printf '{"event":"glmocr_sem_adapter_failed","error":"missing_phase1_checkpoint"}\n' >&2
    exit 64
}
[[ -d "${init_checkpoint_dir}" && -f "${init_checkpoint_dir}/adapter.safetensors" ]] || {
    printf '{"event":"glmocr_sem_adapter_failed","error":"invalid_phase1_checkpoint","checkpoint":"%s"}\n' "${init_checkpoint_dir}" >&2
    exit 66
}

# Fixed arguments are appended after caller arguments so the stage-two
# contract cannot be accidentally replaced by a schedule/path override.
exec bash "${runner}" \
    "$@" \
    "${stage2_defaults[@]}" \
    --run-id "${run_id}" \
    --seed "${seed}" \
    --experiment-group custom \
    --mode layout_ot \
    --decoder-adaptation lora \
    --layout-loss-profile ocr_only \
    --auxiliary-weight 0 \
    --auxiliary-weight-start 0 \
    --sem-adapter-mlp \
    --freeze-layout-branch \
    --init-checkpoint-dir "${init_checkpoint_dir}"
