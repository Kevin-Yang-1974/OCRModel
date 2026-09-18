#!/usr/bin/env bash
# One more 256-step stage-two schedule on the Dunhuang/local-gazetteer corpus,
# resumed from the finished 256-step sem_adapter run instead of from the
# stage-one layout checkpoint.
#
# The parent run trained with --no-validation and was scored only at its final
# step (59-page locked test CER 0.170694).  Its log shows the semantic gate
# still climbing monotonically (0.010253 -> 0.012923) far below the 0.03 cap
# and no over-generation, so "is there training room" was left unresolved
# rather than answered: there is no intermediate point and no step-0 control
# to compare against.  This launcher supplies both, scoring the 80-page
# validation split at 64/128/192/256 plus the untouched resume checkpoint.
#
# Training defers its validation, and the five evaluations then run as five
# independent single-GPU --eval-only processes instead of the trainer's own
# rank-0-only post-hoc loop: one 80-page pass takes ~19 minutes, so the serial
# form costs ~95 minutes and leaves four GPUs idle.  Selection is still the
# lowest validation CER, aggregated by the shared parallel-validation
# summarizer, and the test split is opened once, after selection.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dunhuang_root="${GLMOCR_Q32_DATASET_ROOT:-/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"
gpu_ids="${GLMOCR_DH_CONTINUE_GPU_IDS:-0,1,2,3,4}"
gpu_utilization_limit="${GLMOCR_DH_CONTINUE_GPU_UTILIZATION_LIMIT:-50}"
seed="${GLMOCR_DH_CONTINUE_SEED:-42}"
num_queries="${GLMOCR_DH_CONTINUE_NUM_QUERIES:-32}"
steps="${GLMOCR_DH_CONTINUE_STEPS:-256}"
validation_interval="${GLMOCR_DH_CONTINUE_VALIDATION_INTERVAL:-64}"
learning_rate="${GLMOCR_DH_CONTINUE_LR:-1e-5}"
decoder_learning_rate="${GLMOCR_DH_CONTINUE_DECODER_LR:-1e-6}"
warmup_steps="${GLMOCR_DH_CONTINUE_WARMUP_STEPS:-64}"
min_lr_ratio="${GLMOCR_DH_CONTINUE_MIN_LR_RATIO:-0.1}"
max_eval_new_tokens="${GLMOCR_DH_CONTINUE_MAX_EVAL_NEW_TOKENS:-1536}"
# Fixed by the shared DDP runner.  The parallel-validation summarizer rejects
# any evaluation whose metadata diverges from the training run on these, so the
# eval-only invocations below have to repeat them verbatim.
eval_generation_mode="loop_recovery"
# Visual-token budget.  The ablation in
# results/glmocr_dunhuang_stage2_ablation_resolution_20260918_v1/ showed the
# 1003520 default downscales every Dunhuang page to about a third of its
# native pixels and costs ~0.098 CER, so training and both evaluation
# stages take this one value.
eval_max_pixels="${GLMOCR_DH_CONTINUE_MAX_PIXELS:-1003520}"
# Resumed gate: the parent run's final raw_content_gate.  Passing it explicitly
# overrides the sem_adapter wrapper's 0.01 warm start so the continuation keeps
# the gate the parent already learned instead of resetting it downward.
resume_gate="${GLMOCR_DH_CONTINUE_RESUME_GATE:-0.012923270463943481}"
parent_run_id="${GLMOCR_DH_CONTINUE_PARENT_RUN_ID:-glmocr_dunhuang_local_sem_adapter_gate001_lr1e5_dec1e6_256_from_giou20x10000_20260918_v1}"
run_id="${GLMOCR_DH_CONTINUE_RUN_ID:-glmocr_dunhuang_local_sem_adapter_continue256_val64_gate0013_lr1e5_dec1e6_20260918_v1}"
session="${GLMOCR_DH_CONTINUE_SESSION:-glmocr_dh_sem_adapter_continue256_20260918_v1}"
foreground=0
resume_validation=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --foreground) foreground=1; shift ;;
        --resume-validation) resume_validation=1; shift ;;
        *) printf '{"event":"glmocr_dh_continue_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

[[ "${seed}" =~ ^[0-9]+$ && "${num_queries}" == "32" ]] || exit 64
[[ "${steps}" == "256" && "${validation_interval}" == "64" && "${warmup_steps}" == "64" ]] || exit 64
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || exit 64
[[ "${gpu_utilization_limit}" =~ ^[1-9][0-9]*$ && "${max_eval_new_tokens}" =~ ^[1-9][0-9]*$ ]] || exit 64
for candidate in "${parent_run_id}" "${run_id}" "${session}"; do
    [[ "${candidate}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
done
[[ "${resume_gate}" =~ ^0\.[0-9]+$ ]] || exit 64

python="${env_dir}/bin/python"
sem_adapter_wrapper="${code_root}/tools/training/run_glmocr_mthv2_sem_adapter.sh"
locked_test_launcher="${code_root}/tools/training/run_glmocr_mthv2_locked_test.sh"
manifest_auditor="${code_root}/tools/audit_mthv2_manifest.py"
parallel_summarizer="${code_root}/tools/summarize_glmocr_parallel_validation.py"
workspace_runs="${remote_root}/runs"
status_file="${workspace_runs}/${session}.status.json"
summary_file="${workspace_runs}/${session}.summary.json"
parent_run_dir="${remote_root}/training_runs/${parent_run_id}/seed${seed}"
# The trainer loads adapter_config.json alongside adapter.safetensors, so the
# resume point is the parent run's final per-step checkpoint, not its root.
parent_checkpoint_dir="${GLMOCR_DH_CONTINUE_PARENT_CHECKPOINT:-${parent_run_dir}/checkpoint-256}"
group_root="${remote_root}/training_runs/${run_id}"
run_dir="${group_root}/seed${seed}"
validation_root="${run_dir}/parallel-validation"
train_protocol="${remote_root}/protocols/${run_id}.train_validation_no_test.json"
test_protocol="${remote_root}/protocols/${run_id}.test_locked.json"

write_status() {
    local status="$1"
    local phase="$2"
    local active_run_id="${3:--}"
    printf '{"status":"%s","phase":"%s","run_id":"%s","session":"%s","dataset":"dunhuang_local_gazetteer_q32_v1","seed":%s,"num_queries":%s,"steps":%s,"validation_interval":%s,"parent_run_id":"%s","gpu_ids":"%s","test_used_for_selection":false,"updated_at":"%s"}\n' \
        "${status}" "${phase}" "${active_run_id}" "${session}" "${seed}" "${num_queries}" \
        "${steps}" "${validation_interval}" "${parent_run_id}" "${gpu_ids}" "$(date -u +%FT%TZ)" > "${status_file}"
}

on_error() {
    local rc=$?
    write_status failed "${current_phase:-unknown}_failed" "${current_run_id:--}" || true
    exit "${rc}"
}
trap on_error ERR

admit_gpu_set() {
    local requested="$1"
    declare -A observed=()
    while IFS=',' read -r gpu utilization; do
        gpu="${gpu//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        [[ "${gpu}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || exit 69
        observed["${gpu}"]="${utilization}"
    done < <(nvidia-smi -i "${requested}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
    IFS=',' read -r -a requested_gpus <<< "${requested}"
    for gpu in "${requested_gpus[@]}"; do
        [[ -n "${observed[${gpu}]+present}" ]] || exit 69
        (( observed[${gpu}] < gpu_utilization_limit )) || {
            printf '{"event":"glmocr_dh_continue_failed","error":"gpu_admission_failed","gpu":%s,"utilization":%s,"limit":%s}\n' \
                "${gpu}" "${observed[${gpu}]}" "${gpu_utilization_limit}" >&2
            exit 75
        }
    done
    printf '{"event":"glmocr_dh_continue_gpu_admission_ok","gpu_ids":"%s"}\n' "${requested}"
}

preflight() {
    [[ -x "${python}" && -x "${env_dir}/bin/torchrun" ]] || exit 66
    [[ -f "${model_dir}/model.safetensors" && -f "${model_dir}/config.json" ]] || exit 66
    for path in "${sem_adapter_wrapper}" "${locked_test_launcher}" "${manifest_auditor}" "${parallel_summarizer}"; do
        [[ -f "${path}" ]] || exit 66
    done
    [[ -f "${dunhuang_root}/train/manifest.jsonl" && -f "${dunhuang_root}/validation/manifest.jsonl" ]] || exit 66
    [[ -f "${dunhuang_root}/test/manifest.jsonl" ]] || exit 66
    [[ -f "${parent_checkpoint_dir}/adapter.safetensors" && -f "${parent_checkpoint_dir}/adapter_config.json" ]] || exit 66
    [[ -f "${parent_checkpoint_dir}/decoder_lora.safetensors" ]] || exit 66
    [[ -f "${parent_run_dir}/COMPLETED" ]] || exit 66
    if (( resume_validation == 1 )); then
        # Re-entering after the evaluation stage failed: the training
        # artifacts must be the ones this launcher produced, and no
        # checkpoint may be scored yet, or the aggregator would read a
        # half-written directory.
        [[ -f "${run_dir}/COMPLETED" && -f "${run_dir}/summary.json" && -f "${run_dir}/metadata.json" ]] || exit 66
        local step
        for (( step = validation_interval; step <= steps; step += validation_interval )); do
            [[ ! -e "${validation_root}/step-${step}" ]] || exit 74
        done
        [[ ! -e "${validation_root}/step-0" ]] || exit 74
    else
        [[ ! -e "${remote_root}/training_runs/${run_id}" && ! -e "${remote_root}/training_runs/${run_id}_smoke" ]] || exit 74
        [[ ! -e "${status_file}" && ! -e "${summary_file}" ]] || exit 74
    fi
    command -v nvidia-smi >/dev/null 2>&1 || exit 69
    command -v jq >/dev/null 2>&1 || exit 69
    mkdir -p "${workspace_runs}" "${remote_root}/training_runs" "${remote_root}/protocols"
    admit_gpu_set "${gpu_ids}"
}

prepare_train_validation_protocol() {
    current_phase="prepare_train_validation_protocol"
    write_status running "${current_phase}" "${run_id}"
    "${python}" "${manifest_auditor}" \
        --train-manifest "${dunhuang_root}/train/manifest.jsonl" \
        --validation-manifest "${dunhuang_root}/validation/manifest.jsonl" \
        --num-queries "${num_queries}" --allow-count-mismatch --without-test \
        --dataset-label "dunhuang_local_gazetteer_q32_v1" \
        --protocol-label "glm_ocr_dunhuang_local_gazetteer_group_isolated_v1" \
        --output "${train_protocol}" \
        > "${workspace_runs}/${run_id}.train-protocol.log" 2>&1
}

# The wrapper owns the stage-two architecture and objective; this launcher only
# supplies the schedule, the resume point, and the validation cadence.
common_wrapper_args() {
    printf '%s\n' \
        --foreground --seed "${seed}" --gpu-ids "${gpu_ids}" \
        --gpu-utilization-limit "${gpu_utilization_limit}" --remote-root "${remote_root}" \
        --code-root "${code_root}" --env-dir "${env_dir}" --model-dir "${model_dir}" \
        --dataset-root "${dunhuang_root}" --protocol-file "${train_protocol}" \
        --dataset-label "dunhuang_local_gazetteer_q32_v1" \
        --protocol-label "glm_ocr_dunhuang_local_gazetteer_group_isolated_v1" \
        --allow-count-mismatch --num-queries "${num_queries}" \
        --box-head-mlp --query-refine-layers 1 \
        --learning-rate "${learning_rate}" \
        --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0 \
        --decoder-learning-rate "${decoder_learning_rate}" \
        --warmup-steps "${warmup_steps}" --min-lr-ratio "${min_lr_ratio}" \
        --init-checkpoint-override-residual-scale "${resume_gate}" \
        --max-eval-new-tokens "${max_eval_new_tokens}" \
        --max-pixels "${eval_max_pixels}" --log-steps 16
}

run_smoke() {
    current_phase="smoke"
    current_run_id="${run_id}_smoke"
    write_status running "${current_phase}" "${current_run_id}"
    mapfile -t args < <(common_wrapper_args)
    # The smoke horizon is 8 steps, so the shared 64-step warmup has to be
    # shortened here; the wrapper takes the last occurrence of a flag.
    args+=(--run-id "${current_run_id}" --max-steps 8 --lr-schedule-steps 8 \
        --warmup-steps 2 --validation-interval 9 --without-test --smoke \
        --init-checkpoint-dir "${parent_checkpoint_dir}")
    GLMOCR_SEM_ADAPTER_RUN_ID="${current_run_id}" \
    GLMOCR_SEM_ADAPTER_INIT_CHECKPOINT_DIR="${parent_checkpoint_dir}" \
        bash "${sem_adapter_wrapper}" "${args[@]}" \
        > "${workspace_runs}/${current_run_id}.pipeline.log" 2>&1
    "${python}" - "${remote_root}/training_runs/${current_run_id}/smoke/seed${seed}/smoke_summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
objective = ((summary.get("training") or {}).get("loss_objective") or {})
if summary.get("status") != "complete" or summary.get("checkpoint_reload") is not True:
    raise SystemExit("continuation smoke did not complete")
if summary.get("decoder_adaptation") != "lora":
    raise SystemExit("continuation smoke did not materialize decoder LoRA state")
if objective.get("layout_only") is not False or float(objective.get("layout_weight", -1)) != 0.0:
    raise SystemExit(f"unexpected stage-two objective: {objective}")
print(json.dumps({"event": "glmocr_dh_continue_smoke_ok"}, separators=(",", ":")))
PY
}

run_training() {
    current_phase="training"
    current_run_id="${run_id}"
    write_status running "${current_phase}" "${current_run_id}"
    mapfile -t args < <(common_wrapper_args)
    # --defer-validation saves the 64/128/192/256 checkpoints but skips the
    # trainer's own rank-0-only evaluation loop; run_parallel_validation scores
    # them afterwards.  The step-0 identity control is likewise an eval-only
    # process there, not a --diagnostic-steps entry, which would otherwise stall
    # the other four ranks at a barrier for the length of a full validation pass.
    args+=(--run-id "${run_id}" --max-steps "${steps}" --lr-schedule-steps "${steps}" \
        --validation-interval "${validation_interval}" --without-test \
        --defer-validation --init-checkpoint-dir "${parent_checkpoint_dir}")
    GLMOCR_SEM_ADAPTER_RUN_ID="${run_id}" \
    GLMOCR_SEM_ADAPTER_INIT_CHECKPOINT_DIR="${parent_checkpoint_dir}" \
        bash "${sem_adapter_wrapper}" "${args[@]}" \
        > "${workspace_runs}/${run_id}.pipeline.log" 2>&1
    "${python}" - "${run_dir}" "${steps}" "${validation_interval}" <<'PY'
import json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
steps = int(sys.argv[2])
interval = int(sys.argv[3])
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
expected_steps = list(range(interval, steps + 1, interval))
if summary.get("status") != "complete" or metadata.get("status") != "complete":
    raise SystemExit("continuation training incomplete")
for payload, label in ((summary, "summary"), (metadata, "metadata")):
    if payload.get("test_manifest_read") is not False:
        raise SystemExit(f"continuation training {label} read the test split")
if summary.get("defer_validation") is not True or summary.get("selection_pending") is not True:
    raise SystemExit("continuation training did not defer validation as requested")
training = summary.get("training") or {}
if [int(step) for step in training.get("checkpoint_steps", [])] != expected_steps:
    raise SystemExit(f"checkpoint steps {training.get('checkpoint_steps')} != {expected_steps}")
gate = (metadata.get("content_gate_after_init") or {}).get("effective_residual_scale")
if gate is None or not 0.012 < float(gate) < 0.014:
    raise SystemExit(f"continuation did not resume the parent gate: {gate}")
if summary.get("validation_candidates"):
    raise SystemExit("deferred training unexpectedly produced validation candidates")
print(json.dumps({
    "event": "glmocr_dh_continue_training_ok",
    "checkpoint_steps": expected_steps,
    "resumed_effective_residual_scale": float(gate),
    "final_raw_content_gate": training.get("final_raw_content_gate"),
}, separators=(",", ":")))
PY
}

setup_eval_environment() {
    # Standalone single-process evaluation bypasses the DDP wrapper, so the
    # CUDA/HF environment it would otherwise set has to be reproduced here.
    local cuda_libraries="${env_dir}/lib/python3.11/site-packages/torch/lib"
    local cuda_arch
    cuda_arch="$(uname -m)"
    [[ "${cuda_arch}" != "arm64" ]] || cuda_arch="aarch64"
    local system_cuda="/usr/local/cuda/targets/${cuda_arch}-linux/lib"
    [[ ! -d "${system_cuda}" ]] || cuda_libraries="${system_cuda}:${cuda_libraries}"
    local component lib
    for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
        lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
        [[ ! -d "${lib}" ]] || cuda_libraries="${cuda_libraries}:${lib}"
    done
    export LD_LIBRARY_PATH="${cuda_libraries}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
}

# Score one checkpoint on the validation split.  The flags deliberately repeat
# the wrapper's metadata-relevant choices: the aggregator compares the eval
# metadata against the training metadata and refuses any divergence.  The
# checkpoint's own content gate must load as saved, so no residual-scale
# override is passed here.  --freeze-layout-branch is deliberately absent:
# it only sets requires_grad, and train_screen rejects it under --eval-only.
launch_eval() {
    local gpu="$1" checkpoint="$2" out="$3" label="$4"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export TMPDIR="${group_root}/tmp/validation-${label}"
        export HF_HOME="${TMPDIR}/huggingface"
        export TRANSFORMERS_CACHE="${HF_HOME}"
        mkdir -p "${TMPDIR}" "${HF_HOME}"
        cd "${code_root}"
        exec "${python}" -m layout_ocr.train_screen \
            --mode layout_ot --model-path "${model_dir}" \
            --train-manifest "${dunhuang_root}/train/manifest.jsonl" \
            --validation-manifest "${dunhuang_root}/validation/manifest.jsonl" \
            --protocol-file "${train_protocol}" \
            --output-dir "${out}" \
            --per-device-batch-size 1 --gradient-accumulation-steps 1 \
            --max-steps 1 --lr-schedule-steps 1 --warmup-steps 0 --min-lr-ratio 0.1 \
            --num-queries "${num_queries}" --seed "${seed}" \
            --experiment-label "glmocr_dh_continue_validation_${label}" \
            --learning-rate "${learning_rate}" \
            --decoder-adaptation lora --decoder-lora-rank 8 --decoder-lora-alpha 8 \
            --decoder-lora-dropout 0 --decoder-learning-rate "${decoder_learning_rate}" \
            --residual-scale-cap 0.03 --initial-residual-scale 0 \
            --auxiliary-weight 0 --auxiliary-weight-start 0 --auxiliary-ramp-steps 0 \
            --gate-freeze-steps 0 --max-grad-norm 1.0 \
            --layout-loss-profile ocr_only --query-assignment hungarian \
            --processor-mode fast --adapter-precision fp32 \
            --box-head-mlp --query-refine-layers 1 --sem-adapter-mlp \
            --max-pixels "${eval_max_pixels}" --max-eval-new-tokens "${max_eval_new_tokens}" \
            --generation-mode "${eval_generation_mode}" \
            --validation-interval "${validation_interval}" --log-steps 16 \
            --eval-checkpoint-dir "${checkpoint}" --eval-only
    ) > "${group_root}/logs/validation-${label}.log" 2>&1 &
}

run_parallel_validation() {
    current_phase="validation"
    current_run_id="${run_id}"
    write_status running "${current_phase}" "${current_run_id}"
    local -a candidate_steps=()
    local step
    for (( step = validation_interval; step <= steps; step += validation_interval )); do
        candidate_steps+=("${step}")
    done
    IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
    (( ${#gpu_array[@]} >= ${#candidate_steps[@]} + 1 )) || exit 64

    mkdir -p "${validation_root}" "${group_root}/logs" "${group_root}/tmp"
    setup_eval_environment

    local -a pids=() labels=()
    local index=0
    for step in "${candidate_steps[@]}"; do
        launch_eval "${gpu_array[${index}]}" "${run_dir}/checkpoint-${step}" \
            "${validation_root}/step-${step}" "step${step}"
        pids+=("$!")
        labels+=("step-${step}")
        index=$(( index + 1 ))
    done
    # Step 0 is the identity control: the untouched resume checkpoint, scored on
    # the same split through the same eval path, so the selection curve has a
    # real starting point.  It is a reference value, never a selection candidate.
    launch_eval "${gpu_array[${index}]}" "${parent_checkpoint_dir}" \
        "${validation_root}/step-0" "step0-identity"
    pids+=("$!")
    labels+=("step-0")

    local failed=0 i
    for i in "${!pids[@]}"; do
        if ! wait "${pids[${i}]}"; then
            printf '{"event":"glmocr_dh_continue_failed","error":"validation_process_failed","label":"%s"}\n' \
                "${labels[${i}]}" >&2
            failed=1
        fi
    done
    (( failed == 0 )) || exit 1

    "${python}" "${parallel_summarizer}" \
        --run-dir "${run_dir}" --validation-root "${validation_root}" \
        --group-root "${group_root}" \
        --steps "$(IFS=,; echo "${candidate_steps[*]}")" \
        --expected-world-size 5 --expected-layout-weight 0.0 \
        --dataset-label "dunhuang_local_gazetteer_q32_v1" \
        > "${workspace_runs}/${run_id}.parallel-validation.log" 2>&1

    "${python}" - "${run_dir}" "${validation_root}" "${candidate_steps[@]}" <<'PY'
import json, sys
from pathlib import Path
run_dir, validation_root = Path(sys.argv[1]), Path(sys.argv[2])
expected_steps = [int(step) for step in sys.argv[3:]]
selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
parallel = json.loads((run_dir / "parallel_validation_summary.json").read_text(encoding="utf-8"))
if selection.get("status") != "complete" or parallel.get("status") != "complete":
    raise SystemExit("parallel validation did not complete")
for payload, label in ((selection, "selection"), (parallel, "parallel summary")):
    if payload.get("test_used_for_selection") is not False or payload.get("test_manifest_read") is not False:
        raise SystemExit(f"{label} is not test-free")
if selection.get("selection_metric") != "validation_cer":
    raise SystemExit(f"selection metric is not validation CER: {selection.get('selection_metric')}")
selected = selection.get("selected_step")
if selected not in expected_steps:
    raise SystemExit(f"selected step {selected} is outside {expected_steps}")
candidates = selection.get("candidates") or []
if [row.get("step") for row in candidates] != expected_steps:
    raise SystemExit(f"candidate steps {[row.get('step') for row in candidates]} != {expected_steps}")
cer = {}
for row in candidates:
    value = row.get("cer")
    if value is None:
        raise SystemExit(f"candidate {row.get('step')} is missing its CER")
    cer[str(row["step"])] = float(value)
identity_summary = json.loads((validation_root / "step-0" / "summary.json").read_text(encoding="utf-8"))
identity_cer = (identity_summary.get("validation") or {}).get("cer")
if identity_cer is None:
    raise SystemExit("identity control did not report a CER")
print(json.dumps({
    "event": "glmocr_dh_continue_validation_ok",
    "checkpoint_steps": expected_steps,
    "candidate_cer": cer,
    "identity_baseline_cer": float(identity_cer),
    "selected_step": selected,
    "selected_validation_cer": float(cer[str(selected)]),
}, separators=(",", ":")))
PY
}

run_test() {
    current_phase="test"
    current_run_id="${run_id}"
    write_status running "${current_phase}" "${current_run_id}"
    "${python}" "${manifest_auditor}" \
        --train-manifest "${dunhuang_root}/train/manifest.jsonl" \
        --validation-manifest "${dunhuang_root}/validation/manifest.jsonl" \
        --test-manifest "${dunhuang_root}/test/manifest.jsonl" \
        --num-queries "${num_queries}" --allow-count-mismatch \
        --dataset-label "dunhuang_local_gazetteer_q32_v1" \
        --protocol-label "glm_ocr_dunhuang_local_gazetteer_group_isolated_v1" \
        --output "${test_protocol}" \
        > "${workspace_runs}/${run_id}.test-protocol.log" 2>&1
    bash "${locked_test_launcher}" --foreground --run-id "${run_id}" --seed "${seed}" \
        --gpu-ids "${gpu_ids}" --gpu-utilization-limit "${gpu_utilization_limit}" \
        --mode layout_ot --num-queries "${num_queries}" --max-eval-new-tokens "${max_eval_new_tokens}" \
        --max-pixels "${eval_max_pixels}" \
        --remote-root "${remote_root}" --code-root "${code_root}" --env-dir "${env_dir}" \
        --model-dir "${model_dir}" --dataset-root "${dunhuang_root}" \
        --protocol-file "${test_protocol}" \
        > "${workspace_runs}/${run_id}.locked-test.pipeline.log" 2>&1
    "${python}" - "${run_dir}/locked-test/locked_test_summary.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
metrics = payload.get("metrics") or {}
if payload.get("test_used_for_selection") is not False:
    raise SystemExit("locked test reported selection on the test split")
print(json.dumps({
    "event": "glmocr_dh_continue_test_complete",
    "selected_step": payload.get("selected_step"),
    "test_pages": payload.get("test_pages"),
    "cer": metrics.get("cer"),
    "layout_box_iou": metrics.get("layout_box_iou"),
    "layout_box_mae": metrics.get("layout_box_mae"),
    "generation_limit_hits": metrics.get("generation_limit_hits"),
}, separators=(",", ":")))
PY
}

write_final_summary() {
    current_phase="complete"
    write_status complete complete "${run_id}"
    "${python}" - "${summary_file}" "${run_id}" "${parent_run_id}" "${run_dir}" "${validation_root}" "${seed}" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
run_id, parent_run_id, seed = sys.argv[2], sys.argv[3], sys.argv[6]
run_dir, validation_root = Path(sys.argv[4]), Path(sys.argv[5])
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
parallel = json.loads((run_dir / "parallel_validation_summary.json").read_text(encoding="utf-8"))
identity = json.loads((validation_root / "step-0" / "summary.json").read_text(encoding="utf-8"))
test = json.loads((run_dir / "locked-test" / "locked_test_summary.json").read_text(encoding="utf-8"))
payload = {
    "status": "complete",
    "session": out.stem.replace(".summary", ""),
    "run_id": run_id,
    "parent_run_id": parent_run_id,
    "seed": int(seed),
    "num_queries": 32,
    "steps": (summary.get("training") or {}).get("steps"),
    "validation_interval": (summary.get("training") or {}).get("validation_interval"),
    "selection_metric": selection.get("selection_metric"),
    "selected_step": selection.get("selected_step"),
    "selection_performed": selection.get("selection_performed"),
    "identity_baseline": {
        "step": 0,
        "source": "parent resume checkpoint",
        "cer": (identity.get("validation") or {}).get("cer"),
        "selection_candidate": False,
    },
    "validation_candidates": [
        {"step": row.get("step"), "cer": row.get("cer")}
        for row in (parallel.get("candidates") or [])
    ],
    "test_used_for_selection": test.get("test_used_for_selection"),
    "test": {
        "summary": str(run_dir / "locked-test" / "locked_test_summary.json"),
        "selected_step": test.get("selected_step"),
        "test_pages": test.get("test_pages"),
        "metrics": test.get("metrics") or {},
    },
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({
    "event": "glmocr_dh_continue_pipeline_complete",
    "summary": str(out),
    "selected_step": payload["selected_step"],
    "cer": payload["test"]["metrics"].get("cer"),
}, separators=(",", ":")))
PY
    cat "${summary_file}"
}

run_inner() {
    preflight
    if (( resume_validation == 0 )); then
        prepare_train_validation_protocol
        run_smoke
        run_training
    fi
    run_parallel_validation
    run_test
    write_final_summary
}

if (( foreground == 0 )); then
    command -v tmux >/dev/null 2>&1 || exit 69
    tmux has-session -t "${session}" 2>/dev/null && exit 73
    preflight
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    mkdir -p "${workspace_runs}"
    command_line="$(printf '%q ' bash "${script_path}" --foreground)"
    # tmux attaches new sessions to an existing server, whose environment is
    # the one from when that server started.  Anything configured through
    # GLMOCR_DH_CONTINUE_* therefore has to be restated on the inner command
    # line; otherwise the inner run silently falls back to the defaults,
    # collides with an existing run directory and exits 74 with an empty log.
    env_prefix=""
    env_prefix+="export GLMOCR_DH_CONTINUE_RUN_ID=$(printf '%q' "${run_id}"); "
    env_prefix+="export GLMOCR_DH_CONTINUE_SESSION=$(printf '%q' "${session}"); "
    env_prefix+="export GLMOCR_DH_CONTINUE_MAX_PIXELS=$(printf '%q' "${eval_max_pixels}"); "
    env_prefix+="export GLMOCR_DH_CONTINUE_RESUME_GATE=$(printf '%q' "${resume_gate}"); "
    env_prefix+="export GLMOCR_DH_CONTINUE_PARENT_RUN_ID=$(printf '%q' "${parent_run_id}"); "
    env_prefix+="export GLMOCR_DH_CONTINUE_PARENT_CHECKPOINT=$(printf '%q' "${parent_checkpoint_dir}"); "
    env_prefix+="export GLMOCR_DH_CONTINUE_GPU_IDS=$(printf '%q' "${gpu_ids}"); "
    # The inner invocation re-runs preflight, so it has to know whether it
    # is resuming; without this it takes the fresh-run branch and exits 74.
    (( resume_validation == 0 )) || command_line+="--resume-validation "
    tmux new-session -d -s "${session}" \
        "cd $(printf '%q' "${code_root}") && ${env_prefix}exec ${command_line} >$(printf '%q' "${workspace_runs}/${session}.log") 2>&1"
    printf '{"event":"glmocr_dh_continue_pipeline_armed","session":"%s","run_id":"%s","parent_run_id":"%s","gpu_ids":"%s","log":"%s"}\n' \
        "${session}" "${run_id}" "${parent_run_id}" "${gpu_ids}" "${workspace_runs}/${session}.log"
else
    run_inner
fi
