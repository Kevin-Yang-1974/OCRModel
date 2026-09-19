#!/usr/bin/env bash
# 带布局分支的微调 vs 不带布局分支的微调（受控对比）。
#
# 问题：此前所有 with/without 对比都是跨谱系的（content_only run 的起点是
# ref600/MTHv2 warmstart，不是同一起点），因此从未在受控条件下回答过
# 「布局分支对微调有没有用」。本脚本用同一起点、同一数据、同一步数、同一学习率、
# 同一分辨率，只改布局分支是否参与，来回答它。
#
# 三臂，起点同为 stage-1 布局 checkpoint（IoU 0.672）：
#   N   content_only  布局分支旁路（merged = visual_tokens），只训 decoder LoRA
#   W   layout_ot     分支参与，gate 热启动 0.01，cap 0.03（现有配置）
#   Wx  layout_ot     分支参与，gate 起点 0.03（= cap），有效残差约为 W 的 2.3 倍
#
# 为什么要有 Wx：残差实测只占视觉特征范数的约 2.3%，而训练全程 gate 只到 0.0155、
# 从未触及 0.03 上限。也就是说 0.03 这个 cap 是把布局分支的影响掐住的节流阀，而不是
# 实践中达到的安全边界。若布局分支携带可用信息，放宽 cap 后 with/without 的差距应当
# 变大；若放宽后仍无差异，说明瓶颈在耦合方式（几何/阅读顺序没有任何通往解码器的
# 路径）而不在强度——那是设计问题，调参补不了。
#
# 判定：对每条臂的 80 页 validation 逐页配对 bootstrap，看 W−N 与 Wx−N 是否显著。
set -euo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dunhuang_root="${GLMOCR_Q32_DATASET_ROOT:-/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"

stage1_run="glmocr_mthv2_sparse24_q32_layout_iou_consistent_giou20x_mlp_refine_10000_continue_from_giou10x10000_20260918_v2"
start_checkpoint="${GLMOCR_LAYOUT_CMP_START:-${remote_root}/training_runs/${stage1_run}/seed42/checkpoint-10000}"

run_id="${GLMOCR_LAYOUT_CMP_RUN_ID:-glmocr_dunhuang_layout_branch_finetune_compare_20260919_v1}"
session="${GLMOCR_LAYOUT_CMP_SESSION:-glmocr_layout_cmp_20260919_v1}"
seed=42
num_queries=32
steps="${GLMOCR_LAYOUT_CMP_STEPS:-256}"
validation_interval=64
learning_rate=1e-5
decoder_learning_rate=1e-6
warmup_steps=64
min_lr_ratio=0.1
warm_start_gate=0.01
max_eval_new_tokens=1536
eval_generation_mode="loop_recovery"
max_pixels="${GLMOCR_LAYOUT_CMP_MAX_PIXELS:-4000000}"
gpu_ids="${GLMOCR_LAYOUT_CMP_GPU_IDS:-0,1,2,3,4}"
gpu_utilization_limit="${GLMOCR_LAYOUT_CMP_GPU_UTILIZATION_LIMIT:-50}"
foreground="${GLMOCR_LAYOUT_CMP_FOREGROUND:-0}"

manifest_auditor="${code_root}/tools/audit_mthv2_manifest.py"
ddp_launcher="${code_root}/tools/training/run_glmocr_mthv2_ddp.sh"
python="${env_dir}/bin/python3"
script_path="$(readlink -f "${BASH_SOURCE[0]}")"

setup_environment() {
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

preflight() {
    [[ -x "${python}" ]] || { echo "python missing: ${python}" >&2; exit 64; }
    [[ -f "${start_checkpoint}/adapter.safetensors" ]] || { echo "stage-1 checkpoint missing" >&2; exit 64; }
    [[ -f "${ddp_launcher}" ]] || { echo "ddp launcher missing" >&2; exit 64; }
    # main() creates eval_root and group_root before the tmux session starts, and
    # train_arm resumes any arm that already has a summary, so the only real
    # collision is a completed evaluation.
    if [[ -e "${eval_root}/arms" ]]; then
        echo "evaluation artefacts already exist under ${eval_root}/arms" >&2
        exit 74
    fi
}

prepare_protocol() {
    setup_environment
    "${python}" "${manifest_auditor}" \
        --train-manifest "${dunhuang_root}/train/manifest.jsonl" \
        --validation-manifest "${dunhuang_root}/validation/manifest.jsonl" \
        --num-queries "${num_queries}" --allow-count-mismatch --without-test \
        --dataset-label "dunhuang_local_gazetteer_q32_v1" \
        --protocol-label "glm_ocr_dunhuang_local_gazetteer_group_isolated_v1" \
        --output "${train_protocol}" \
        > "${eval_root}/train-protocol.log" 2>&1
}

# train_arm <arm> <mode> <cap>
train_arm() {
    local arm="$1" mode="$2" cap="$3" gate_start="$4"
    local arm_run_id="${run_id}_${arm}"
    local arm_summary="${runs_root}/${arm_run_id}/seed${seed}/summary.json"
    current_arm="${arm}"
    if [[ -f "${arm_summary}" ]]; then
        # Resumable: a completed arm is verified rather than retrained.
        echo "{\"event\":\"glmocr_layout_cmp_arm_reused\",\"arm\":\"${arm}\"}"
        verify_arm_summary "${arm_summary}" "${arm}"
        return
    fi
    write_status running "training_${arm}" "${arm_run_id}"
    bash "${ddp_launcher}" --foreground --seed "${seed}" --gpu-ids "${gpu_ids}" \
        --gpu-utilization-limit "${gpu_utilization_limit}" \
        --remote-root "${remote_root}" --code-root "${code_root}" --env-dir "${env_dir}" \
        --model-dir "${model_dir}" --dataset-root "${dunhuang_root}" \
        --protocol-file "${train_protocol}" \
        --dataset-label "dunhuang_local_gazetteer_q32_v1" \
        --protocol-label "glm_ocr_dunhuang_local_gazetteer_group_isolated_v1" \
        --allow-count-mismatch --num-queries "${num_queries}" \
        --box-head-mlp --query-refine-layers 1 \
        --mode "${mode}" --sem-adapter-mlp --freeze-layout-branch \
        --layout-loss-profile ocr_only \
        --auxiliary-weight 0 --auxiliary-weight-start 0 --auxiliary-ramp-steps 0 \
        --residual-scale-cap "${cap}" \
        --init-checkpoint-override-residual-scale "${gate_start}" \
        --init-checkpoint-dir "${start_checkpoint}" \
        --init-checkpoint-allow-mode-mismatch \
        --learning-rate "${learning_rate}" \
        --decoder-adaptation lora \
        --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0 \
        --decoder-learning-rate "${decoder_learning_rate}" \
        --warmup-steps "${warmup_steps}" --min-lr-ratio "${min_lr_ratio}" \
        --max-steps "${steps}" --lr-schedule-steps "${steps}" \
        --validation-interval "${validation_interval}" \
        --defer-validation --without-test \
        --max-pixels "${max_pixels}" --max-eval-new-tokens "${max_eval_new_tokens}" \
        --log-steps 16 --run-id "${arm_run_id}" \
        > "${eval_root}/${arm}.train.log" 2>&1
    verify_arm_summary "${arm_summary}" "${arm}"
}

verify_arm_summary() {
    "${python}" - "$1" "$2" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
training = payload.get("training") or {}
if payload.get("status") != "complete":
    raise SystemExit(f"arm {sys.argv[2]} training did not complete")
if payload.get("test_manifest_read") is not False:
    raise SystemExit(f"arm {sys.argv[2]} read the test manifest")
print(json.dumps({
    "event": "glmocr_layout_cmp_arm_trained",
    "arm": sys.argv[2],
    "mode": payload.get("mode"),
    "gate": training.get("final_raw_content_gate"),
    "checkpoints": training.get("checkpoint_steps"),
}, separators=(",", ":")))
PY
}

# launch_eval <gpu> <arm-label> <checkpoint> <mode> <out_dir>
launch_eval() {
    local gpu="$1" label="$2" checkpoint="$3" mode="$4" out="$5"
    (
        setup_environment
        export CUDA_VISIBLE_DEVICES="${gpu}"
        cd "${code_root}"
        local extra=()
        if [[ "${mode}" == "layout_ot" ]]; then
            extra=(--decoder-adaptation lora --decoder-lora-rank 8 --decoder-lora-alpha 8 \
                   --decoder-lora-dropout 0 --decoder-learning-rate "${decoder_learning_rate}" \
                   --box-head-mlp --query-refine-layers 1 --sem-adapter-mlp \
                   --eval-checkpoint-dir "${checkpoint}")
        else
            extra=(--decoder-adaptation frozen)
        fi
        exec "${python}" -m layout_ocr.train_screen \
            --mode "${mode}" --model-path "${model_dir}" \
            --train-manifest "${dunhuang_root}/train/manifest.jsonl" \
            --validation-manifest "${dunhuang_root}/validation/manifest.jsonl" \
            --protocol-file "${train_protocol}" \
            --output-dir "${out}" \
            --per-device-batch-size 1 --gradient-accumulation-steps 1 \
            --max-steps 1 --lr-schedule-steps 1 --warmup-steps 0 --min-lr-ratio 0.1 \
            --num-queries "${num_queries}" --seed "${seed}" \
            --experiment-label "glmocr_layout_cmp_eval_${label}" \
            --learning-rate "${learning_rate}" --max-grad-norm 1.0 \
            --residual-scale-cap 0.03 --initial-residual-scale 0 \
            --auxiliary-weight 0 --auxiliary-weight-start 0 --auxiliary-ramp-steps 0 \
            --layout-loss-profile ocr_only --query-assignment hungarian \
            --processor-mode fast --adapter-precision fp32 \
            --max-pixels "${max_pixels}" --max-eval-new-tokens "${max_eval_new_tokens}" \
            --generation-mode "${eval_generation_mode}" \
            --validation-interval 1 --log-steps 16 \
            "${extra[@]}" --eval-only
    ) > "${out}.log" 2>&1
}

run_evaluation() {
    current_arm="evaluation"
    write_status running "evaluation" "${run_id}"
    local arms="${eval_root}/arms"
    mkdir -p "${arms}"
    # Each arm is evaluated in the mode it was trained in: "fine-tune and deploy
    # without the branch" versus "fine-tune and deploy with it".  The stage-1
    # start checkpoint is scored too, to show where fine-tuning began.
    local pids=()
    launch_eval 0 "N-nobranch" "${runs_root}/${run_id}_N/seed${seed}/checkpoint-${steps}" content_only "${arms}/N-nobranch" & pids+=("$!")
    launch_eval 1 "W-branch-cap03" "${runs_root}/${run_id}_W/seed${seed}/checkpoint-${steps}" layout_ot "${arms}/W-branch-cap03" & pids+=("$!")
    launch_eval 2 "Wx-branch-cap30" "${runs_root}/${run_id}_Wx/seed${seed}/checkpoint-${steps}" layout_ot "${arms}/Wx-branch-cap30" & pids+=("$!")
    launch_eval 3 "S-start-stage1" "${start_checkpoint}" layout_ot "${arms}/S-start-stage1" & pids+=("$!")

    local failed=0 pid
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || {
        echo "{\"event\":\"glmocr_layout_cmp_failed\",\"error\":\"eval_failed\"}" >&2
        exit 1
    }
}

summarize() {
    setup_environment
    "${python}" - "${eval_root}" "${train_protocol}" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
protocol = Path(sys.argv[2])
arms = root / "arms"
labels = {
    "S-start-stage1": "stage-1 start (before this fine-tune)",
    "N-nobranch": "fine-tuned WITHOUT the layout branch (content_only)",
    "W-branch-cap03": "fine-tuned WITH the branch, cap 0.03",
    "Wx-branch-cap30": "fine-tuned WITH the branch pinned at the cap (gate start 0.03, effective scale 0.03 via clamp)",
}
payload = {"status": "complete", "protocol": str(protocol), "arms": {}}
for name, label in labels.items():
    summary = json.loads((arms / name / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["validation"]
    payload["arms"][name] = {
        "label": label,
        "cer": metrics["cer"],
        "pages": metrics["pages"],
        "insertions": metrics["insertions"],
        "deletions": metrics["deletions"],
        "substitutions": metrics["substitutions"],
        "generation_limit_hits": metrics["generation_limit_hits"],
        "mode": summary["mode"],
        "effective_residual_scale": metrics.get("effective_residual_scale"),
        "residual_relative_norm": metrics.get("residual_relative_norm"),
        "predictions": str(arms / name / "validation_predictions.jsonl"),
    }
(root / "comparison_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
for name in ("S-start-stage1", "N-nobranch", "W-branch-cap03", "Wx-branch-cap30"):
    row = payload["arms"][name]
    print(f"{name:18s} CER {row['cer']:.6f}  mode={row['mode']:12s} res_norm={row['residual_relative_norm']}")
PY
}

write_status() {
    local status="$1" phase="$2" rid="$3"
    mkdir -p "${remote_root}/runs"
    printf '{"status":"%s","phase":"%s","run_id":"%s","updated_at":"%s"}\n' \
        "${status}" "${phase}" "${rid}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "${remote_root}/runs/${session}.status.json"
}

main() {
    preflight
    setup_environment
    mkdir -p "${eval_root}" "${group_root}"
    write_status running "protocol" "${run_id}"
    prepare_protocol
    train_arm N  content_only 0.03 0.01
    train_arm W  layout_ot    0.03 0.01
    train_arm Wx layout_ot    0.03 0.03
    run_evaluation
    summarize
    write_status complete complete "${run_id}"
    echo "{\"event\":\"glmocr_layout_cmp_complete\",\"run_id\":\"${run_id}\"}"
}

while (( $# > 0 )); do
    case "$1" in
        --foreground) foreground=1; shift ;;
        --run-id) run_id="$2"; shift 2 ;;
        --steps) steps="$2"; shift 2 ;;
        --max-pixels) max_pixels="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

# Derived paths must be computed after --run-id is parsed: the outer process
# builds the tmux command from them, and the inner process re-derives them
# from the environment, so any divergence splits the logs and artefacts.
train_protocol="${remote_root}/protocols/${run_id}.train_validation_no_test.json"
runs_root="${remote_root}/training_runs"
group_root="${remote_root}/training_runs/${run_id}"
eval_root="${remote_root}/layout_branch_finetune_compare/${run_id}"
session="${GLMOCR_LAYOUT_CMP_SESSION:-${run_id}}"

if (( foreground == 1 )); then
    main
    exit 0
fi

mkdir -p "${eval_root}"
tmux kill-session -t "${session}" 2>/dev/null || true
tmux new-session -d -s "${session}" \
    "export GLMOCR_LAYOUT_CMP_RUN_ID=$(printf '%q' "${run_id}"); ""\
bash $(printf '%q' "${script_path}") --foreground > ${eval_root}/pipeline.log 2>&1"
echo "{\"event\":\"glmocr_layout_cmp_armed\",\"session\":\"${session}\",\"root\":\"${eval_root}\"}"
