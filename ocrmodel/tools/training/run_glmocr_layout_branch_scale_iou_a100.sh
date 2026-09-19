#!/usr/bin/env bash
# 两个后续实验共用启动器：
#
# 阶段 A（IoU 阶梯）：把不同质量的布局分支接到同一套微调配方上，看最终文字识别指标
#   是否随布局 IoU 单调。四臂只差 19 个 query_* 张量（查询/传输机制 = 布局信息本身），
#   其余（sem_adapter / content_norm / content_gate / 各头）与起点完全相同：
#     Q-rand 随机初始化（无布局信息，IoU ≈ 0 的实际下限）
#     Q-042  giou10x checkpoint-1000   IoU 0.4225
#     Q-065  giou10x checkpoint-10000  IoU 0.6480
#     Q-067  giou20x checkpoint-10000  IoU 0.6721 —— 与对比实验的 W 臂配方完全一致，
#            直接复用其结果，不重训
#
# 阶段 B（取消上限长训）：max_residual_scale 烧在 stage-1 checkpoint 的
#   adapter_config.json 里，且加载时既校验 cap 又限制「起点不得超过 cap」，所以取消
#   上限必须做 checkpoint 副本（config 里把 max_residual_scale 置 null）+ 传
#   --residual-scale-cap none。decoder 冻结（该谱系下 LoRA 在 1e-6 下无效果，实测
#   与基座逐位相同），只训 sem_adapter 与 content_gate，1024 步、每 256 步验证，
#   观察 gate 是否继续上升以及 CER 是否随之改善。
set -euo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dunhuang_root="${GLMOCR_Q32_DATASET_ROOT:-/data3/yky/yangky_ocr_models/datasets/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"

runs_root="${remote_root}/training_runs"
stage1_run="glmocr_mthv2_sparse24_q32_layout_iou_consistent_giou20x_mlp_refine_10000_continue_from_giou10x10000_20260918_v2"
stage1_checkpoint="${runs_root}/${stage1_run}/seed42/checkpoint-10000"
donor_low="${runs_root}/glmocr_mthv2_sparse24_q32_layout_iou_consistent_giou10x_mlp_refine_3000_from_boxeq820_20260917_v1/seed42/checkpoint-1000"
donor_mid="${runs_root}/glmocr_mthv2_sparse24_q32_layout_iou_consistent_giou10x_mlp_refine_10000_continue_from_giou10x3000_20260917_v2/seed42/checkpoint-10000"
compare_run_id="glmocr_dunhuang_layout_branch_finetune_compare_20260919_v1"

run_id="${GLMOCR_SCALE_IOU_RUN_ID:-glmocr_dunhuang_layout_branch_scale_iou_20260919_v1}"
session="${GLMOCR_SCALE_IOU_SESSION:-${run_id}}"
seed=42
num_queries=32
max_eval_new_tokens=1536
eval_generation_mode="loop_recovery"
max_pixels="${GLMOCR_SCALE_IOU_MAX_PIXELS:-4000000}"
learning_rate=1e-5
decoder_learning_rate=1e-6
warmup_steps=64
min_lr_ratio=0.1
cap=0.03
gate_start=0.01
# The uncapped arm starts much higher on purpose: with the decoder frozen the
# fusion branch carries the whole adaptation, and the hypothesis under test is
# that it can absorb that role without the repetition failure mode that
# teacher-forcing the decoder produced.
u_gate_start="${GLMOCR_SCALE_IOU_U_GATE_START:-0.10}"
long_steps="${GLMOCR_SCALE_IOU_LONG_STEPS:-1024}"
gpu_ids="${GLMOCR_SCALE_IOU_GPU_IDS:-0,1,2,3,4}"
gpu_utilization_limit="${GLMOCR_SCALE_IOU_GPU_UTILIZATION_LIMIT:-50}"
foreground="${GLMOCR_SCALE_IOU_FOREGROUND:-0}"
checkpoints_only=0

train_protocol="${remote_root}/protocols/${run_id}.train_validation_no_test.json"
root="${remote_root}/layout_branch_scale_iou/${run_id}"
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
    [[ -f "${stage1_checkpoint}/adapter.safetensors" ]] || { echo "stage-1 checkpoint missing" >&2; exit 64; }
    [[ -f "${donor_low}/adapter.safetensors" ]] || { echo "low-IoU donor missing" >&2; exit 64; }
    [[ -f "${donor_mid}/adapter.safetensors" ]] || { echo "mid-IoU donor missing" >&2; exit 64; }
    [[ -e "${root}/arms" ]] && { echo "evaluation already exists: ${root}/arms" >&2; exit 74; }
    return 0
}

prepare_protocol() {
    setup_environment
    "${python}" "${manifest_auditor}" \
        --train-manifest "${dunhuang_root}/train/manifest.jsonl" \
        --validation-manifest "${dunhuang_root}/validation/manifest.jsonl" \
        --num-queries "${num_queries}" --allow-count-mismatch --without-test \
        --dataset-label "dunhuang_local_gazetteer_q32_v1" \
        --protocol-label "glm_ocr_dunhuang_local_gazetteer_group_isolated_v1" \
        --output "${train_protocol}" > "${root}/train-protocol.log" 2>&1
}

# build_checkpoint <target> <query_source> <uncap>
#   query_source: keep | random | <path to another adapter.safetensors>
#   uncap: 1 sets max_residual_scale to null so the gate is unbounded
build_checkpoint() {
    local target="$1" query_source="$2" uncap="$3"
    # Idempotent: a checkpoint built by an earlier invocation is kept,
    # which also allows the surgery to be done outside this launcher.
    if [[ -f "$1/adapter_config.json" ]]; then
        echo "{\"event\":\"glmocr_scale_iou_checkpoint_reused\",\"target\":\"$1\"}"
        return 0
    fi
    setup_environment
    mkdir -p "${target}"
    cp -r "${stage1_checkpoint}/." "${target}/"
    "${python}" - "${target}" "${query_source}" "${uncap}" <<'PY'
import json, math, sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

target, query_source, uncap = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
adapter_path = target / "adapter.safetensors"
state = load_file(adapter_path, device="cpu")
QUERY_PREFIXES = ("query_seed", "query_attention.", "query_norm.", "query_refine.")
query_keys = [k for k in state if k.startswith(QUERY_PREFIXES)]
if len(query_keys) != 19:
    raise SystemExit(f"expected 19 query tensors, found {len(query_keys)}")

if query_source == "random":
    generator = torch.Generator().manual_seed(20260919)
    for key in query_keys:
        tensor = state[key]
        noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
        state[key] = (noise * tensor.float().std() + tensor.float().mean()).to(tensor.dtype)
elif query_source != "keep":
    donor = load_file(query_source, device="cpu")
    missing = [k for k in query_keys if k not in donor]
    mismatch = [k for k in query_keys if k in donor and donor[k].shape != state[k].shape]
    if missing or mismatch:
        raise SystemExit(f"donor incompatible: missing={missing[:3]} mismatch={mismatch[:3]}")
    for key in query_keys:
        state[key] = donor[key].to(state[key].dtype)
save_file(state, adapter_path, metadata={"format": "pt"})

if uncap:
    config_path = target / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["max_residual_scale"] = None
    config_path.write_text(json.dumps(config, indent=2) + chr(10), encoding="utf-8")
print(json.dumps({"target": str(target), "query_source": query_source, "uncapped": uncap}))
PY
}

# train_arm <arm> <checkpoint> <steps> <interval> <decoder> <cap_arg> <gate>
train_arm() {
    local arm="$1" checkpoint="$2" steps="$3" interval="$4" decoder="$5" cap_arg="$6" gate="$7"
    local arm_run_id="${run_id}_${arm}"
    local summary="${runs_root}/${arm_run_id}/seed${seed}/summary.json"
    if [[ -f "${summary}" ]]; then
        echo "{\"event\":\"glmocr_scale_iou_arm_reused\",\"arm\":\"${arm}\"}"
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
        --mode layout_ot --sem-adapter-mlp --freeze-layout-branch \
        --layout-loss-profile ocr_only \
        --auxiliary-weight 0 --auxiliary-weight-start 0 --auxiliary-ramp-steps 0 \
        --residual-scale-cap "${cap_arg}" \
        --init-checkpoint-override-residual-scale "${gate}" \
        --init-checkpoint-dir "${checkpoint}" \
        --init-checkpoint-allow-mode-mismatch \
        --learning-rate "${learning_rate}" \
        --decoder-adaptation "${decoder}" \
        --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0 \
        --decoder-learning-rate "${decoder_learning_rate}" \
        --warmup-steps "${warmup_steps}" --min-lr-ratio "${min_lr_ratio}" \
        --max-steps "${steps}" --lr-schedule-steps "${steps}" \
        --validation-interval "${interval}" \
        --defer-validation --without-test \
        --max-pixels "${max_pixels}" --max-eval-new-tokens "${max_eval_new_tokens}" \
        --log-steps 16 --run-id "${arm_run_id}" \
        > "${root}/${arm}.train.log" 2>&1
    "${python}" - "${summary}" "${arm}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
training = payload.get("training") or {}
if payload.get("status") != "complete":
    raise SystemExit(f"arm {sys.argv[2]} did not complete")
if payload.get("test_manifest_read") is not False:
    raise SystemExit(f"arm {sys.argv[2]} read the test manifest")
print(json.dumps({
    "event": "glmocr_scale_iou_arm_trained",
    "arm": sys.argv[2],
    "gate": training.get("final_raw_content_gate"),
    "checkpoints": training.get("checkpoint_steps"),
}, separators=(",", ":")))
PY
}

# launch_eval <gpu> <label> <checkpoint> <out> [cap_arg]
launch_eval() {
    local gpu="$1" label="$2" checkpoint="$3" out="$4" cap_arg="${5:-0.03}"
    (
        setup_environment
        export CUDA_VISIBLE_DEVICES="${gpu}"
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
            --experiment-label "glmocr_scale_iou_${label}" \
            --learning-rate "${learning_rate}" --max-grad-norm 1.0 \
            --decoder-adaptation lora --decoder-lora-rank 8 --decoder-lora-alpha 8 \
            --decoder-lora-dropout 0 --decoder-learning-rate "${decoder_learning_rate}" \
            --box-head-mlp --query-refine-layers 1 --sem-adapter-mlp \
            --residual-scale-cap "${cap_arg}" --initial-residual-scale 0 \
            --auxiliary-weight 0 --auxiliary-weight-start 0 --auxiliary-ramp-steps 0 \
            --layout-loss-profile ocr_only --query-assignment hungarian \
            --processor-mode fast --adapter-precision fp32 \
            --max-pixels "${max_pixels}" --max-eval-new-tokens "${max_eval_new_tokens}" \
            --generation-mode "${eval_generation_mode}" \
            --validation-interval 1 --log-steps 16 \
            --eval-checkpoint-dir "${checkpoint}" --eval-only
    ) > "${out}.log" 2>&1
}

write_status() {
    printf '{"status":"%s","phase":"%s","run_id":"%s","updated_at":"%s"}\n' \
        "$1" "$2" "$3" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "${remote_root}/runs/${session}.status.json"
}

run_phase_a() {
    local ckpts="${root}/checkpoints" arms="${root}/arms"
    mkdir -p "${ckpts}" "${arms}"
    build_checkpoint "${ckpts}/Q-rand" random 0
    build_checkpoint "${ckpts}/Q-042" "${donor_low}" 0
    build_checkpoint "${ckpts}/Q-065" "${donor_mid}" 0
    train_arm Q-rand "${ckpts}/Q-rand" 256 128 lora "${cap}" "${gate_start}"
    train_arm Q-042  "${ckpts}/Q-042"  256 128 lora "${cap}" "${gate_start}"
    train_arm Q-065  "${ckpts}/Q-065"  256 128 lora "${cap}" "${gate_start}"

    local pids=()
    launch_eval 0 Q-rand "${runs_root}/${run_id}_Q-rand/seed${seed}/checkpoint-256" "${arms}/Q-rand" & pids+=("$!")
    launch_eval 1 Q-042  "${runs_root}/${run_id}_Q-042/seed${seed}/checkpoint-256"  "${arms}/Q-042"  & pids+=("$!")
    launch_eval 2 Q-065  "${runs_root}/${run_id}_Q-065/seed${seed}/checkpoint-256"  "${arms}/Q-065"  & pids+=("$!")
    wait "${pids[@]}" 2>/dev/null || true
}

run_phase_b() {
    local ckpts="${root}/checkpoints" arms="${root}/arms"
    mkdir -p "${arms}"
    build_checkpoint "${ckpts}/U-uncapped" keep 1
    train_arm U-uncapped "${ckpts}/U-uncapped" "${long_steps}" 256 frozen none "${u_gate_start}"

    # Four saved points, scored in parallel.
    local pids=() gpu=0 step
    for step in 256 512 768 "${long_steps}"; do
        [[ -f "${runs_root}/${run_id}_U-uncapped/seed${seed}/checkpoint-${step}/adapter.safetensors" ]] || continue
        launch_eval "${gpu}" "U-${step}" \
            "${runs_root}/${run_id}_U-uncapped/seed${seed}/checkpoint-${step}" \
            "${arms}/U-${step}" none & pids+=("$!")
        gpu=$((gpu + 1))
    done
    wait "${pids[@]}" 2>/dev/null || true
}

summarize() {
    setup_environment
    "${python}" - "${root}" "${runs_root}" "${compare_run_id}" <<'PY'
import json, sys
from pathlib import Path

root, runs_root, compare_run_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
arms = root / "arms"
iou = {
    "Q-rand": ("randomised query machinery (no layout information)", None),
    "Q-042": ("giou10x checkpoint-1000", 0.422549),
    "Q-065": ("giou10x checkpoint-10000", 0.647969),
}
payload = {"status": "complete", "arms": {}}
for name, (label, value) in iou.items():
    summary = json.loads((arms / name / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["validation"]
    payload["arms"][name] = {
        "label": label, "layout_iou": value, "cer": metrics["cer"],
        "substitutions": metrics["substitutions"], "insertions": metrics["insertions"],
        "deletions": metrics["deletions"],
        "residual_relative_norm": metrics.get("residual_relative_norm"),
        "predictions": str(arms / name / "validation_predictions.jsonl"),
    }
# The IoU=0.672 point is the W arm of the earlier comparison run, same recipe.
payload["arms"]["Q-067"] = {
    "label": "giou20x checkpoint-10000 (reused from the comparison run W arm)",
    "layout_iou": 0.672084, "cer": 0.11198120595144871,
    "predictions": str(Path(runs_root).parent / "layout_branch_finetune_compare" /
                       compare_run_id / "arms" / "W-branch-cap03" / "validation_predictions.jsonl"),
}
for step in (256, 512, 768, 1024):
    d = arms / f"U-{step}"
    if not (d / "summary.json").is_file():
        continue
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    metrics = summary["validation"]
    payload["arms"][f"U-{step}"] = {
        "label": f"uncapped gate, frozen decoder, step {step}",
        "cer": metrics["cer"],
        "effective_residual_scale": metrics.get("effective_residual_scale"),
        "residual_relative_norm": metrics.get("residual_relative_norm"),
        "repeated_trigram_rate": metrics.get("repeated_trigram_rate"),
        "repeated_cycle_page_rate": metrics.get("repeated_cycle_page_rate"),
        "generation_limit_hits": metrics.get("generation_limit_hits"),
        "insertions": metrics.get("insertions"),
        "mean_new_tokens": metrics.get("generation_mean_new_tokens"),
        "predictions": str(d / "validation_predictions.jsonl"),
    }
(root / "scale_iou_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
for name, row in sorted(payload["arms"].items()):
    print(f"{name:10s} IoU {str(row.get('layout_iou')):9s} CER {row['cer']:.6f}  res_norm={row.get('residual_relative_norm')}")
PY
}

main() {
    preflight
    setup_environment
    mkdir -p "${root}"
    write_status running "protocol" "${run_id}"
    prepare_protocol
    run_phase_a
    run_phase_b
    summarize
    write_status complete complete "${run_id}"
    echo "{\"event\":\"glmocr_scale_iou_complete\",\"run_id\":\"${run_id}\"}"
}

while (( $# > 0 )); do
    case "$1" in
        --foreground) foreground=1; shift ;;
        --run-id) run_id="$2"; session="$2"; shift 2 ;;
        --long-steps) long_steps="$2"; shift 2 ;;
        --checkpoints-only) checkpoints_only=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

train_protocol="${remote_root}/protocols/${run_id}.train_validation_no_test.json"
root="${remote_root}/layout_branch_scale_iou/${run_id}"

if (( checkpoints_only == 1 )); then
    # Build every arm checkpoint and stop: the surgery is the risky part and
    # this validates it without spending GPU time on training.
    preflight
    setup_environment
    mkdir -p "${root}/checkpoints"
    build_checkpoint "${root}/checkpoints/Q-rand" random 0
    build_checkpoint "${root}/checkpoints/Q-042" "${donor_low}" 0
    build_checkpoint "${root}/checkpoints/Q-065" "${donor_mid}" 0
    build_checkpoint "${root}/checkpoints/U-uncapped" keep 1
    exit 0
fi

if (( foreground == 1 )); then
    main
    exit 0
fi

mkdir -p "${root}"
tmux kill-session -t "${session}" 2>/dev/null || true
tmux new-session -d -s "${session}" \
    "export GLMOCR_SCALE_IOU_RUN_ID=$(printf '%q' "${run_id}"); "\
"export GLMOCR_SCALE_IOU_SESSION=$(printf '%q' "${session}"); "\
"export GLMOCR_SCALE_IOU_LONG_STEPS=$(printf '%q' "${long_steps}"); "\
"bash $(printf '%q' "${script_path}") --foreground > ${root}/pipeline.log 2>&1"
echo "{\"event\":\"glmocr_scale_iou_armed\",\"session\":\"${session}\",\"root\":\"${root}\"}"
