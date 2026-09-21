#!/usr/bin/env bash
# Decoder learned-mask routing: Gate B smoke and Gate C screen on A100.
#
# Each arm is a self-contained single-device run (train_decoder_mask.py consumes
# one page at a time, no DDP), so the arms are independent and are dispatched one
# per allowlisted GPU in waves.  Parallelism changes wall-clock only: every arm
# keeps the same base, seed, data order, budget and selection metric.  It never
# opens the MTHv2 test manifest: the screen validates on a fixed 64-page subset
# of the official validation split and selects by free-generation CER only.
#
# Modes:
#   --mode smoke   Gate B: 4+4 pages, 8 updates, checkpoint save + reload check.
#   --mode screen  Gate C: 128/64 split (seed 42), arms B0..B3 serially, then a
#                  validation-only selection via summarize_decoder_mask.py.
#   --mode formal  Gate D: not implemented in the first cut; reserved.
#
# The GPU gate follows AGENTS.md rule 9: it queries only the allowlisted card's
# instantaneous ``utilization.gpu`` and starts only if every target is strictly
# below the limit; otherwise it exits entirely.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
# This branch has no env of its own yet: it reuses the layout_ot Python env
# (torch 2.8 + transformers 5.3.0).  Only code and artifacts are branch-isolated
# per AGENTS.md rule 7; the env is a shared read-only resource.
env_dir="${GLMOCR_A100_ENV:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dataset_root="${GLMOCR_A100_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1}"

mode="screen"
# The allowlist is exactly the cards the screen uses: one card per arm, one arm
# per card.  AGENTS.md rule 9 requires every allowlisted card to pass the
# utilization gate before anything starts, so listing idle cards would only
# tighten the gate for no gain.
gpu_ids="0,1,2,3"
gpu_utilization_limit=50
seed=42
screen_id="glmocr_decoder_mask_finegrid_v1"
session=""
foreground=0
max_steps=1024
screen_train_pages=128
screen_validation_pages=64
smoke_train_pages=4
smoke_validation_pages=4
smoke_max_steps=8
max_pixels=1003520
# 15.6% of MTHv2 pages have more than 512 reference tokens (max 1349), so a 512
# cap silently truncates them; 1536 covers every page in both splits.
max_eval_new_tokens=1536

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) mode="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --gpu-utilization-limit) gpu_utilization_limit="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --screen-id) screen_id="$2"; shift 2 ;;
        --max-steps) max_steps="$2"; shift 2 ;;
        --max-pixels) max_pixels="$2"; shift 2 ;;
        --max-eval-new-tokens) max_eval_new_tokens="$2"; shift 2 ;;
        --session) session="$2"; shift 2 ;;
        --foreground) foreground=1; shift ;;
        --remote-root) remote_root="$2"; shift 2 ;;
        --code-root) code_root="$2"; shift 2 ;;
        --env-dir) env_dir="$2"; shift 2 ;;
        --model-dir) model_dir="$2"; shift 2 ;;
        --dataset-root) dataset_root="$2"; shift 2 ;;
        *) printf '{"event":"decoder_mask_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

case "${mode}" in
    smoke|screen|formal) ;;
    *) printf '{"event":"decoder_mask_failed","error":"invalid_mode","value":"%s"}\n' "${mode}" >&2; exit 64 ;;
esac
[[ "${screen_id}" =~ ^[A-Za-z0-9_.-]+$ && "${seed}" =~ ^[0-9]+$ ]] || {
    printf '{"event":"decoder_mask_failed","error":"invalid_screen_id_or_seed"}\n' >&2; exit 64
}
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    printf '{"event":"decoder_mask_failed","error":"invalid_gpu_ids","gpu_ids":"%s"}\n' "${gpu_ids}" >&2; exit 64
}
[[ "${gpu_utilization_limit}" =~ ^[1-9][0-9]*$ && "${max_steps}" =~ ^[1-9][0-9]*$ ]] || {
    printf '{"event":"decoder_mask_failed","error":"invalid_numeric_configuration"}\n' >&2; exit 64
}
[[ "${max_pixels}" =~ ^[1-9][0-9]*$ && "${max_eval_new_tokens}" =~ ^[1-9][0-9]*$ ]] || {
    printf '{"event":"decoder_mask_failed","error":"invalid_evaluation_configuration"}\n' >&2; exit 64
}
IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
[[ -z "${session}" ]] && session="${screen_id}_${mode}"
[[ "${session}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    printf '{"event":"decoder_mask_failed","error":"invalid_session"}\n' >&2; exit 64
}

python="${env_dir}/bin/python"
train_cli="${code_root}/src/layout_ocr/train_decoder_mask.py"
evaluate_cli="${code_root}/tools/evaluation/evaluate_decoder_mask.py"
summarize_cli="${code_root}/tools/summarize_decoder_mask.py"
char_tool="${code_root}/tools/prepare_mthv2_char_manifest.py"
train_char_manifest="${dataset_root}/train/manifest.char.jsonl"
validation_char_manifest="${dataset_root}/validation/manifest.char.jsonl"
screen_root="${remote_root}/training_runs/${screen_id}"
arm_root="${screen_root}/arms"
split_root="${screen_root}/split"
smoke_dir="${screen_root}/smoke"
launcher_log="${remote_root}/runs/${screen_id}.${mode}.launcher.log"

torch_lib="${env_dir}/lib/python3.11/site-packages/torch/lib"
machine_arch="$(uname -m)"
case "${machine_arch}" in
    x86_64) cuda_target_arch="x86_64" ;;
    aarch64|arm64) cuda_target_arch="aarch64" ;;
    *) cuda_target_arch="${machine_arch}" ;;
esac
cuda_library_path="${torch_lib}"
system_cuda_library="/usr/local/cuda/targets/${cuda_target_arch}-linux/lib"
if [[ -d "${system_cuda_library}" ]]; then
    cuda_library_path="${system_cuda_library}:${cuda_library_path}"
fi
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    component_lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
    if [[ -d "${component_lib}" ]]; then
        cuda_library_path="${cuda_library_path}:${component_lib}"
    fi
done

preflight_paths() {
    [[ -x "${python}" ]] || {
        printf '{"event":"decoder_mask_failed","error":"missing_python"}\n' >&2; exit 66
    }
    [[ -f "${model_dir}/model.safetensors" && -f "${model_dir}/config.json" ]] || {
        printf '{"event":"decoder_mask_failed","error":"missing_model"}\n' >&2; exit 66
    }
    [[ -f "${train_cli}" && -f "${evaluate_cli}" && -f "${summarize_cli}" && -f "${char_tool}" ]] || {
        printf '{"event":"decoder_mask_failed","error":"missing_decoder_mask_source"}\n' >&2; exit 66
    }
    [[ -f "${dataset_root}/train/manifest.jsonl" && -f "${dataset_root}/validation/manifest.jsonl" ]] || {
        printf '{"event":"decoder_mask_failed","error":"missing_mthv2_manifest"}\n' >&2; exit 66
    }
}

query_gpu_utilization() {
    command -v nvidia-smi >/dev/null 2>&1 || {
        printf '{"event":"decoder_mask_failed","error":"nvidia_smi_missing"}\n' >&2; exit 69
    }
    declare -A observed=()
    while IFS=',' read -r observed_id utilization; do
        observed_id="${observed_id//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        [[ "${observed_id}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || {
            printf '{"event":"decoder_mask_failed","error":"cannot_parse_gpu_utilization"}\n' >&2; exit 69
        }
        observed[${observed_id}]="${utilization}"
    done < <(nvidia-smi -i "${gpu_ids}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
    for gpu in "${gpu_array[@]}"; do
        [[ -n "${observed[${gpu}]+present}" ]] || {
            printf '{"event":"decoder_mask_failed","error":"requested_gpu_not_reported","gpu":"%s"}\n' "${gpu}" >&2; exit 69
        }
        (( observed[${gpu}] < gpu_utilization_limit )) || {
            printf '{"event":"decoder_mask_failed","error":"gpu_admission_failed","gpu":"%s","utilization":%s,"limit":%s}\n' "${gpu}" "${observed[${gpu}]}" "${gpu_utilization_limit}" >&2
            exit 75
        }
    done
    printf '{"event":"decoder_mask_gpu_admission_ok","gpu_ids":"%s","limit":%s}\n' "${gpu_ids}" "${gpu_utilization_limit}"
}

write_status() {
    local status="$1"
    mkdir -p "${screen_root}/status"
    printf '{"status":"%s","screen_id":"%s","mode":"%s","seed":%s,"gpu_ids":"%s"}\n' \
        "${status}" "${screen_id}" "${mode}" "${seed}" "${gpu_ids}" \
        > "${screen_root}/status/${mode}.json"
}

export_environment() {
    export CUDA_VISIBLE_DEVICES="${gpu_ids}"
    export TMPDIR="${screen_root}/tmp/${mode}"
    export HF_HOME="${TMPDIR}/huggingface"
    export TRANSFORMERS_CACHE="${HF_HOME}"
    export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="${cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    mkdir -p "${TMPDIR}" "${HF_HOME}"
    cd "${code_root}"
}

ensure_char_manifests() {
    if [[ ! -f "${train_char_manifest}" || ! -f "${validation_char_manifest}" ]]; then
        "${python}" "${char_tool}" \
            --dataset-root "${dataset_root}" \
            --splits train validation > "${screen_root}/logs/char_manifest.log" 2>&1
    fi
    [[ -f "${train_char_manifest}" && -f "${validation_char_manifest}" ]] || {
        printf '{"event":"decoder_mask_failed","error":"char_manifest_missing"}\n' >&2; exit 66
    }
}

prepare_split() {
    local train_n="$1" validation_n="$2" tag="$3"
    mkdir -p "${split_root}"
    "${python}" - "${train_char_manifest}" "${validation_char_manifest}" "${train_n}" "${validation_n}" "${seed}" "${split_root}" "${tag}" <<'PY'
import hashlib
import json
import random
import sys
from pathlib import Path

train_src, val_src = Path(sys.argv[1]), Path(sys.argv[2])
train_n, val_n = int(sys.argv[3]), int(sys.argv[4])
seed = int(sys.argv[5])
out_dir = Path(sys.argv[6])
tag = sys.argv[7]


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path, records):
    digest = hashlib.sha256()
    lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records]
    for line in lines:
        digest.update(line.encode("utf-8"))
    path.write_text("".join(lines), encoding="utf-8")
    return digest.hexdigest()


train = load(train_src)
val = load(val_src)
rng = random.Random(seed)
train_sel = sorted(rng.sample(train, train_n), key=lambda r: str(r["page_id"]))
val_sel = sorted(rng.sample(val, val_n), key=lambda r: str(r["page_id"]))

# The subset manifests live under split_root/, but their images live under the
# source split dirs.  load_records() resolves a relative image path against the
# manifest's own parent, so rewrite each image to an absolute path now.
def absolutize(records, src_parent):
    for r in records:
        img = r.get("image_path") or r.get("image")
        p = Path(str(img))
        if not p.is_absolute():
            r["image"] = str((src_parent / p).resolve())

absolutize(train_sel, train_src.parent)
absolutize(val_sel, val_src.parent)

# The converter already records the isolation signal.  The official split is
# random at the page level (group_isolation_status=unavailable_official_random_
# page_split), so true book/collection isolation is not available for this
# first-stage screen; we record the authoritative status and the train/validation
# source-group overlap rather than pretending isolation held.
isolation_status = sorted({str(r.get("group_isolation_status")) for r in train if "group_isolation_status" in r})
group_fields = [f for f in ("source_group_id", "source", "book", "collection", "book_id") if any(f in r for r in train)]
group_meta = {"isolation_status": isolation_status}
if group_fields:
    field = group_fields[0]
    train_groups = {str(r[field]) for r in train_sel}
    val_groups = {str(r[field]) for r in val_sel}
    group_meta.update(
        {
            "field": field,
            "train_group_count": len(train_groups),
            "validation_group_count": len(val_groups),
            "overlap_group_count": len(train_groups & val_groups),
        }
    )
else:
    group_meta.update({"field": None, "note": "official_split_only; full source isolation unavailable"})

train_sha = write(out_dir / f"train{train_n}_{tag}_seed{seed}.jsonl", train_sel)
val_sha = write(out_dir / f"validation{val_n}_{tag}_seed{seed}.jsonl", val_sel)
report = {
    "seed": seed,
    "tag": tag,
    "train_pages": len(train_sel),
    "validation_pages": len(val_sel),
    "train_manifest_sha256": train_sha,
    "validation_manifest_sha256": val_sha,
    "train_page_ids": [r["page_id"] for r in train_sel],
    "validation_page_ids": [r["page_id"] for r in val_sel],
    "source_group": group_meta,
    "test_manifest_read": False,
}
(out_dir / f"split_{tag}_seed{seed}.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
PY
}

arm_training_args() {
    # Echoes the flag composition for one arm; the caller appends the subset
    # manifests and output dir.  Everything the three learned arms share (fine
    # grid, max pooling, BCE+Dice, both learning rates, beta) lives in
    # ``run_training``'s fixed list, so each arm below changes exactly ONE
    # variable.  That is what makes the comparisons interpretable:
    #   G1 vs G2 : window target vs single-token target
    #   G2 vs G3 : deterministic MLP head vs conditional VAE head
    #   each vs B0 : does a head help at all
    # G1 vs G3 differs in two factors and must not be read as a single contrast.
    local arm="$1"
    case "${arm}" in
        B0) printf '%s' "--routing-mode none" ;;
        G2) printf '%s' "--routing-mode learned --router-target-mode token" ;;
        G1) printf '%s' "--routing-mode learned --router-target-mode window" ;;
        G3) printf '%s' "--routing-mode learned --router-head vae" ;;
        *) printf '{"event":"decoder_mask_failed","error":"invalid_arm","arm":"%s"}\n' "${arm}" >&2; exit 64 ;;
    esac
}

run_training() {
    local arm="$1" train_manifest="$2" validation_manifest="$3" output_dir="$4" steps="$5" validation_steps="$6"
    local routing_args
    routing_args="$(arm_training_args "${arm}")"
    mkdir -p "$(dirname -- "${output_dir}")"
    "${python}" -m layout_ocr.train_decoder_mask \
        --model-path "${model_dir}" \
        --train-manifest "${train_manifest}" \
        --validation-manifest "${validation_manifest}" \
        --output-dir "${output_dir}" \
        --seed "${seed}" \
        --max-steps "${steps}" \
        --max-pixels "${max_pixels}" \
        --processor-mode slow \
        --lora-rank 8 --lora-alpha 8 \
        --learning-rate 1e-6 \
        --router-learning-rate 1e-4 \
        --router-visual-source fine \
        --router-pool-mode max \
        --router-dice-weight 1.0 \
        --router-mask-bce balanced \
        --router-window-size 3 5 \
        --router-vae-latent-channels 4 --router-vae-latent-size 16 \
        --router-vae-kl-weight 1.0 --router-vae-kl-warmup-steps 200 \
        --router-vae-kl-free-bits 0.05 \
        --checkpoint-every 256 \
        --validation-every 256 \
        --validation-steps ${validation_steps} \
        --max-eval-new-tokens "${max_eval_new_tokens}" \
        --router-split-layer 8 \
        --router-dim 256 \
        --router-bias-max 2.0 \
        --router-bias-warmup-steps 100 \
        --router-mask-loss-weight 0.2 \
        --router-stop-loss-weight 0.05 \
        --router-detach-every 64 \
        --router-mask-feedback-noise 0.15 \
        --router-input-noise 0.05 \
        --router-noise-warmup-steps 200 \
        ${routing_args}
}

run_smoke() {
    local train_manifest validation_manifest
    train_manifest="${split_root}/train${smoke_train_pages}_smoke_seed${seed}.jsonl"
    validation_manifest="${split_root}/validation${smoke_validation_pages}_smoke_seed${seed}.jsonl"
    if [[ ! -f "${train_manifest}" ]]; then
        prepare_split "${smoke_train_pages}" "${smoke_validation_pages}" "smoke" > "${split_root}/split_smoke.log"
    fi
    # Both new code paths in one wall clock: G3 exercises the conditional VAE,
    # the pre-merge capture and the fine->merged pooling; G1 exercises the window
    # target and the annotation-driven line grouping.  One arm per card, parallel.
    local pids=()
    local smoke_arm
    for slot in 0 1; do
        smoke_arm="G3"
        [[ "${slot}" -eq 1 ]] && smoke_arm="G1"
        local smoke_card="${gpu_array[${slot}]}"
        local smoke_arm_dir="${smoke_dir}/${smoke_arm}"
        mkdir -p "${smoke_arm_dir}"
        (
            export CUDA_VISIBLE_DEVICES="${smoke_card}"
            export TMPDIR="${screen_root}/tmp/${mode}/${smoke_arm}"
            export HF_HOME="${TMPDIR}/huggingface"
            mkdir -p "${TMPDIR}" "${HF_HOME}"
            run_training "${smoke_arm}" "${train_manifest}" "${validation_manifest}" \
                "${smoke_arm_dir}" "${smoke_max_steps}" "${smoke_max_steps}"
        ) > "${smoke_arm_dir}/train.log" 2>&1 &
        pids+=("$!")
    done
    local smoke_failed=0
    for pid in "${pids[@]}"; do wait "${pid}" || smoke_failed=$((smoke_failed + 1)); done
    if (( smoke_failed > 0 )); then
        printf '{"event":"decoder_mask_failed","error":"smoke_train_failed","count":%s}\n' "${smoke_failed}" >&2
        return 1
    fi
    # Checkpoint reload: score the last G3 checkpoint on the smoke validation pages.
    local checkpoint_dir="${smoke_dir}/G3/step-${smoke_max_steps}"
    [[ -d "${checkpoint_dir}" ]] || {
        printf '{"event":"decoder_mask_failed","error":"smoke_checkpoint_missing","dir":"%s"}\n' "${checkpoint_dir}" >&2; return 1
    }
    "${python}" "${evaluate_cli}" \
        --model-path "${model_dir}" \
        --checkpoint-dir "${checkpoint_dir}" \
        --manifest "${validation_manifest}" \
        --output-dir "${smoke_dir}/reload-check" \
        --max-new-tokens "${max_eval_new_tokens}" \
        > "${smoke_dir}/reload-check.log" 2>&1
    printf '{"event":"decoder_mask_smoke_complete","screen_id":"%s","output_dir":"%s"}\n' "${screen_id}" "${smoke_dir}"
}

run_screen() {
    local train_manifest validation_manifest
    train_manifest="${split_root}/train${screen_train_pages}_screen_seed${seed}.jsonl"
    validation_manifest="${split_root}/validation${screen_validation_pages}_screen_seed${seed}.jsonl"
    if [[ ! -f "${train_manifest}" ]]; then
        prepare_split "${screen_train_pages}" "${screen_validation_pages}" "screen" > "${split_root}/split_screen.log"
    fi
    # One arm per allowlisted card, dispatched in waves of ``n_cards`` so the
    # same code stays correct if the allowlist is later narrowed to fewer cards
    # than arms (a narrow allowlist must not silently co-schedule two arms on one
    # card, which would oversubscribe its memory instead of queueing).
    local arms=(G1 G2 G3 B0)
    local n_cards=${#gpu_array[@]}
    local total=${#arms[@]}
    local index=0
    local failures=0
    while (( index < total )); do
        local pids=() wave_arms=() wave_logs=() slot=0
        while (( slot < n_cards && index < total )); do
            local arm="${arms[${index}]}"
            local card="${gpu_array[${slot}]}"
            local arm_dir="${arm_root}/${arm}"
            local arm_log="${arm_dir}/train.log"
            # The redirect below opens the log before the arm can create its own
            # output dir, so the dir has to exist first.
            mkdir -p "${arm_dir}"
            (
                export CUDA_VISIBLE_DEVICES="${card}"
                export TMPDIR="${screen_root}/tmp/${mode}/${arm}"
                export HF_HOME="${TMPDIR}/huggingface"
                export TRANSFORMERS_CACHE="${HF_HOME}"
                mkdir -p "${TMPDIR}" "${HF_HOME}"
                run_training "${arm}" "${train_manifest}" "${validation_manifest}" "${arm_dir}" "${max_steps}" "256 512 1024"
            ) > "${arm_log}" 2>&1 &
            pids+=("$!")
            wave_arms+=("${arm}")
            wave_logs+=("${arm_log}")
            printf '{"event":"decoder_mask_arm_launched","screen_id":"%s","arm":"%s","gpu":"%s","log":"%s"}\n' \
                "${screen_id}" "${arm}" "${card}" "${arm_log}"
            index=$((index + 1))
            slot=$((slot + 1))
        done
        local wave_index=0
        while (( wave_index < ${#pids[@]} )); do
            if wait "${pids[${wave_index}]}"; then
                printf '{"event":"decoder_mask_arm_complete","screen_id":"%s","arm":"%s"}\n' \
                    "${screen_id}" "${wave_arms[${wave_index}]}"
            else
                failures=$((failures + 1))
                printf '{"event":"decoder_mask_arm_failed","screen_id":"%s","arm":"%s","log":"%s"}\n' \
                    "${screen_id}" "${wave_arms[${wave_index}]}" "${wave_logs[${wave_index}]}" >&2
            fi
            wave_index=$((wave_index + 1))
        done
    done
    if (( failures > 0 )); then
        printf '{"event":"decoder_mask_failed","error":"arm_failures","count":%s}\n' "${failures}" >&2
        return 1
    fi
    "${python}" "${summarize_cli}" --screen-root "${screen_root}" --output "${screen_root}/selection.json"
}

run_inner() {
    trap 'rc=$?; write_status failed; exit "$rc"' ERR
    preflight_paths
    query_gpu_utilization
    mkdir -p "${screen_root}/logs" "${screen_root}/status" "${split_root}" "${remote_root}/runs"
    write_status running
    export_environment
    ensure_char_manifests
    case "${mode}" in
        smoke) run_smoke ;;
        screen) run_screen ;;
        formal) printf '{"event":"decoder_mask_failed","error":"formal_mode_not_implemented"}\n' >&2; exit 64 ;;
    esac
    write_status complete
    trap - ERR
    printf '{"event":"decoder_mask_complete","screen_id":"%s","mode":"%s","seed":%s,"test_used_for_selection":false}\n' "${screen_id}" "${mode}" "${seed}"
}

preflight_paths
if (( foreground == 0 )); then
    command -v tmux >/dev/null 2>&1 || {
        printf '{"event":"decoder_mask_failed","error":"tmux_missing"}\n' >&2; exit 69
    }
    tmux has-session -t "${session}" 2>/dev/null && {
        printf '{"event":"decoder_mask_failed","error":"session_already_exists","session":"%s"}\n' "${session}" >&2; exit 73
    }
    mkdir -p "${remote_root}/runs"
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    command_line="$(printf '%q ' bash "${script_path}" --foreground --mode "${mode}" --gpu-ids "${gpu_ids}" --gpu-utilization-limit "${gpu_utilization_limit}" --seed "${seed}" --screen-id "${screen_id}" --max-steps "${max_steps}" --max-pixels "${max_pixels}" --max-eval-new-tokens "${max_eval_new_tokens}" --remote-root "${remote_root}" --code-root "${code_root}" --env-dir "${env_dir}" --model-dir "${model_dir}" --dataset-root "${dataset_root}")"
    tmux new-session -d -s "${session}" "cd $(printf '%q' "${code_root}") && exec ${command_line} >$(printf '%q' "${launcher_log}") 2>&1"
    printf '{"event":"decoder_mask_armed","session":"%s","screen_id":"%s","mode":"%s","seed":%s,"gpu_ids":"%s","test_used_for_selection":false,"log":"%s"}\n' \
        "${session}" "${screen_id}" "${mode}" "${seed}" "${gpu_ids}" "${launcher_log}"
else
    run_inner
fi
