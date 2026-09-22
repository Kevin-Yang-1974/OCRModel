#!/usr/bin/env bash
# Five-card screening: two oracle ceilings + three 128-step head retrains.
#
# The prior head-only screen (run_glmocr_decoder_mask_headonly_a100.sh) trained
# G1/G2/G3 serially on all five cards via data-parallel DDP and the head never
# learned to localize (Dice ~0.02, mask_mean 0.02 -> 0.125, column-level strips).
# This launcher is the follow-up the diagnosis motivates:
#
#   * the mask loss gains a centre-of-mass (centroid) term and a sparsity term
#     so a dense, confident, mislocated mask is punished directly;
#   * the head LR gets a linear warmup then cosine anneal instead of a constant
#     rate, so the recurrence does not overshoot into the dense regime early;
#   * two oracle ceilings (line-level and 3-5-char window GT masks injected
#     during generation) establish what *perfect* routing would buy against
#     B0's zero-shot CER (0.287) -- the number the retrained heads must beat.
#
# Card assignment (one job per card, all five run in parallel):
#   GPU 0 -> G1 (window head)
#   GPU 1 -> G2 (token  head)
#   GPU 2 -> G3 (vae    head)
#   GPU 3 -> oracle line
#   GPU 4 -> oracle window (3-5)
#
# Each arm is a single process (no torchrun): the 128-step screening trades the
# previous five-card DDP for one card per arm so the three arms and the two
# oracles all finish in one wave.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dataset_root="${GLMOCR_A100_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1}"

screen_id="glmocr_decoder_mask_oracle_128step_v1"
seed=42
gpu_ids="0,1,2,3,4"
gpu_utilization_limit=50
max_pixels=4000000
max_steps=128
checkpoint_every=64
validation_steps="128"
train_pages=128
validation_pages=64
max_eval_new_tokens=1536
foreground=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --screen-id) screen_id="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --max-pixels) max_pixels="$2"; shift 2 ;;
        --max-steps) max_steps="$2"; shift 2 ;;
        --foreground) foreground=1; shift ;;
        --remote-root) remote_root="$2"; shift 2 ;;
        --code-root) code_root="$2"; shift 2 ;;
        --env-dir) env_dir="$2"; shift 2 ;;
        --model-dir) model_dir="$2"; shift 2 ;;
        --dataset-root) dataset_root="$2"; shift 2 ;;
        *) printf '{"event":"oracle128_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
[[ "${#gpu_array[@]}" -eq 5 ]] || {
    printf '{"event":"oracle128_failed","error":"need_exactly_five_gpus"}\n' >&2; exit 64
}
python="${env_dir}/bin/python"
oracle_cli="${code_root}/tools/oracle_decoder_mask_generate.py"
char_tool="${code_root}/tools/prepare_mthv2_char_manifest.py"
train_char_manifest="${dataset_root}/train/manifest.char.jsonl"
validation_char_manifest="${dataset_root}/validation/manifest.char.jsonl"
screen_root="${remote_root}/training_runs/${screen_id}"
split_root="${screen_root}/split"
arm_root="${screen_root}/arms"
oracle_root="${screen_root}/oracle"
launcher_log="${remote_root}/runs/${screen_id}.launcher.log"

torch_lib="${env_dir}/lib/python3.11/site-packages/torch/lib"
cuda_library_path="${torch_lib}"
system_cuda="/usr/local/cuda/targets/$(uname -m)-linux/lib"
[[ -d "${system_cuda}" ]] && cuda_library_path="${system_cuda}:${cuda_library_path}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    component_lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
    [[ -d "${component_lib}" ]] && cuda_library_path="${cuda_library_path}:${component_lib}"
done

write_status() {
    mkdir -p "${screen_root}/status"
    printf '{"status":"%s","screen_id":"%s","seed":%s,"gpu_ids":"%s"}\n' \
        "$1" "${screen_id}" "${seed}" "${gpu_ids}" > "${screen_root}/status/screen.json"
}

prepare_split() {
    local train_n="$1" validation_n="$2" tag="$3"
    mkdir -p "${split_root}"
    "${python}" - "${train_char_manifest}" "${validation_char_manifest}" "${train_n}" "${validation_n}" "${seed}" "${split_root}" "${tag}" <<'PY'
import hashlib, json, random, sys
from pathlib import Path
train_src, val_src = Path(sys.argv[1]), Path(sys.argv[2])
train_n, val_n, seed = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
out_dir, tag = Path(sys.argv[6]), sys.argv[7]
load = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
train, val = load(train_src), load(val_src)
rng = random.Random(seed)
train_sel = sorted(rng.sample(train, train_n), key=lambda r: str(r["page_id"]))
val_sel = sorted(rng.sample(val, val_n), key=lambda r: str(r["page_id"]))
def absolutize(records, src_parent):
    for r in records:
        img = r.get("image_path") or r.get("image")
        p = Path(str(img))
        if not p.is_absolute():
            r["image"] = str((src_parent / p).resolve())
absolutize(train_sel, train_src.parent)
absolutize(val_sel, val_src.parent)
def write(path, records):
    d = hashlib.sha256()
    lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records]
    for line in lines:
        d.update(line.encode("utf-8"))
    path.write_text("".join(lines), encoding="utf-8")
    return d.hexdigest()
report = {
    "seed": seed, "tag": tag,
    "train_pages": len(train_sel), "validation_pages": len(val_sel),
    "train_manifest_sha256": write(out_dir / f"train{train_n}_{tag}_seed{seed}.jsonl", train_sel),
    "validation_manifest_sha256": write(out_dir / f"validation{val_n}_{tag}_seed{seed}.jsonl", val_sel),
    "train_page_ids": [r["page_id"] for r in train_sel],
    "validation_page_ids": [r["page_id"] for r in val_sel],
    "test_manifest_read": False,
}
(out_dir / f"split_{tag}_seed{seed}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
PY
}

arm_training_args() {
    case "$1" in
        G2) printf '%s' "--router-target-mode token" ;;
        G1) printf '%s' "--router-target-mode window" ;;
        G3) printf '%s' "--router-head vae" ;;
        *) printf '{"event":"oracle128_failed","error":"invalid_arm","arm":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
}

run_training_arm() {
    local arm="$1" gpu="$2" train_manifest="$3" validation_manifest="$4"
    local arm_dir="${arm_root}/${arm}"
    local tmp="${screen_root}/tmp/train_${arm}"
    mkdir -p "${arm_dir}" "${tmp}"
    local routing_args
    routing_args="$(arm_training_args "${arm}")"
    printf '{"event":"oracle128_arm_launched","arm":"%s","gpu":"%s"}\n' "${arm}" "${gpu}"
    TMPDIR="${tmp}" HF_HOME="${tmp}/huggingface" CUDA_VISIBLE_DEVICES="${gpu}" \
        setsid nohup bash -c "exec ${python} -m layout_ocr.train_decoder_mask \
        --model-path ${model_dir} \
        --train-manifest ${train_manifest} \
        --validation-manifest ${validation_manifest} \
        --output-dir ${arm_dir} \
        --seed ${seed} \
        --max-steps ${max_steps} \
        --max-pixels ${max_pixels} \
        --processor-mode slow \
        --head-only \
        --routing-mode learned \
        --router-split-layer 8 --router-dim 256 \
        --router-bias-max 2.0 \
        --router-bias-warmup-steps 100 \
        --router-mask-loss-weight 0.2 \
        --router-stop-loss-weight 0.05 \
        --router-detach-every 64 \
        --router-mask-feedback-noise 0.15 --router-input-noise 0.05 \
        --router-noise-warmup-steps 200 \
        --router-visual-source merged \
        --router-pool-mode max \
        --router-dice-weight 1.0 \
        --router-mask-bce balanced \
        --router-centroid-weight 1.0 \
        --router-sparsity-weight 1.0 \
        --router-window-size 3 5 \
        --router-vae-latent-channels 4 --router-vae-latent-size 16 \
        --router-vae-kl-weight 1.0 --router-vae-kl-warmup-steps 200 --router-vae-kl-free-bits 0.05 \
        --learning-rate 1e-6 \
        --router-learning-rate 1e-4 \
        --router-lr-warmup-steps 32 \
        --router-lr-min-ratio 0.1 \
        --checkpoint-every ${checkpoint_every} \
        --validation-every ${checkpoint_every} \
        --validation-steps ${validation_steps} \
        --max-eval-new-tokens ${max_eval_new_tokens} \
        ${routing_args}" > "${arm_dir}/train.log" 2>&1 < /dev/null &
    disown 2>/dev/null || true
}

run_oracle() {
    local mode="$1" gpu="$2" train_manifest="$3" validation_manifest="$4"
    local out_dir="${oracle_root}/${mode}"
    local tmp="${screen_root}/tmp/oracle_${mode}"
    mkdir -p "${out_dir}" "${tmp}"
    local mode_args="--oracle-mode ${mode}"
    if [[ "${mode}" == "window" ]]; then
        mode_args="--oracle-mode window --window-size 3 5"
    fi
    printf '{"event":"oracle128_oracle_launched","mode":"%s","gpu":"%s"}\n' "${mode}" "${gpu}"
    TMPDIR="${tmp}" HF_HOME="${tmp}/huggingface" CUDA_VISIBLE_DEVICES="${gpu}" \
        setsid nohup bash -c "exec ${python} ${oracle_cli} \
        --model-path ${model_dir} \
        --manifest ${validation_manifest} \
        --train-manifest ${train_manifest} \
        --output-dir ${out_dir} \
        ${mode_args} \
        --bias-max 2.0 \
        --max-pixels ${max_pixels} \
        --max-new-tokens ${max_eval_new_tokens} \
        --processor-mode slow" > "${out_dir}/run.log" 2>&1 < /dev/null &
    disown 2>/dev/null || true
}

run_inner() {
    trap 'rc=$?; write_status failed; exit "$rc"' ERR
    [[ -x "${python}" && -f "${oracle_cli}" ]] || {
        printf '{"event":"oracle128_failed","error":"missing_source"}\n' >&2; exit 66
    }
    # A100 admission: every target card must be under the utilization limit, or
    # the whole screen exits without touching any GPU.
    declare -A observed=()
    while IFS=',' read -r observed_id utilization; do
        observed_id="${observed_id//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"
        observed[${observed_id}]="${utilization}"
    done < <(nvidia-smi -i "${gpu_ids}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
    for gpu in "${gpu_array[@]}"; do
        [[ -n "${observed[${gpu}]+present}" ]] || {
            printf '{"event":"oracle128_failed","error":"gpu_not_reported","gpu":"%s"}\n' "${gpu}" >&2; exit 69; }
        (( observed[${gpu}] < gpu_utilization_limit )) || {
            printf '{"event":"oracle128_failed","error":"gpu_admission_failed","gpu":"%s","utilization":%s}\n' \
                "${gpu}" "${observed[${gpu}]}" >&2; exit 75; }
    done
    printf '{"event":"oracle128_gpu_admission_ok","gpu_ids":"%s","limit":%s}\n' "${gpu_ids}" "${gpu_utilization_limit}"

    export PYTHONPATH="${code_root}/src:${code_root}"
    export LD_LIBRARY_PATH="${cuda_library_path}"
    export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    mkdir -p "${screen_root}/logs" "${screen_root}/status" "${split_root}" "${arm_root}" "${oracle_root}" "${remote_root}/runs"
    write_status running
    cd "${code_root}"

    if [[ ! -f "${train_char_manifest}" || ! -f "${validation_char_manifest}" ]]; then
        "${python}" "${char_tool}" --dataset-root "${dataset_root}" --splits train validation \
            > "${screen_root}/logs/char_manifest.log" 2>&1
    fi
    local train_manifest validation_manifest
    train_manifest="${split_root}/train${train_pages}_screen_seed${seed}.jsonl"
    validation_manifest="${split_root}/validation${validation_pages}_screen_seed${seed}.jsonl"
    [[ -f "${train_manifest}" ]] || prepare_split "${train_pages}" "${validation_pages}" "screen" > "${split_root}/split_screen.log"

    run_training_arm "G1" "${gpu_array[0]}" "${train_manifest}" "${validation_manifest}"
    run_training_arm "G2" "${gpu_array[1]}" "${train_manifest}" "${validation_manifest}"
    run_training_arm "G3" "${gpu_array[2]}" "${train_manifest}" "${validation_manifest}"
    run_oracle "line" "${gpu_array[3]}" "${train_manifest}" "${validation_manifest}"
    run_oracle "window" "${gpu_array[4]}" "${train_manifest}" "${validation_manifest}"

    write_status launched
    trap - ERR
    printf '{"event":"oracle128_launched","screen_id":"%s","seed":%s,"test_used_for_selection":false}\n' "${screen_id}" "${seed}"
}

if (( foreground == 1 )); then
    run_inner
else
    mkdir -p "${remote_root}/runs"
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    command_line="$(printf '%q ' bash "${script_path}" --foreground --screen-id "${screen_id}" --seed "${seed}" --gpu-ids "${gpu_ids}" --max-pixels "${max_pixels}" --max-steps "${max_steps}" --remote-root "${remote_root}" --code-root "${code_root}" --env-dir "${env_dir}" --model-dir "${model_dir}" --dataset-root "${dataset_root}")"
    setsid nohup bash -c "exec ${command_line}" > "${launcher_log}" 2>&1 < /dev/null &
    disown 2>/dev/null || true
    printf '{"event":"oracle128_armed","screen_id":"%s","seed":%s,"gpu_ids":"%s","log":"%s","test_used_for_selection":false}\n' \
        "${screen_id}" "${seed}" "${gpu_ids}" "${launcher_log}"
fi
