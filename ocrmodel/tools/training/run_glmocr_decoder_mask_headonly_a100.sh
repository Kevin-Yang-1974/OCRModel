#!/usr/bin/env bash
# Head-only mask training at 4M resolution: freeze everything, train the head.
#
# Why this is a separate launcher from the fine-grid screen: the regime is
# different in three ways that change the orchestration.
#
#   1. No LoRA is trained, so the backbone builds no autograd graph and the
#      language-model CE is skipped.  That is the only reason a 4M-resolution run
#      fits on a 40GB card at all -- five cards do NOT pool memory, they only run
#      more work at once.
#   2. One arm is split across five cards by page shard (five processes, disjoint
#      pages), then the five heads are averaged back into one.  Arms run in
#      series, so the five cards are busy on one arm at a time.
#   3. There is nothing to compare the learned arms against except the frozen
#      base itself, which is a zero-shot evaluation rather than a training run.
#
# 4M matters because the merge is 2x2 and baked into the checkpoint's conv: the
# only way to give a character more than 2-6 grid cells is to raise the input
# resolution.  At 1M the grid is 64x38 (608 merged cells); at 4M it is 128x78
# (2496), which is the 8-24 cells per character the design is after.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_mask_routing}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dataset_root="${GLMOCR_A100_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1}"

mode="screen"
screen_id="glmocr_decoder_mask_headonly4m_v1"
seed=42
gpu_ids="0,1,2,3,4"
gpu_utilization_limit=50
max_pixels=4000000
max_steps=1024
checkpoint_every=256
ddp_timeout_seconds=3600
# One validation point, not three.  Every shard validates the *whole* validation
# set on its own, so three points would cost fifteen full generations at 4M --
# and the artifact that matters is the merged head, which is scored separately
# afterwards.  Checkpoints are still written every `checkpoint_every` steps, so
# any step can be evaluated later.
validation_steps="1024"
max_eval_new_tokens=1536
train_pages=128
validation_pages=64
# At least one page per shard: the smoke runs the same five-way split as the real
# screen, so it must not leave a shard empty.
smoke_pages=10
smoke_max_steps=4
foreground=0
start_arm="G1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) mode="$2"; shift 2 ;;
        --screen-id) screen_id="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --max-pixels) max_pixels="$2"; shift 2 ;;
        --max-steps) max_steps="$2"; shift 2 ;;
        --ddp-timeout-seconds) ddp_timeout_seconds="$2"; shift 2 ;;
        --train-pages) train_pages="$2"; shift 2 ;;
        --validation-pages) validation_pages="$2"; shift 2 ;;
        --foreground) foreground=1; shift ;;
        --start-arm) start_arm="$2"; shift 2 ;;
        --remote-root) remote_root="$2"; shift 2 ;;
        --code-root) code_root="$2"; shift 2 ;;
        --env-dir) env_dir="$2"; shift 2 ;;
        --model-dir) model_dir="$2"; shift 2 ;;
        --dataset-root) dataset_root="$2"; shift 2 ;;
        *) printf '{"event":"headonly_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done
case "${mode}" in smoke|screen) ;; *) printf '{"event":"headonly_failed","error":"invalid_mode","value":"%s"}\n' "${mode}" >&2; exit 64 ;; esac
case "${start_arm}" in G1|G2|G3) ;; *) printf '{"event":"headonly_failed","error":"invalid_start_arm","value":"%s"}\n' "${start_arm}" >&2; exit 64 ;; esac

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
python="${env_dir}/bin/python"
train_cli="${code_root}/src/layout_ocr/train_decoder_mask.py"
# torchrun rendezvous port; distinct per mode so a smoke can overlap a screen.
master_port="${GLMOCR_MASTER_PORT:-29541}"
char_tool="${code_root}/tools/prepare_mthv2_char_manifest.py"
train_char_manifest="${dataset_root}/train/manifest.char.jsonl"
validation_char_manifest="${dataset_root}/validation/manifest.char.jsonl"
screen_root="${remote_root}/training_runs/${screen_id}"
split_root="${screen_root}/split"
arm_root="${screen_root}/arms"
launcher_log="${remote_root}/runs/${screen_id}.${mode}.launcher.log"

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
    printf '{"status":"%s","screen_id":"%s","mode":"%s","seed":%s,"gpu_ids":"%s"}\n' \
        "$1" "${screen_id}" "${mode}" "${seed}" "${gpu_ids}" > "${screen_root}/status/${mode}.json"
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
    # Each arm changes exactly one variable against G2 (the reference).
    case "$1" in
        G2) printf '%s' "--router-target-mode token" ;;
        G1) printf '%s' "--router-target-mode window" ;;
        G3) printf '%s' "--router-head vae" ;;
        *) printf '{"event":"headonly_failed","error":"invalid_arm","arm":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
}

run_arm() {
    # ONE head trained on all five cards: torchrun starts one process per card,
    # each rank takes a different slice of the pages, and the head's gradients
    # are averaged every step -- so there is a single set of weights, not five
    # that get combined afterwards.
    local arm="$1" train_manifest="$2" validation_manifest="$3" steps="$4" validation_steps="$5"
    local arm_dir="${arm_root}/${arm}"
    mkdir -p "${arm_dir}"
    local routing_args
    routing_args="$(arm_training_args "${arm}")"
    export TMPDIR="${screen_root}/tmp/${mode}/${arm}"
    export HF_HOME="${TMPDIR}/huggingface"
    mkdir -p "${TMPDIR}" "${HF_HOME}"
    printf '{"event":"headonly_arm_launched","screen_id":"%s","arm":"%s","gpus":"%s","world_size":%s}\n' \
        "${screen_id}" "${arm}" "${gpu_ids}" "${#gpu_array[@]}"
    # CUDA_VISIBLE_DEVICES is deliberately NOT set: torchrun assigns LOCAL_RANK
    # and each process selects its own device, so all five cards stay visible to
    # the launcher's own bookkeeping.
    "${env_dir}/bin/torchrun" \
        --nproc_per_node="${#gpu_array[@]}" \
        --master_port="${master_port}" \
        --module layout_ocr.train_decoder_mask \
        --model-path "${model_dir}" \
        --train-manifest "${train_manifest}" \
        --validation-manifest "${validation_manifest}" \
        --output-dir "${arm_dir}" \
        --seed "${seed}" \
        --max-steps "${steps}" \
        --max-pixels "${max_pixels}" \
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
        --router-window-size 3 5 \
        --router-vae-latent-channels 4 --router-vae-latent-size 16 \
        --router-vae-kl-weight 1.0 --router-vae-kl-warmup-steps 200 \
        --router-vae-kl-free-bits 0.05 \
        --learning-rate 1e-6 \
        --router-learning-rate 1e-4 \
        --checkpoint-every "${checkpoint_every}" \
        --validation-every "${checkpoint_every}" \
        --validation-steps ${validation_steps} \
        --ddp-timeout-seconds "${ddp_timeout_seconds}" \
        --max-eval-new-tokens "${max_eval_new_tokens}" \
        ${routing_args} > "${arm_dir}/train.log" 2>&1
    printf '{"event":"headonly_arm_complete","screen_id":"%s","arm":"%s","output_dir":"%s"}\n' \
        "${screen_id}" "${arm}" "${arm_dir}"
}

run_inner() {
    trap 'rc=$?; write_status failed; exit "$rc"' ERR
    [[ -x "${python}" && -f "${train_cli}" && -x "${env_dir}/bin/torchrun" ]] || {
        printf '{"event":"headonly_failed","error":"missing_source"}\n' >&2; exit 66
    }
    declare -A observed=()
    while IFS=',' read -r observed_id utilization; do
        observed_id="${observed_id//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"
        observed[${observed_id}]="${utilization}"
    done < <(nvidia-smi -i "${gpu_ids}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
    for gpu in "${gpu_array[@]}"; do
        [[ -n "${observed[${gpu}]+present}" ]] || {
            printf '{"event":"headonly_failed","error":"gpu_not_reported","gpu":"%s"}\n' "${gpu}" >&2; exit 69; }
        (( observed[${gpu}] < gpu_utilization_limit )) || {
            printf '{"event":"headonly_failed","error":"gpu_admission_failed","gpu":"%s","utilization":%s}\n' \
                "${gpu}" "${observed[${gpu}]}" >&2; exit 75; }
    done
    printf '{"event":"headonly_gpu_admission_ok","gpu_ids":"%s","limit":%s}\n' "${gpu_ids}" "${gpu_utilization_limit}"

    export PYTHONPATH="${code_root}/src:${code_root}"
    export LD_LIBRARY_PATH="${cuda_library_path}"
    export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    # At 4M the eager attention mask is large enough that the allocator's
    # reserved-but-unallocated pool was measured in the tens of GB, which is
    # fragmentation rather than live memory; this keeps the carve-up from
    # stranding blocks a page-boundary allocation then cannot use.
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    # This node's five A100-PCIE cards are a fragmented topology (0-1 NV12,
    # 3-4 NV12, 2 only PIX/SYS, every cross-NUMA link SYS).  Default NCCL hangs
    # on the first all-reduce here -- measured: five ranks enqueued twelve
    # collectives and completed none, until the watchdog fired at ten minutes.
    # Disabling P2P makes the same all-reduce return in milliseconds over shared
    # memory, which is plenty for a ~1M-parameter head.
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export TMPDIR="${screen_root}/tmp/${mode}"
    mkdir -p "${screen_root}/logs" "${screen_root}/status" "${split_root}" "${remote_root}/runs" "${TMPDIR}"
    write_status running
    cd "${code_root}"

    if [[ ! -f "${train_char_manifest}" || ! -f "${validation_char_manifest}" ]]; then
        "${python}" "${char_tool}" --dataset-root "${dataset_root}" --splits train validation \
            > "${screen_root}/logs/char_manifest.log" 2>&1
    fi
    local train_manifest validation_manifest
    if [[ "${mode}" == "smoke" ]]; then
        train_manifest="${split_root}/train${smoke_pages}_smoke_seed${seed}.jsonl"
        validation_manifest="${split_root}/validation${smoke_pages}_smoke_seed${seed}.jsonl"
        [[ -f "${train_manifest}" ]] || prepare_split "${smoke_pages}" "${smoke_pages}" "smoke" > "${split_root}/split_smoke.log"
        run_arm "G3" "${train_manifest}" "${validation_manifest}" "${smoke_max_steps}" "${smoke_max_steps}"
        run_arm "G1" "${train_manifest}" "${validation_manifest}" "${smoke_max_steps}" "${smoke_max_steps}"
    else
        train_manifest="${split_root}/train${train_pages}_screen_seed${seed}.jsonl"
        validation_manifest="${split_root}/validation${validation_pages}_screen_seed${seed}.jsonl"
        [[ -f "${train_manifest}" ]] || prepare_split "${train_pages}" "${validation_pages}" "screen" > "${split_root}/split_screen.log"
        # Series, not parallel: all five cards work on one arm at a time, so each
        # arm gets the full page set rather than a fifth of it.
        case "${start_arm}" in
            G1) arms=(G1 G2 G3) ;;
            G2) arms=(G2 G3) ;;
            G3) arms=(G3) ;;
        esac
        for arm in "${arms[@]}"; do
            run_arm "${arm}" "${train_manifest}" "${validation_manifest}" "${max_steps}" "${validation_steps}"
        done
    fi
    write_status complete
    trap - ERR
    printf '{"event":"headonly_complete","screen_id":"%s","mode":"%s","seed":%s,"test_used_for_selection":false}\n' \
        "${screen_id}" "${mode}" "${seed}"
}

# Same detachment contract as the fine-grid launcher: the caller sets it off
# with setsid and the launcher itself never depends on an interactive session.
if (( foreground == 1 )); then
    run_inner
else
    mkdir -p "${remote_root}/runs"
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    command_line="$(printf '%q ' bash "${script_path}" --foreground --mode "${mode}" --screen-id "${screen_id}" --seed "${seed}" --gpu-ids "${gpu_ids}" --max-pixels "${max_pixels}" --max-steps "${max_steps}" --ddp-timeout-seconds "${ddp_timeout_seconds}" --start-arm "${start_arm}" --train-pages "${train_pages}" --validation-pages "${validation_pages}" --remote-root "${remote_root}" --code-root "${code_root}" --env-dir "${env_dir}" --model-dir "${model_dir}" --dataset-root "${dataset_root}")"
    setsid nohup bash -c "exec ${command_line}" > "${launcher_log}" 2>&1 < /dev/null &
    disown 2>/dev/null || true
    printf '{"event":"headonly_armed","screen_id":"%s","mode":"%s","seed":%s,"gpu_ids":"%s","log":"%s","test_used_for_selection":false}\n' \
        "${screen_id}" "${mode}" "${seed}" "${gpu_ids}" "${launcher_log}"
fi
