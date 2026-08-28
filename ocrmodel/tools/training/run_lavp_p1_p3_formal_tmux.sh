#!/usr/bin/env bash
# Formal new-visual-gradient PVLD pipeline. It never reuses existing run roots.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

session="lavp_p1_p3_formal_20260827_v5"
run_prefix="lavp_p1_p3_formal_20260827_v5"
synthetic_root="${GOT_LAYOUT_DATA}/ancient_photo_diverse_formal_s3s4_dense_20260827_v4"
mthv2_root="/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1"
layout_memory_resolution=64
gpu_utilization_limit=50
p1_steps=12000
p2_steps=30000
p3_steps=8000
checkpoint_steps=2000
seed=42
session_inner=0
confirm_formal_training=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="$2"; shift 2 ;;
        --run-prefix) run_prefix="$2"; shift 2 ;;
        --synthetic-root) synthetic_root="$2"; shift 2 ;;
        --mthv2-root) mthv2_root="$2"; shift 2 ;;
        --layout-memory-resolution) layout_memory_resolution="$2"; shift 2 ;;
        --gpu-utilization-limit) gpu_utilization_limit="$2"; shift 2 ;;
        --p1-steps) p1_steps="$2"; shift 2 ;;
        --p2-steps) p2_steps="$2"; shift 2 ;;
        --p3-steps) p3_steps="$2"; shift 2 ;;
        --checkpoint-steps) checkpoint_steps="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --confirm-formal-training) confirm_formal_training=1; shift ;;
        --session-inner) session_inner=1; shift ;;
        *) printf '{"event":"lavp_formal_pipeline_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

[[ "${session}" =~ ^[A-Za-z0-9_.-]+$ && "${run_prefix}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    printf '%s\n' '{"event":"lavp_formal_pipeline_failed","error":"invalid_session_or_run_prefix"}' >&2; exit 64;
}
[[ "${layout_memory_resolution}" == 16 || "${layout_memory_resolution}" == 64 ]] || {
    printf '%s\n' '{"event":"lavp_formal_pipeline_failed","error":"layout_memory_resolution_must_be_16_or_64"}' >&2; exit 64;
}
for value in "${gpu_utilization_limit}" "${p1_steps}" "${p2_steps}" "${p3_steps}" "${checkpoint_steps}" "${seed}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || { printf '%s\n' '{"event":"lavp_formal_pipeline_failed","error":"numeric_argument_required"}' >&2; exit 64; }
done

python_bin="${OCR_WORKSPACE}/envs/layout-synthesis/bin/python"
got_runner=(bash "${ocrmodel_root}/tools/environment/run_got2.sh")
audit_script="${ocrmodel_root}/tools/preprocessing/audit_synthetic_layout.py"
runner="${ocrmodel_root}/tools/training/run_variable_layout_a100.py"
selector="${ocrmodel_root}/tools/evaluation/select_layout_ablation_checkpoint.py"
tester="${ocrmodel_root}/tools/evaluation/evaluate_layout_ablation_test.py"
project_root="${ocrmodel_root}/src/GOT-OCR-2.0"
log_root="${GOT_TRAINING_RUNS}/${run_prefix}_pipeline_logs"
pipeline_log="${log_root}/${session}.log"

audit_summary="${synthetic_root}/${run_prefix}_full_audit_summary.json"

require_new_outputs() {
    local path
    for path in "$@"; do
        [[ ! -e "${path}" ]] || { printf '{"event":"lavp_formal_pipeline_failed","error":"output_already_exists","path":"%s"}\n' "${path}" >&2; return 1; }
    done
}

manifest_ready() {
    [[ -f "${synthetic_root}/train/manifest.jsonl" && -f "${synthetic_root}/validation/manifest.jsonl" && -f "${synthetic_root}/test/manifest.jsonl" ]]
}

audit_synthetic_data() {
    "${python_bin}" "${audit_script}" \
        --manifest "${synthetic_root}/train/manifest.jsonl" \
        --manifest "${synthetic_root}/validation/manifest.jsonl" \
        --manifest "${synthetic_root}/test/manifest.jsonl" \
        --min-train-high-region-page-fraction 0.50 \
        --summary-json "${audit_summary}"
    "${python_bin}" - "${synthetic_root}/dataset_protocol.json" "${audit_summary}" <<'PY'
import json
import sys
from pathlib import Path

protocol = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
audit = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if protocol.get("status") != "ready":
    raise SystemExit("dataset_protocol status is not ready")
if audit.get("status") != "ok":
    raise SystemExit("full manifest audit did not pass")
if int(protocol.get("train_page_count", 0)) < 10000:
    raise SystemExit("train pages are below the formal minimum")
if audit.get("split_counts", {}).get("train", 0) < 10000:
    raise SystemExit("full audit train page count is below the formal minimum")
if float(audit.get("train_high_region_page_fraction", 0.0)) < 0.50:
    raise SystemExit("full audit train high-region page fraction is below 0.50")
print(json.dumps({"event":"lavp_formal_data_gate_passed", "train_pages": protocol["train_page_count"], "train_regions": protocol["train_region_exposures"], "train_high_region_page_fraction": audit["train_high_region_page_fraction"], "audit": str(Path(sys.argv[2]).resolve())}, separators=(",", ":")))
PY
}

require_mthv2_manifests() {
    local split
    for split in train validation test; do
        [[ -f "${mthv2_root}/${split}/manifest.jsonl" ]] || {
            printf '{"event":"lavp_formal_pipeline_failed","error":"missing_mthv2_manifest","split":"%s"}\n' "${split}" >&2; return 1;
        }
    done
}

eligible_gpus() {
    local rows ids
    rows="$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)"
    ids="$(awk -F, -v limit="${gpu_utilization_limit}" '{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); if ($2 ~ /^[0-9]+$/ && $2 < limit) printf "%s,", $1}' <<<"${rows}" | sed 's/,$//')"
    [[ -n "${ids}" ]] || { printf '{"event":"lavp_formal_pipeline_failed","error":"no_gpu_below_utilization_limit","observed":"%s"}\n' "${rows//$'\n'/;}" >&2; return 1; }
    printf '%s\n' "${ids}"
}

select_checkpoint() {
    local model_root="$1" validation_manifest="$2" selection_root="$3" purpose="$4" max_regions="$5"
    local gpu_ids
    gpu_ids="$(eligible_gpus)"
    "${got_runner[@]}" "${selector}" \
        --ablation vlqa_layout_p1_p2 --model-root "${model_root}" --model-kind pvld \
        --selection-purpose "${purpose}" --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --validation-manifest "${validation_manifest}" --validation-image-root "$(dirname "${validation_manifest}")" \
        --output-dir "${selection_root}" --project-root "${project_root}" \
        --max-regions "${max_regions}" --max-records 0 --max-new-tokens 2048 --no-repeat-ngram-size 20 \
        --parallel-gpu-ids "${gpu_ids}" --gpu-utilization-limit "${gpu_utilization_limit}"
}

run_locked_test() {
    local selection="$1" test_manifest="$2" test_root="$3" category="$4" max_regions="$5"
    local gpu_ids
    gpu_ids="$(eligible_gpus)"
    "${got_runner[@]}" "${tester}" \
        --selection "${selection}" --test-category "${category}" --test-manifest "${test_manifest}" \
        --test-image-root "$(dirname "${test_manifest}")" --model-kind pvld \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" --project-root "${project_root}" \
        --output-dir "${test_root}" --max-regions "${max_regions}" --max-records 0 --max-new-tokens 2048 \
        --no-repeat-ngram-size 20 --parallel-gpu-ids "${gpu_ids}" --gpu-utilization-limit "${gpu_utilization_limit}"
}

run_m4_validation_controls() {
    local group="$1" selection="$2" output_root="$3"
    local selected_model gpu_ids control gpu_id index=0
    [[ ! -e "${output_root}" ]] || { printf '{"event":"lavp_formal_pipeline_failed","error":"m4_control_output_already_exists","path":"%s"}\n' "${output_root}" >&2; return 1; }
    selected_model="$(python3 - "${selection}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

selection = json.load(open(sys.argv[1], encoding="utf-8"))
if selection.get("selection_split") != "validation" or selection.get("test_used_for_selection") is not False:
    raise SystemExit("M4 controls require a validation-only selection")
selected = selection["selected"]
model = Path(selected["model_path"]).resolve()
for filename, expected_key in (("config.json", "config_sha256"), ("model.safetensors", "weights_sha256")):
    digestor = hashlib.sha256()
    with (model / filename).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digestor.update(block)
    digest = digestor.hexdigest()
    if digest != selected.get(expected_key):
        raise SystemExit(f"selected checkpoint hash changed: {filename}")
print(model)
PY
)"
    gpu_ids="$(eligible_gpus)"
    IFS=',' read -r -a control_gpu_ids <<<"${gpu_ids}"
    mkdir -p "${output_root}"
    for control in normal alpha_zero shuffled_evidence; do
        gpu_id="${control_gpu_ids[$((index % ${#control_gpu_ids[@]}))]}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${got_runner[@]}" "${project_root}/scripts/evaluate_GOT_layout.py" \
            --model-name-or-path "${selected_model}" --model-kind pvld \
            --tokenizer-name-or-path "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
            --layout-manifest "${synthetic_root}/validation/manifest.jsonl" \
            --layout-image-root "${synthetic_root}/validation" --layout-split validation \
            --output-dir "${output_root}/${control}" --max-regions 512 --max-records 0 \
            --batch-size 2 --max-new-tokens 2048 --no-repeat-ngram-size 20 \
            --pvld-routing-control "${control}" --device cuda \
            >"${output_root}/${control}.log" 2>&1
        index=$((index + 1))
    done
    python3 - "${group}" "${selection}" "${output_root}" <<'PY'
import json
import sys
from pathlib import Path

group, selection_path, output_root = sys.argv[1:]
root = Path(output_root)
summaries = {
    condition: json.loads((root / condition / "layout_validation_metrics.json").read_text(encoding="utf-8"))
    for condition in ("normal", "alpha_zero", "shuffled_evidence")
}
for condition, summary in summaries.items():
    if summary.get("status") != "ok" or summary.get("split") != "validation":
        raise RuntimeError(f"incomplete M4 validation control: {condition}")
    if summary.get("pvld_routing_control", {}).get("condition") != condition:
        raise RuntimeError(f"M4 validation control provenance mismatch: {condition}")
    if summary.get("input_protocol", {}).get("layout_metadata_as_model_input") is not False:
        raise RuntimeError(f"layout metadata entered model input: {condition}")

metrics = {
    condition: {
        "page_cer": float(summary["metrics"]["ocr"]["page_cer"]),
        "whitespace_normalized_page_cer": float(summary["metrics"]["ocr"]["whitespace_normalized_page_cer"]),
        "exact_matches": int(summary["metrics"]["ocr"]["exact_matches"]),
        "pages": int(summary["metrics"]["ocr"]["pages"]),
    }
    for condition, summary in summaries.items()
}
normal = metrics["normal"]["page_cer"]
supported = normal < metrics["alpha_zero"]["page_cer"] and normal < metrics["shuffled_evidence"]["page_cer"]
payload = {
    "status": "ok" if supported else "routing_effect_not_supported",
    "group": group,
    "purpose": "M4 validation-only routing controls; test not read",
    "selection": str(Path(selection_path).resolve()),
    "selection_split": "validation",
    "test_used_for_selection": False,
    "test_manifest_read": False,
    "conditions": metrics,
    "page_cer_delta_vs_normal": {
        condition: values["page_cer"] - normal
        for condition, values in metrics.items() if condition != "normal"
    },
}
(root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if not supported:
    raise SystemExit("normal routing did not strictly outperform both validation controls")
print(json.dumps({"event": "lavp_m4_validation_controls_passed", **payload}, ensure_ascii=False, separators=(",", ":")))
PY
}

group_flags() {
    local group="$1"
    case "${group}" in
        legacy) printf '%s\n' 'false 1.0 1.0 false' ;;
        m2) printf '%s\n' 'true 1.0 1.0 false' ;;
        m3) printf '%s\n' 'false 0.25 0.25 false' ;;
        m4) printf '%s\n' 'false 0.25 0.25 true' ;;
        all) printf '%s\n' 'true 0.25 0.25 true' ;;
        *) printf '{"event":"lavp_formal_pipeline_failed","error":"unknown_ablation_group","group":"%s"}\n' "${group}" >&2; return 64 ;;
    esac
}

run_group() {
    local group="$1" spatial_memory shared_scale record_scale predicted_routing
    read -r spatial_memory shared_scale record_scale predicted_routing <<<"$(group_flags "${group}")"
    local group_prefix="${run_prefix}_${group}"
    local p1_run_id="${group_prefix}_p1_seed${seed}"
    local p2_run_id="${group_prefix}_p2_seed${seed}"
    local p3_run_id="${group_prefix}_p3_seed${seed}"
    local p1_root="${GOT_TRAINING_RUNS}/${p1_run_id}"
    local p2_root="${GOT_TRAINING_RUNS}/${p2_run_id}"
    local p3_root="${GOT_TRAINING_RUNS}/${p3_run_id}"
    local p1_selection="${p1_root}/p1/validation_selection/selection.json"
    local p2_selection_root="${GOT_EVALUATION_RUNS}/${group_prefix}_p2_validation_selection"
    local p2_selection="${p2_selection_root}/selection.json"
    local m4_controls_root="${GOT_EVALUATION_RUNS}/${group_prefix}_m4_validation_controls"
    local p2_test_root="${GOT_EVALUATION_RUNS}/${group_prefix}_p2_synthetic_id_test"
    local p3_selection_root="${GOT_EVALUATION_RUNS}/${group_prefix}_p3_validation_selection"
    local p3_selection="${p3_selection_root}/selection.json"
    local p3_test_root="${GOT_EVALUATION_RUNS}/${group_prefix}_p3_real_ood_test"
    local runner_flags=(--pvld-shared-gradient-scale "${shared_scale}" --pvld-record-gradient-scale "${record_scale}")
    [[ "${spatial_memory}" == true ]] && runner_flags+=(--pvld-use-spatial-memory)
    [[ "${predicted_routing}" == true ]] && runner_flags+=(--pvld-predicted-layout-routing)

    require_new_outputs "${p1_root}" "${p2_root}" "${p3_root}" "${p2_selection_root}" "${p2_test_root}" "${p3_selection_root}" "${p3_test_root}" "${m4_controls_root}"
    printf '{"event":"lavp_formal_ablation_group_started","group":"%s","run_prefix":"%s","spatial_memory":%s,"shared_gradient_scale":%s,"record_gradient_scale":%s,"predicted_layout_routing":%s}\n' \
        "${group}" "${group_prefix}" "${spatial_memory}" "${shared_scale}" "${record_scale}" "${predicted_routing}"

    "${got_runner[@]}" "${runner}" \
        --dataset-root "${synthetic_root}/train" --manifest "${synthetic_root}/train/manifest.jsonl" \
        --validation-manifest "${synthetic_root}/validation/manifest.jsonl" --test-manifest "${synthetic_root}/test/manifest.jsonl" \
        --source-model "${GOT_SOURCE_MODEL}" --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --stages p1 --ablation vlqa_layout_p1_p2 --layout-loss-preset layout_full \
        --layout-memory-resolution "${layout_memory_resolution}" --max-layout-records 512 \
        --max-layout-tokens 2048 --p1-max-steps "${p1_steps}" --checkpoint-steps "${checkpoint_steps}" --checkpoint-retention 16 \
        --replay-manifest "${mthv2_root}/train/manifest.jsonl" --replay-image-root "${mthv2_root}/train" \
        --gpu-utilization-limit "${gpu_utilization_limit}" --distributed-strategy deepspeed_zero2 --nccl-p2p-disable \
        --max-grad-norm 1.0 \
        --seed "${seed}" --run-id "${p1_run_id}"

    "${got_runner[@]}" "${runner}" \
        --dataset-root "${synthetic_root}/train" --manifest "${synthetic_root}/train/manifest.jsonl" \
        --validation-manifest "${synthetic_root}/validation/manifest.jsonl" --test-manifest "${synthetic_root}/test/manifest.jsonl" \
        --source-model "$(python3 - "${p1_selection}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["model_path"])
PY
)" --source-validation-selection "${p1_selection}" \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" --stages p2 --ablation vlqa_layout_p1_p2 --layout-loss-preset layout_full \
        --layout-memory-resolution "${layout_memory_resolution}" --max-layout-records 512 \
        --max-layout-tokens 2048 --p2-max-steps "${p2_steps}" --checkpoint-steps "${checkpoint_steps}" --checkpoint-retention 16 \
        --gpu-utilization-limit "${gpu_utilization_limit}" --distributed-strategy deepspeed_zero2 --nccl-p2p-disable \
        --max-grad-norm 1.0 \
        --seed "${seed}" --run-id "${p2_run_id}" "${runner_flags[@]}"
    select_checkpoint "${p2_root}/p2/model" "${synthetic_root}/validation/manifest.jsonl" "${p2_selection_root}" ocr 512
    if [[ "${predicted_routing}" == true ]]; then
        run_m4_validation_controls "${group}" "${p2_selection}" "${m4_controls_root}"
    fi

    # The P2-selected checkpoint must initialize P3 before any locked test is
    # executed. This preserves the registered stage-to-stage selection chain.
    "${got_runner[@]}" "${runner}" \
        --dataset-root "${mthv2_root}/train" --manifest "${mthv2_root}/train/manifest.jsonl" \
        --validation-manifest "${mthv2_root}/validation/manifest.jsonl" --test-manifest "${mthv2_root}/test/manifest.jsonl" \
        --source-model "$(python3 - "${p2_selection}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["model_path"])
PY
)" --source-validation-selection "${p2_selection}" \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" --stages p3 --ablation vlqa_layout_p1_p2 --layout-loss-preset layout_full \
        --layout-memory-resolution "${layout_memory_resolution}" --max-layout-records 512 \
        --max-layout-tokens 2048 --p3-max-steps "${p3_steps}" --checkpoint-steps "${checkpoint_steps}" --checkpoint-retention 16 \
        --gpu-utilization-limit "${gpu_utilization_limit}" --distributed-strategy deepspeed_zero2 --nccl-p2p-disable \
        --max-grad-norm 1.0 \
        --seed "${seed}" --run-id "${p3_run_id}" "${runner_flags[@]}"
    select_checkpoint "${p3_root}/p3/model" "${mthv2_root}/validation/manifest.jsonl" "${p3_selection_root}" ocr 512
    run_locked_test "${p2_selection}" "${synthetic_root}/test/manifest.jsonl" "${p2_test_root}" Synthetic-ID 512
    run_locked_test "${p3_selection}" "${mthv2_root}/test/manifest.jsonl" "${p3_test_root}" Real-OOD 512
    printf '{"event":"lavp_formal_ablation_group_completed","group":"%s","p1_selection":"%s","p2_selection":"%s","p2_test":"%s","p3_selection":"%s","p3_test":"%s","test_used_for_selection":false}\n' \
        "${group}" "${p1_selection}" "${p2_selection}" "${p2_test_root}/summary.json" "${p3_selection}" "${p3_test_root}/summary.json"
}

run_pipeline() {
    if ! manifest_ready; then
        printf '{"event":"lavp_formal_data_not_ready","synthetic_root":"%s","required_manifests":["train","validation","test"],"training_started":false}\n' \
            "${synthetic_root}"
        return 0
    fi
    audit_synthetic_data
    if (( ! confirm_formal_training )); then
        printf '{"event":"lavp_formal_data_ready_for_user_confirmation","synthetic_root":"%s","audit":"%s","training_started":false,"required_flag":"--confirm-formal-training","ablation_groups":["legacy","m2","m3","m4","all"]}\n' \
            "${synthetic_root}" "${audit_summary}"
        return 0
    fi
    require_mthv2_manifests
    local group
    for group in legacy m2 m3 m4 all; do
        run_group "${group}"
    done
    printf '{"event":"lavp_p1_p3_formal_pipeline_completed","run_prefix":"%s","ablation_groups":["legacy","m2","m3","m4","all"],"test_used_for_selection":false}\n' "${run_prefix}"
}

if (( session_inner == 1 )); then
    run_pipeline
    exit
fi

[[ -x "${python_bin}" && -x "${OCR_WORKSPACE}/envs/got2/bin/python" ]] || { printf '%s\n' '{"event":"lavp_formal_pipeline_failed","error":"required_python_environment_missing"}' >&2; exit 66; }
if (( ! confirm_formal_training )); then
    # A gate-only audit is deliberately synchronous: it must return its JSON
    # verdict, not leave a completed tmux session that looks like a failure.
    run_pipeline
    exit
fi
command -v tmux >/dev/null 2>&1 || { printf '%s\n' '{"event":"lavp_formal_pipeline_failed","error":"tmux_unavailable"}' >&2; exit 69; }
tmux has-session -t "${session}" 2>/dev/null && { printf '{"event":"lavp_formal_pipeline_failed","error":"tmux_session_exists","session":"%s"}\n' "${session}" >&2; exit 73; }
mkdir -p "${log_root}"
script_path="$(realpath "${BASH_SOURCE[0]}")"
inner_confirm_argument=""
if (( confirm_formal_training )); then
    inner_confirm_argument=" --confirm-formal-training"
fi
tmux new-session -d -s "${session}" \
    "cd '${ocrmodel_root}' && exec bash '${script_path}' --session-inner --session '${session}' --run-prefix '${run_prefix}' --synthetic-root '${synthetic_root}' --mthv2-root '${mthv2_root}' --layout-memory-resolution '${layout_memory_resolution}' --gpu-utilization-limit '${gpu_utilization_limit}' --p1-steps '${p1_steps}' --p2-steps '${p2_steps}' --p3-steps '${p3_steps}' --checkpoint-steps '${checkpoint_steps}' --seed '${seed}'${inner_confirm_argument} >'${pipeline_log}' 2>&1"
sleep 5
tmux has-session -t "${session}" 2>/dev/null || { tail -n 20 "${pipeline_log}" >&2 || true; exit 1; }
printf '{"event":"lavp_p1_p3_formal_pipeline_armed","session":"%s","synthetic_root":"%s","layout_memory_resolution":"%s","confirm_formal_training":%s,"log":"%s"}\n' \
    "${session}" "${synthetic_root}" "${layout_memory_resolution}" "${confirm_formal_training}" "${pipeline_log}"
