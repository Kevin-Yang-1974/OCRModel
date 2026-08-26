#!/usr/bin/env bash
#SBATCH --job-name=pvld_m4_val_controls
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -euo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
env_dir="${BSCC_GOT_ENV:-${workspace}/envs/got2-py310-cu118}"
model_dir="${BSCC_GOT_MODEL:-${workspace}/models/GOT-OCR2-original}"
dataset_root="${BSCC_MTHV2_PAGE_ROOT:-${workspace}/datasets/MTHv2/converted/mthv2_layout_page_v1}"
formal_run_id="${PVLD_M4_FORMAL_RUN_ID:-mthv2_pvld_m4_predrouting_20260826_v1}"
control_run_id="${PVLD_M4_CONTROL_RUN_ID:?set a new PVLD_M4_CONTROL_RUN_ID}"
selection="${workspace}/evaluation_runs/GOT/${formal_run_id}_p2_validation_selection/selection.json"
output_root="${workspace}/evaluation_runs/GOT/${control_run_id}"
snapshot_root="${workspace}/run_code/${control_run_id}/ocrmodel"

[[ -f "${selection}" ]] || { printf 'ERROR: M4 P2 validation selection is missing.\n' >&2; exit 66; }
[[ -f "${dataset_root}/validation/manifest.jsonl" ]] || {
    printf 'ERROR: MTHv2 validation manifest is missing.\n' >&2
    exit 66
}
[[ ! -e "${output_root}" && ! -e "${snapshot_root}" ]] || {
    printf 'ERROR: control output or code snapshot already exists.\n' >&2
    exit 73
}

gpu_ids="${CUDA_VISIBLE_DEVICES:?Slurm did not assign CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a gpus <<< "${gpu_ids}"
[[ ${#gpus[@]} -eq 3 ]] || {
    printf 'ERROR: M4 validation controls require three assigned GPUs.\n' >&2
    exit 75
}
for gpu in "${gpus[@]}"; do
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
    [[ "${utilization}" =~ ^[0-9]+$ ]] || { printf 'ERROR: invalid GPU utilization.\n' >&2; exit 75; }
    (( utilization < 50 )) || { printf 'ERROR: GPU%s utilization=%s.\n' "${gpu}" "${utilization}" >&2; exit 75; }
done

mkdir -p "${workspace}/evaluation_runs/GOT" "${workspace}/runs" \
    "${workspace}/tmp" "${workspace}/cache/huggingface" \
    "${workspace}/cache/torch_extensions" "$(dirname "${snapshot_root}")"
mkdir "${output_root}" "${snapshot_root}"
cp -a "${workspace}/ocrmodel/src" "${workspace}/ocrmodel/tools" \
    "${workspace}/ocrmodel/config" "${snapshot_root}/"

export TMPDIR="${workspace}/tmp"
export HF_HOME="${workspace}/cache/huggingface"
export TORCH_EXTENSIONS_DIR="${workspace}/cache/torch_extensions"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
source /home/bingxing2/apps/package/pytorch/2.0.1+cu118_cp310/env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u

selected_model="$(python - "${selection}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

selection_path = Path(sys.argv[1]).resolve()
payload = json.loads(selection_path.read_text(encoding="utf-8"))
if payload.get("purpose") != "layout_ablation_validation_selection":
    raise RuntimeError("Unexpected M4 selection purpose")
if payload.get("selection_split") != "validation" or payload.get("test_used_for_selection") is not False:
    raise RuntimeError("M4 checkpoint was not selected using validation only")
selected = payload["selected"]
model = Path(selected["model_path"]).resolve()
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
for name, expected in (("config.json", selected["config_sha256"]), ("model.safetensors", selected["weights_sha256"])):
    digest = sha256(model / name)
    if digest != expected:
        raise RuntimeError(f"Selected M4 checkpoint hash changed: {name}")
print(model)
PY
)"

project_root="${snapshot_root}/src/GOT-OCR-2.0"
evaluator="${project_root}/scripts/evaluate_GOT_layout.py"
controls=(normal alpha_zero shuffled_evidence)
pids=()
for index in "${!controls[@]}"; do
    control="${controls[$index]}"
    control_output="${output_root}/${control}"
    mkdir "${control_output}"
    CUDA_VISIBLE_DEVICES="${gpus[$index]}" python "${evaluator}" \
        --model-name-or-path "${selected_model}" --model-kind pvld \
        --tokenizer-name-or-path "${model_dir}" \
        --layout-manifest "${dataset_root}/validation/manifest.jsonl" \
        --layout-image-root "${dataset_root}/validation" --layout-split validation \
        --output-dir "${control_output}" --max-regions 512 --max-records 0 \
        --batch-size 2 --max-new-tokens 2048 --no-repeat-ngram-size 20 \
        --pvld-routing-control "${control}" --device cuda \
        >"${output_root}/${control}.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
(( failed == 0 )) || { printf 'ERROR: one or more M4 validation controls failed.\n' >&2; exit 1; }

python - "${control_run_id}" "${formal_run_id}" "${selection}" "${output_root}" <<'PY'
import json
import sys
from pathlib import Path

control_run_id, formal_run_id, selection_path, output_root = sys.argv[1:]
root = Path(output_root)
summaries = {
    condition: json.loads(
        (root / condition / "layout_validation_metrics.json").read_text(encoding="utf-8")
    )
    for condition in ("normal", "alpha_zero", "shuffled_evidence")
}
for condition, summary in summaries.items():
    if summary.get("status") != "ok" or summary.get("split") != "validation":
        raise RuntimeError(f"Incomplete validation control: {condition}")
    routing = summary.get("pvld_routing_control", {})
    if routing.get("condition") != condition:
        raise RuntimeError(f"Routing control mismatch: {condition}")
    if summary.get("input_protocol", {}).get("layout_metadata_as_model_input") is not False:
        raise RuntimeError(f"Layout metadata entered model inputs: {condition}")

def metrics(summary):
    ocr = summary["metrics"]["ocr"]
    layout = summary["metrics"]["layout"]
    return {
        "page_cer": ocr["page_cer"],
        "whitespace_normalized_page_cer": ocr["whitespace_normalized_page_cer"],
        "exact_matches": ocr["exact_matches"],
        "pages": ocr["pages"],
        "complete_region_f1": layout["complete_region_f1"],
        "ordered_bbox_mean_iou": layout["ordered_slot_bbox_mean_iou"],
    }

compact = {condition: metrics(summary) for condition, summary in summaries.items()}
normal_cer = compact["normal"]["page_cer"]
payload = {
    "schema_version": 1,
    "status": "ok",
    "purpose": "M4 validation-only routing controls; test not read",
    "control_run_id": control_run_id,
    "formal_run_id": formal_run_id,
    "selection": str(Path(selection_path).resolve()),
    "selection_split": "validation",
    "test_used_for_selection": False,
    "test_manifest_read": False,
    "input_granularity": "whole_page_image",
    "conditions": compact,
    "page_cer_delta_vs_normal": {
        condition: values["page_cer"] - normal_cer
        for condition, values in compact.items()
        if condition != "normal"
    },
}
summary_path = root / "summary.json"
summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"event": "pvld_m4_validation_controls_completed", "summary": str(summary_path), **payload}, separators=(",", ":")))
PY
